import json
import os
import random
from typing import List, Dict

import anthropic
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd


client = anthropic.Anthropic(
    api_key=os.getenv("CLAUDE_API_KEY")
)

# Initialize SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define schema for LLM response
cross_industry_qa_schema = {
    "name": "cross_industry_qa_schema",
    "description": "Schema for generating global sustainability reporting multiple-choice questions",
    "input_schema": {
        "type": "object",
        "properties": {
            "qa_pairs": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "qa_type": {"type": "string", "description": "The assigned type of the question (without the type description)"},
                        "question": {"type": "string", "description": "The question"},
                        "answer": {"type": "string", "description": "The correct answer"},
                        "explanation": {"type": "string", "description": "Explanation of the correct answer with specific detail from the report(s) being referenced"},
                        "references": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of report names or sources"
                        },
                        "reference_text": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "the verbatim text taken directly from the report(s) that is used to generate the question and answer"
                        },
                        "pages": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of page numbers"
                        },
                        "pairing_explanation": {"type": "string", "description": "Explanation of why these industries are paired together"}
                    },
                    "required": ["qa_type", "question", "answer", "explanation", "references", "reference_text", "pages"]
                }
            }
        },
        "required": ["qa_pairs"]
    }
}

# Based on this pairing explanation and the following markdown content, create three questions that both highlight the sustainability reporting comparisons, challenges, or trends between these industries:

def generate_question(industries: List[str], markdown_files: List[Dict], existing_questions: List[str], explanation: str) -> Dict:
    combined_content = "\n\n".join([file['content'] for file in markdown_files if file['industry'] in industries])
    
    prompt = f"""
    You are a sustainability reporting consultant that helps companies draft their corporate sustainability reports using the IFRS reporting standards. You are preparing some questions that a company might ask while preparing it's sustainability report, for which the answer can be taken from the IFRS standards (which are given in the markdown content below). 
    
    Generate six free-text question-answer pairs related to sustainability reporting for the following industries: {', '.join(industries)}.

    These industries are paired for the following reason:
    {explanation}

    Here is the markdown content for the IFRS documents that set out the requirements for identifying, measuring and disclosing information related to significant climate-related risks and opportunities associated with particular industries:
    {combined_content}

    Based on this pairing explanation and the provided markdown content, create six questions, two of each type:

    1. Reporting Requirements: Questions about specific disclosure requirements, focusing on what companies in these industries must report according to IFRS standards.
    2. Metric Selection and Calculation: Questions about choosing appropriate metrics, calculating them correctly, and understanding their significance in the context of these industries.
    3. Industry-Specific Challenges: Questions addressing unique sustainability reporting challenges or considerations specific to these paired industries.

    The questions you generate should:

    - Be directly relevant to companies preparing their sustainability reports
    - Focus on practical aspects of report preparation, data collection, and compliance
    - Reflect real-world scenarios that reporting teams might encounter
    - Require understanding of both industries to answer correctly
    - Be clear, specific, and aligned with current IFRS sustainability reporting standards

    For questions comparing industries, specify which industries are being compared in the question text.

    Generate six unique and diverse multiple-choice QA pairs that a company's sustainability reporting team would find valuable when preparing their reports, and return them using the provided schema.
    """

    response = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=4096,
        tools=[cross_industry_qa_schema],
        tool_choice={"type": "tool", "name": "cross_industry_qa_schema"},
        messages=[{"role": "user", "content": prompt}]
    )

    try:
        qa_pairs = response.content[0].input['qa_pairs']
        for qa_pair in qa_pairs:
            qa_pair['industries'] = industries
            qa_pair['pairing_explanation'] = explanation
        return qa_pairs
    except (IndexError, KeyError, AttributeError) as e:
        print(f"Error generating questions: {e}")
        return None

def is_question_unique(new_question: str, existing_questions: List[str], threshold: float = 0.8) -> bool:
    if not existing_questions:
        return True
    new_embedding = model.encode([new_question])[0]
    existing_embeddings = model.encode(existing_questions)
    similarities = cosine_similarity([new_embedding], existing_embeddings)[0]
    return max(similarities) < threshold

def save_non_unique_questions(questions, industries, output_folder="cross_industry_qa_output"):
    industry_codes = ''.join([industry[:3] for industry in industries])
    output_file = os.path.join(output_folder, f"cross_industry_mcq_non_unique_{industry_codes}.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=2)
    print(f"Saved {len(questions)} non-unique questions to {output_file}")

def read_markdown_from_folders(main_folder: str) -> List[Dict[str, str]]:
    markdown_files = []
    for industry_folder in os.listdir(main_folder):
        folder_path = os.path.join(main_folder, industry_folder)
        if os.path.isdir(folder_path):
            file_path = os.path.join(folder_path, "full_content.md")
            if os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as file:
                    content = file.read()
                    markdown_files.append(
                        {"industry": industry_folder, "content": content}
                    )
    return markdown_files

def generate_cross_industry_questions(markdown_files: List[Dict], industry_pairs: List[Dict], output_folder: str) -> None:
    for pair in industry_pairs['pairings']:
        industries = pair['industries']
        explanation = pair['explanation']
        print(f"Generating questions for industries: {', '.join(industries)}")
        
        questions = []
        existing_questions = []
        non_unique_questions = []
        
        qa_pairs = generate_question(industries, markdown_files, existing_questions, explanation)
        if qa_pairs:
            for qa_pair in qa_pairs:
                questions.append(qa_pair)
                existing_questions.append(qa_pair['question'])
                # if is_question_unique(qa_pair['question'], existing_questions):
                #     questions.append(qa_pair)
                #     existing_questions.append(qa_pair['question'])
                # else:
                #     print(f"Found non-unique question: {qa_pair['question'][:50]}...")
                #     non_unique_questions.append(qa_pair)

        # Save unique questions
        industry_codes = ''.join([industry[:3] for industry in industries])
        output_json = os.path.join(output_folder, f"cross_industry_mcq_{industry_codes}.json")
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(questions, f, indent=2)
        
        # Save to CSV
        df = pd.DataFrame(questions)
        output_csv = os.path.join(output_folder, f"cross_industry_mcq_{industry_codes}.csv")
        df.to_csv(output_csv, index=False)
        
        print(f"Generated {len(questions)} cross-industry questions and saved to {output_json} and {output_csv}")

        # Save non-unique questions
        if non_unique_questions:
            save_non_unique_questions(non_unique_questions, industries)


def process_cross_industry_questions(main_folder: str, output_folder: str, industry_pairs_file: str):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    markdown_files = read_markdown_from_folders(main_folder)
    
    with open(industry_pairs_file, 'r') as f:
        industry_pairs = json.load(f)

    generate_cross_industry_questions(markdown_files, industry_pairs, output_folder)

    print(f"Finished processing all industry pairs.")

if __name__ == "__main__":
    main_folder = "/homes/ml6823/fyp/Thesis/markdowns"
    output_folder = "/homes/ml6823/fyp/Thesis/qa_output/cross_industry_qa_free_output"
    industry_pairs_file = "/homes/ml6823/fyp/Thesis/generate_qa/industry_pairs_test.json"
    process_cross_industry_questions(main_folder, output_folder, industry_pairs_file)