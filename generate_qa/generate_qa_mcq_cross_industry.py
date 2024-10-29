import json
import os
import random
from typing import List, Dict

import anthropic
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

# Initialize Anthropic client
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
                        "optionA": {"type": "string", "description": "Option A"},
                        "optionB": {"type": "string", "description": "Option B"},
                        "optionC": {"type": "string", "description": "Option C"},
                        "optionD": {"type": "string", "description": "Option D"},
                        "optionE": {"type": "string", "description": "Option E"},
                        "answer": {"type": "string", "description": "The correct answer option with both the option capital letter and the actual answer"},
                        "explanation": {"type": "string", "description": "Explanation of the correct answer"},
                        "references": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of report names or sources"
                        },
                        "pages": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of page numbers"
                        },
                        "pairing_explanation": {"type": "string", "description": "Explanation of why these industries are paired together"}
                    },
                    "required": ["qa_type", "question", "optionA", "optionB", "optionC", "optionD", "optionE", "answer", "explanation", "references", "pages"]
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
    As a specialist consultant on IFRS sustainability reporting standards, generate six multiple choice question-answer pairs that compares aspects of these industries: {', '.join(industries)}.

    These industries are paired for the following reason:
    {explanation}

    Based on this pairing explanation and the following markdown content, create six questions, two of each type:

    1. Cross-Industry Comparison: Questions that compare and contrast sustainability reporting aspects of different industries.
    2. Metric Comparison: Questions that compare and contrast Sustainability Disclosure Topics & Metrics as well as Activity Metrics from the tables in the documents across multiple industries, as well as questions that ask specific details about the metrics themselves.
    3. Specific Reporting Differences: Questions that compare and contrast what specific topics must be reported between the infustries, and how these reporting standards might differ.

    Here is the markdown content:
    {combined_content}

    The questions should:
    - Require knowledge from both industries to answer correctly
    - Have five answer options, with only one correct answer
    - Be clear, specific, and cover sustainability themes across the selected industries
    - Reflect the reason why these industries are paired, as explained above

    *IMPORTANT* for questions comparing industries, please specify in the question which industries are being compared.

    Generate six unique and diverse QA pairs and return them using the provided schema.
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


# def process_cross_industry_questions(main_folder: str, output_folder: str, industry_pairs_file: str):
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)

#     markdown_files = read_markdown_from_folders(main_folder)
    
#     with open(industry_pairs_file, 'r') as f:
#         industry_pairs = json.load(f)

#     questions = generate_cross_industry_questions(markdown_files, industry_pairs)

#     # Save questions to JSON file
#     output_file = os.path.join(output_folder, "cross_industry_mcq.json")
#     with open(output_file, "w", encoding="utf-8") as f:
#         json.dump(questions, f, indent=2)

#     print(f"Generated {len(questions)} cross-industry questions and saved to {output_file}")

#     # Create a DataFrame and save to CSV
#     df = pd.DataFrame(questions)
#     output_csv = os.path.join(output_folder, "cross_industry_mcq.csv")
#     df.to_csv(output_csv, index=False)
#     print(f"Saved cross-industry questions to {output_csv}")

def process_cross_industry_questions(main_folder: str, output_folder: str, industry_pairs_file: str):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    markdown_files = read_markdown_from_folders(main_folder)
    
    with open(industry_pairs_file, 'r') as f:
        industry_pairs = json.load(f)

    generate_cross_industry_questions(markdown_files, industry_pairs, output_folder)

    print(f"Finished processing all industry pairs.")

if __name__ == "__main__":
    main_folder = "./markdown_output"
    output_folder = "./cross_industry_qa_output"
    industry_pairs_file = "industry_pairs_test.json"
    process_cross_industry_questions(main_folder, output_folder, industry_pairs_file)