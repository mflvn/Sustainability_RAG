import json
import os
from typing import Dict, List

import anthropic
import pandas as pd

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import hashlib


client = anthropic.Anthropic(
    api_key=os.getenv("CLAUDE_API_KEY")
)


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


qa_pair_schema = {
    "name": "qa_pair_schema",
    "description": "Generate multiple choice question-answer pairs from industry markdown",
    "input_schema": {
        "type": "object",
        "properties": {
            "qa_pairs": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "industry": {"type": "string","description": "the industry the question is related to, taken directly from 'Industry:' in the markdown"},
                        "qa_type": {"type": "string", "description": "the assigned type of the question (without the type description)"},
                        "question": {"type": "string", "description": "the question"},
                        "optionA": {"type": "string", "description": "option A"},
                        "optionB": {"type": "string", "description": "option B"},
                        "optionC": {"type": "string", "description": "option C"},
                        "optionD": {"type": "string", "description": "option D"},
                        "optionE": {"type": "string", "description": "option E"},
                        "answer": {"type": "string", "description": "the correct answer option with both the option capital letter and the actual answer"},
                        "reference": {"type": "string", "description": "report name"},
                        "page": {"type": "string", "description": "page number(s)"}
                    },
                    "required": ["industry", "qa_type","question", "optionA", "optionB", "optionC", "optionD", "optionE", "answer", "reference", "page"]
                }
            }
        },
        "required": ["qa_pairs"]
    }
}

# qa_types = [
#     "Factual Retrieval: Questions that require finding and stating specific facts from the document, but without asking what is NOT mentioned or covered in the document.",
#     "Multi-hop Reasoning: Questions that require connecting information from multiple parts of the document.",
#     "Definition or Explanation: Questions asking to define or explain industry-specific terms or concepts.",
#     "Contextual Understanding: Questions that test understanding of the broader context or implications of the information.",
#     "Policy or Procedure Questions: Questions about specific policies, standards, or procedures mentioned in the document.",
#     "Open-ended Analysis: Questions that require synthesizing information to form an opinion or analysis.",
#     "Negative Questions: Questions about what is NOT mentioned or covered in the document.",
#     "Specific Reporting Standards: Questions about specific sustainability reporting standards for the industry.",
#     "Metric Retrieval: Questions about what Sustainability Disclosure Topics & Metrics and Activity Metrics should be reported and what unit of measurement should be used."
# ]
qa_types = [
    "Disclosure Requirements: Questions about specific sustainability disclosure requirements for this industry.",
    "Metric Calculation: Questions about how to calculate or interpret specific sustainability metrics for this industry.",
    "Industry-Specific Challenges: Questions addressing unique sustainability reporting challenges or considerations specific to this industry.",
]


# Load a pre-trained sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

def is_similar(new_question: str, existing_questions: List[str], threshold: float = 0.8) -> bool:
    if not existing_questions:
        return False
    
    new_embedding = model.encode([new_question])
    existing_embeddings = model.encode(existing_questions)
    
    similarities = cosine_similarity(new_embedding, existing_embeddings)[0]
    return np.max(similarities) > threshold

def generate_qa(markdown_content: str, industry: str) -> List[Dict]:
    all_qa_pairs = []
    question_hashes = set()
    
    for qa_type in qa_types:
        qa_pairs_for_type = generate_qa_for_type(markdown_content, industry, qa_type, all_qa_pairs)
        all_qa_pairs.extend(qa_pairs_for_type)
    
    return all_qa_pairs

def generate_qa_for_type(markdown_content: str, industry: str, qa_type: str, existing_qa_pairs: List[Dict]) -> List[Dict]:
    context_window = 3
    existing_questions = [qa['question'] for qa in existing_qa_pairs]
    unique_qa_pairs = []
    max_attempts = 2  # Maximum number of API calls to make
    attempts = 0

    while len(unique_qa_pairs) < 4 and attempts < max_attempts:
        prompt = f"""
        You are a sustainability reporting expert that helps companies draft their corporate sustainability reports using the IFRS reporting standards. You are preparing some questions that a company might ask while preparing it's sustainability report, for which the answer can be taken from the IFRS standards (which are given in the markdown content below). 
        
        Based on the following markdown content from the {industry} industry, generate {4 - len(unique_qa_pairs)} multiple choice question-answer pairs of the type: {qa_type}

        Here is the markdown content for the IFRS documents that set out the requirements for identifying, measuring and disclosing information related to significant climate-related risks and opportunities associated with particular industries:
        {markdown_content}

        Each question must have five answer options, and only one option must be the correct one, all other four options must be wrong. The questions should be the type that could occur when a human wants information from the chatbot.

        The questions should:

        - Be directly relevant to companies preparing their sustainability reports
        - Reflect real-world scenarios that reporting teams might encounter
        - Include specific page numbers in the 'page' field.
        - Have five answer options, with only one correct answer
        - The 'reference' field contains the name of the report or document section.

        To ensure diversity, here are some of the previously generated questions (for context, do not repeat these):
        {format_previous_questions(existing_qa_pairs[-context_window:] + unique_qa_pairs)}

        Generate {4 - len(unique_qa_pairs)} unique and diverse QA pairs for the specified type and return them using the provided schema.
        """

        response = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=4096,
            tools=[qa_pair_schema],
            tool_choice={"type": "tool", "name": "qa_pair_schema"},
            messages=[{"role": "user", "content": prompt}]
        )

        try:
            new_qa_pairs = response.content[0].input['qa_pairs']
        except (IndexError, KeyError, AttributeError) as e:
            print(f"Error accessing qa_pairs for {qa_type}: {e}")
            attempts += 1
            continue

        for qa_pair in new_qa_pairs:
            if not is_similar(qa_pair['question'], existing_questions + [qa['question'] for qa in unique_qa_pairs]):
                qa_pair['industry'] = industry
                qa_pair['qa_type'] = qa_type
                unique_qa_pairs.append(qa_pair)
                existing_questions.append(qa_pair['question'])
                if len(unique_qa_pairs) == 4:
                    break

        attempts += 1

    if len(unique_qa_pairs) < 4:
        print(f"Warning: Only generated {len(unique_qa_pairs)} unique questions for {qa_type} after {attempts} attempts.")

    return unique_qa_pairs

def format_previous_questions(qa_pairs: List[Dict]) -> str:
    return "\n".join([f"- {qa['question']}" for qa in qa_pairs])

def process_all_markdowns(main_folder: str, output_folder: str):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    markdown_files = read_markdown_from_folders(main_folder)

    all_qa_pairs = []

    for file in markdown_files:
        print(f"Processing {file['industry']}...")
        qa_pairs = []
        for qa_type in qa_types:
            print(f"Generating questions for {qa_type}...")
            qa_pairs_for_type = generate_qa_for_type(file["content"], file["industry"], qa_type, qa_pairs)
            qa_pairs.extend(qa_pairs_for_type)

        all_qa_pairs.extend(qa_pairs)

        output_file = os.path.join(output_folder, f"{file['industry']}_qa.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(qa_pairs, f, indent=2)

        print(f"Processed {file['industry']} and saved {len(qa_pairs)} QA pairs to {output_file}")

    # Create a DataFrame from all QA pairs
    df = pd.DataFrame(all_qa_pairs)

    # Save all QA pairs to a single CSV file
    output_csv = os.path.join(output_folder, "all_qa_pairs.csv")
    df.to_csv(output_csv, index=False)
    print(f"Saved all {len(all_qa_pairs)} QA pairs to {output_csv}")

    # Optional: Create separate CSV files for each question type
    for qa_type in qa_types:
        type_df = df[df['qa_type'] == qa_type]
    
    # Group by industry and create separate CSV files
        for industry, industry_df in type_df.groupby('industry'):
            industry_prefix = industry[:3]  # Get first 3 characters of industry name
            type_name = qa_type.split(':')[0].strip()
            type_csv = os.path.join(output_folder, f"{industry_prefix}_{type_name}_qa_pairs.csv")
            industry_df.to_csv(type_csv, index=False)
            print(f"Saved {len(industry_df)} {type_name} QA pairs for {industry} to {type_csv}")

if __name__ == "__main__":
    main_folder = "./markdown_output"
    output_folder = "./mcq_local_output2"
    process_all_markdowns(main_folder, output_folder)