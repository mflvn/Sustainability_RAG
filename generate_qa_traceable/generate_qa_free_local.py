import json
import os
from typing import Dict, List

import anthropic
import pandas as pd

# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
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
                        "question": {"type": "string", "description": "the question"},
                        "answer": {"type": "string", "description": "the correct answer"},
                        "reference_text": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "the verbatim text taken directly from the report that is used to generate the question and answer"
                        },
                        "pages": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of page numbers"
                        },
                    },
                    "required": [ "question", "answer", "reference_text", "pages"]
                }
            }
        },
        "required": ["qa_pairs"]
    }
}

qa_types = [
    "single_hop",
    "multi_hop"
]

def load_question_structures(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# Load a pre-trained sentence transformer model
# model = SentenceTransformer('all-MiniLM-L6-v2')

# def is_similar(new_question: str, existing_questions: List[str], threshold: float = 0.8) -> bool:
#     if not existing_questions:
#         return False
    
#     new_embedding = model.encode([new_question])
#     existing_embeddings = model.encode(existing_questions)
    
#     similarities = cosine_similarity(new_embedding, existing_embeddings)[0]
#     return np.max(similarities) > threshold

def generate_qa(markdown_content: str, industry: str) -> List[Dict]:
    all_qa_pairs = []
    question_hashes = set()
    
    for qa_type in qa_types:
        qa_pairs_for_type = generate_qa_for_type(markdown_content, industry, qa_type, all_qa_pairs)
        all_qa_pairs.extend(qa_pairs_for_type)
    
    return all_qa_pairs

# The questions must be of the type: {qa_type}
# To ensure diversity, here are some of the previously generated questions (for context, do not repeat these):
        # {format_previous_questions(existing_qa_pairs[-context_window:] + unique_qa_pairs)}
def generate_qa_for_type(markdown_content: str, industry: str, qa_type: str, existing_qa_pairs: List[Dict], question_structures: Dict) -> List[Dict]:
    context_window = 3
    existing_questions = [qa['question'] for qa in existing_qa_pairs]
    unique_qa_pairs = []
    max_attempts = 2  # Maximum number of API calls to make
    attempts = 0

    question_structures = question_structures["Free_local"][qa_type]

    while len(unique_qa_pairs) < 4 and attempts < max_attempts:
        
        question_structures_str = "\n".join([f'"{structure}"' for structure in question_structures])
        
        with open("./generate_qa/industry_dictionary.json","r") as file:
            industry_dict = json.load(file)
            industry_str = industry_dict.get(industry,None)

        prompt = f"""
        You are a sustainability reporting expert that helps companies draft their corporate sustainability reports using the IFRS reporting standards. You are preparing some questions that a company might ask while preparing its sustainability report, for which the answer can be taken from the context given in the markdown below. 
        
        Based on the following markdown content, generate {4 - len(unique_qa_pairs)} questions with a free-text answer. The answer *should really require specific information from the source document to be able to be answered*. The questions must be of the type: {qa_type}

        Here is the markdown content:
        {markdown_content}

        The questions should be ones that could occur when a human wants information from the chatbot. They should be directly relevant to companies preparing their sustainability reports and reflect real-world scenarios that reporting teams might encounter.

        Some example {qa_type} question structures are shown below. Please choose {4 - len(unique_qa_pairs)} structures at random but do not limit yourself to these types only. If some of these structures do not make sense given the content of the document, adapt them to the context as appropriate or choose ones that you think are appropriate.

        Here are some question structures:
        {question_structures_str}

        When asking about a specific thing from the document (such as what a company must do for xx), make sure that the correct answer is complete and taken verbatim from the document.

        Generate {4 - len(unique_qa_pairs)} unique and diverse QA pairs for the specified type and return them using the provided schema.
        """

        response = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=4096,
            tools=[qa_pair_schema],
            temperature=0.2,
            tool_choice={"type": "tool", "name": "qa_pair_schema"},
            messages=[{"role": "user", "content": prompt}]
        )

        try:
            new_qa_pairs = response.content[0].input['qa_pairs']
       
            for qa_pair in new_qa_pairs:
           
                # if not is_similar(qa_pair['question'], existing_questions + [qa['question'] for qa in unique_qa_pairs]):
                qa_pair['industry'] = industry
                qa_pair['qa_type'] = qa_type
                unique_qa_pairs.append(qa_pair)
                existing_questions.append(qa_pair['question'])
                if len(unique_qa_pairs) == 4:
                    break
        except Exception as e:
            print(f"Error accessing qa_pairs for {qa_type}: {e}")
            print(new_qa_pairs)

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
    question_structures = load_question_structures('./generate_qa_traceable/question_structures.json')

    all_qa_pairs = []

    for file in markdown_files:
        print(f"Processing {file['industry']}...")
        qa_pairs = []
        for qa_type in qa_types:
            print(f"Generating questions for {qa_type}...")
            qa_pairs_for_type = generate_qa_for_type(file["content"], file["industry"], qa_type, qa_pairs, question_structures)
            qa_pairs.extend(qa_pairs_for_type)

        all_qa_pairs.extend(qa_pairs)

        output_file = os.path.join(output_folder, f"{file['industry']}_qa.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(qa_pairs, f, indent=2)

        print(f"Processed {file['industry']} and saved {len(qa_pairs)} QA pairs to {output_file}")

if __name__ == "__main__":
    main_folder = "./generate_qa_traceable/markdowns_test"
    output_folder = "./generate_qa_traceable/free_local_output_traceable"
    process_all_markdowns(main_folder, output_folder)