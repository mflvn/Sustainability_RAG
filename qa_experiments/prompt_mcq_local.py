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
                        "optionA": {"type": "string", "description": "option A"},
                        "optionB": {"type": "string", "description": "option B"},
                        "optionC": {"type": "string", "description": "option C"},
                        "optionD": {"type": "string", "description": "option D"},
                        "optionE": {"type": "string", "description": "option E"},
                        "answer": {"type": "string", "description": "the correct answer option letter"},
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
                    "required": [ "question", "optionA", "optionB", "optionC", "optionD", "optionE", "answer", "reference_text", "pages"]
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

def generate_qa_for_type(markdown_content: str, industry: str, qa_type: str,temperature:float) -> List[Dict]:

    qa_type_str = "Single Hop" if qa_type == "single_hop" else "Multi Hop"
    qa_pairs = []

    # with open("/homes/ml6823/fyp/Thesis/generate_qa/industry_dictionary.json","r") as file:
    #     industry_dict = json.load(file)
    #     industry_str = industry_dict.get(industry,None)

    prompt = f"""
    Here is the markdown content:
    {markdown_content}

    Based on the markdown content, generate 4 'Single Best Answer' (SBA) questions that have only one correct answer out of five options. The correct answer should not be obvious and *should really require specific information from the source document to be able to be answered*. The incorrect answer options should not be so ridiculous or extreme that they are obviously wrong. The questions must be of the type: {qa_type_str}

    The questions should be ones that could occur when a human wants information from the chatbot. They should be directly relevant to companies preparing their sustainability reports and reflect real-world scenarios that reporting teams might encounter.

    To generate questions, follow these steps:
    1. Select a list of one or more sentences/snippets of the markdown content that can be used to form an answer to a question. This will form the reference text. Remember this should be relevant to the human for drafting sustainability reports.
    2. Write a question that requires the reader to understand the content of the selected text to answer correctly. The question should be based only on the selected text and should not require any additional information. Remember this should be the type of question a human would ask when drafting sustainability reports.
    3. Write five answer options, one of which is correct and the other four are incorrect. The correct answer should complete and taken verbatim from the selected section(s) of the markdown content.

    Generate 4 unique and diverse QA pairs for the specified type and return them using the provided schema."""

    response = client.messages.create(
        model="claude-3-haiku-20240307",
        max_tokens=4096,
        tools=[qa_pair_schema],
        temperature=temperature,
        tool_choice={"type": "tool", "name": "qa_pair_schema"},
        messages=[
                {
                    "role": "system",
                    "content": "You are a sustainability reporting expert that helps companies draft their corporate sustainability reports using the IFRS reporting standards. You are preparing some questions that a company might ask while preparing its sustainability report, for which the answer can be taken from the context in the markdown given.",
                },
                {"role": "user", "content": prompt},
            ],
    )

    try:
        new_qa_pairs = response.content[0].input['qa_pairs']
    
        for qa_pair in new_qa_pairs:
        
            qa_pair['industry'] = industry
            qa_pair['qa_type'] = qa_type
            qa_pair['temperature'] = temperature
            qa_pairs.append(qa_pair)
            if len(qa_pairs) == 8:
                break
    except Exception as e:
        print(f"Error accessing qa_pairs for {qa_type}: {e}")
        print(new_qa_pairs)

    return qa_pairs

def format_previous_questions(qa_pairs: List[Dict]) -> str:
    return "\n".join([f"- {qa['question']}" for qa in qa_pairs])

def process_all_markdowns(main_folder: str, output_folder: str,temperature:float):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    markdown_files = read_markdown_from_folders(main_folder)

    all_qa_pairs = []

    for file in markdown_files:
        print(f"Processing {file['industry']}...")
        qa_pairs = []
        for qa_type in qa_types:
            print(f"Generating questions for {qa_type}...")
            qa_pairs_for_type = generate_qa_for_type(file["content"], file["industry"], qa_type, temperature)
            qa_pairs.extend(qa_pairs_for_type)

        all_qa_pairs.extend(qa_pairs)

        output_file = os.path.join(output_folder, f"{file['industry'][:3]}_prompt_mcq_temp_{temperature}.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(qa_pairs, f, indent=2)

        print(f"Processed {file['industry']} and saved {len(qa_pairs)} QA pairs to {output_file}")

if __name__ == "__main__":
    for temperature in [1,0.5,0.2,0]:
        main_folder = "./qa_experiments/markdowns_test"
        output_folder = "./qa_experiments/prompt_mcq_local"
        process_all_markdowns(main_folder, output_folder,temperature)