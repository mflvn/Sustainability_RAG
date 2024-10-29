import json
import os
from typing import Dict, List

import numpy as np

# import anthropic
import pandas as pd
from openai import OpenAI

# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# model = SentenceTransformer('all-MiniLM-L6-v2')

cross_industry_qa_schema = {
    "type": "function",
    "function": {
        "name": "cross_industry_qa_schema",
        "description": "Schema for generating cross-industry sustainability reporting multiple-choice questions",
        "parameters": {
            "type": "object",
            "properties": {
                "qa_pairs": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "The question",
                            },
                            "optionA": {"type": "string", "description": "Option A"},
                            "optionB": {"type": "string", "description": "Option B"},
                            "optionC": {"type": "string", "description": "Option C"},
                            "optionD": {"type": "string", "description": "Option D"},
                            "optionE": {"type": "string", "description": "Option E"},
                            "answer": {
                                "type": "string",
                                "description": "the correct answer option letter",
                            },
                            "pages": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of page numbers joined to the first three characters of the industry name the page is found in by an underscore (e.g. [7_b1, 43-46_b7])",
                            },
                        },
                        "required": [
                            "question",
                            "optionA",
                            "optionB",
                            "optionC",
                            "optionD",
                            "optionE",
                            "answer",
                            "pages",
                        ],
                    },
                }
            },
            "required": ["qa_pairs"],
        },
    },
}

qa_types = ["single_hop", "multi_hop"]


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


def generate_qa(
    industries: List[str], markdown_files: List[Dict], temperature: float
) -> List[Dict]:
    all_qa_pairs = []

    for qa_type in qa_types:
        qa_pairs_for_type = generate_qa_for_type(
            industries, markdown_files, qa_type, temperature
        )
        all_qa_pairs.extend(qa_pairs_for_type)

    return all_qa_pairs


def generate_qa_for_type(
    industries: List[str], markdown_files: List[Dict], qa_type: str, temperature: float
) -> List[Dict]:
    combined_content = "\n\n".join(
        [file["content"] for file in markdown_files if file["industry"] in industries]
    )

    qa_type_str = "Single Hop" if qa_type == "single_hop" else "Multi Hop"
    qa_pairs = []

    prompt = f"""
    Here is the markdown content for two industries:
    {combined_content}

    Based on the markdown content, generate 8 'Single Best Answer' (SBA) questions that have only one correct answer out of five options. These questions must be comparative, to compare specific information found in the two industry documents. The correct answer should not be obvious and *should really require specific information from both source documents to be able to be answered*. The incorrect answer options should not be so ridiculous or extreme that they are obviously wrong. The questions must be of the type: {qa_type_str}
    
    The questions should be ones that could occur when a human wants information from the chatbot. They should be directly relevant to companies preparing their sustainability reports and reflect real-world scenarios that reporting teams might encounter.

    Make sure that the correct answer is complete and taken verbatim from the document.

    To generate questions, follow these steps:
    1. Select a list of one or more sentences/snippets of the markdown content that can be used to form an answer to a question. This will form the reference text. Remember this should be relevant to the human for drafting sustainability reports.
    2. Write a question that requires the reader to understand the content of the selected text to answer correctly. The question should be based only on the selected text and should not require any additional information. Remember this should be the type of question a human would ask when drafting sustainability reports.
    3. Write five answer options, one of which is correct and the other four are incorrect. The correct answer should complete and taken verbatim from the selected section(s) of the markdown content.

    Generate 4 unique and diverse QA pairs for the specified type and return them using the provided schema."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        tools=[cross_industry_qa_schema],
        temperature=temperature,
        tool_choice={
            "type": "function",
            "function": {"name": "cross_industry_qa_schema"},
        },
        messages=[
                {
                    "role": "system",
                    "content": "You are a sustainability reporting expert that helps companies draft their corporate sustainability reports using the IFRS reporting standards. You are preparing some questions that a company might ask while preparing its sustainability report, for which the answer can be taken from the context in the markdown given.",
                },
                {"role": "user", "content": prompt},
            ],
    )

    try:
        tool_call = response.choices[0].message.tool_calls[0]
        new_qa_pairs = json.loads(tool_call.function.arguments)["qa_pairs"]

        for qa_pair in new_qa_pairs:
            qa_pair["industries"] = industries
            qa_pair["qa_type"] = qa_type
            qa_pair["temperature"] = temperature
            qa_pairs.append(qa_pair)
            if len(qa_pairs) == 8:
                break
    except Exception as e:
        print(f"Error accessing qa_pairs for {qa_type}: {e}")
        print(new_qa_pairs)

    return qa_pairs


def process_cross_industry_questions(
    main_folder: str, output_folder: str, industry_pairs_file: str, temperature: float
):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    markdown_files = read_markdown_from_folders(main_folder)

    with open(industry_pairs_file, "r") as f:
        industry_pairs = json.load(f)

    all_qa_pairs = []

    for pair in industry_pairs["pairings"]:
        industries = pair["industries"]
        print(f"Generating questions for industries: {', '.join(industries)}")

        qa_pairs = generate_qa(industries, markdown_files, temperature)
        all_qa_pairs.extend(qa_pairs)

        # Save questions for this industry pair
        industry_codes = "".join([industry[:3] for industry in industries])
        output_json = os.path.join(
            output_folder, f"{industry_codes}_mcq_prompt_cross_temp_{temperature}.json"
        )
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(qa_pairs, f, indent=2)

        print(
            f"Generated {len(qa_pairs)} cross-industry questions and saved to {output_json}"
        )


if __name__ == "__main__":
    for temperature in [1, 0.5]:
        main_folder = "./markdowns"
        output_folder = "./qa_experiments/prompt_mcq_cross"
        industry_pairs_file = "./qa_experiments/industry_pairs.json"
        process_cross_industry_questions(
            main_folder, output_folder, industry_pairs_file, temperature
        )
