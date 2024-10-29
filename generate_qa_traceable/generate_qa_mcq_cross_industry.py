import json
import os
from typing import Dict, List

import anthropic
import numpy as np
import pandas as pd

# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity

client = anthropic.Anthropic(
    api_key=os.getenv("CLAUDE_API_KEY")
)

# model = SentenceTransformer('all-MiniLM-L6-v2')

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
                        "question": {"type": "string", "description": "The question"},
                        "optionA": {"type": "string", "description": "Option A"},
                        "optionB": {"type": "string", "description": "Option B"},
                        "optionC": {"type": "string", "description": "Option C"},
                        "optionD": {"type": "string", "description": "Option D"},
                        "optionE": {"type": "string", "description": "Option E"},
                        "answer": {
                            "type": "string",
                            "description": "the correct answer option letter",
                        },
                        "reference_text": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "the verbatim reference text taken directly from the report that is used to generate the question and answer",
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
                        "reference_text",
                        "pages",
                    ],
                },
            }
        },
        "required": ["qa_pairs"],
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


def load_question_structures(file_path):
    with open(file_path, "r") as file:
        return json.load(file)


# def is_similar(
#     new_question: str, existing_questions: List[str], threshold: float = 0.8
# ) -> bool:
#     if not existing_questions:
#         return False

#     new_embedding = model.encode([new_question])
#     existing_embeddings = model.encode(existing_questions)

#     similarities = cosine_similarity(new_embedding, existing_embeddings)[0]
#     return np.max(similarities) > threshold


def generate_qa(
    industries: List[str], markdown_files: List[Dict], explanation: str
) -> List[Dict]:
    all_qa_pairs = []

    for qa_type in qa_types:
        qa_pairs_for_type = generate_qa_for_type(
            industries, markdown_files, qa_type, all_qa_pairs, explanation
        )
        all_qa_pairs.extend(qa_pairs_for_type)

    return all_qa_pairs


def generate_qa_for_type(
    industries: List[str],
    markdown_files: List[Dict],
    qa_type: str,
    existing_qa_pairs: List[Dict],
    explanation: str,
) -> List[Dict]:
    combined_content = "\n\n".join(
        [file["content"] for file in markdown_files if file["industry"] in industries]
    )
    context_window = 3
    unique_qa_pairs = []
    max_attempts = 2
    attempts = 0

    question_structures = load_question_structures(
        "./generate_qa_traceable/question_structures.json"
    )
    question_structures_str = "\n".join(
        [
            f'"{structure}"'
            for structure in question_structures["Cross-industry"][qa_type]
        ]
    )

    while len(unique_qa_pairs) < 4 and attempts < max_attempts:
        with open(
            "./generate_qa/industry_dictionary.json", "r"
        ) as file:
            industry_dict = json.load(file)
            industry_names = [
                industry_dict.get(industry, None) for industry in industries
            ]
        prompt = f"""
        You are a sustainability reporting expert that helps companies draft their corporate sustainability reports using the IFRS reporting standards. You are preparing some questions that a company might ask while preparing its sustainability report, for which the answer can be taken from the context given in the markdown below. 

        Based on the following markdown content for the industries {', '.join(industry_names)}, generate {4 - len(unique_qa_pairs)} 'Single Best Answer' (SBA) questions that have only one correct answer out of five options. The correct answer should not be obvious and *should really require specific information from the source document to be able to be answered*. The incorrect answer options should not be so ridiculous or extreme that they are obviously wrong. The questions must include all the industries you have been given, and must be of the type: {qa_type}

        Here is the markdown content:
        {combined_content}

        The questions should include all {len(industries)} given industries ({', '.join(industry_names)}) and should be ones that could occur when a human wants information from the chatbot. They should be directly relevant to companies preparing their sustainability reports and reflect real-world scenarios that reporting teams might encounter.

        Some example {qa_type} question structures are shown below. Please choose structures at random but do not limit yourself to these types only. If some of these structures do not make sense given the content of the document, adapt them to the context as appropriate or choose ones that you think are appropriate.
 

        Be specific. When asking about a specific thing from the documents, make sure that the correct answer is complete and taken verbatim from the document. For questions comparing industries, specify which industries are being compared in the question text.

        Generate {4 - len(unique_qa_pairs)} unique and diverse QA pairs for the specified type and return them using the provided schema. VERY IMPORTANT: DO NOT MENTION EXPLICITLY ANY INDUSTRY NAMES IN THE QUESTIONS, rather than a general reference to the industry type. example groups INDUSTRY_GROUPS = 
    "Consumer Goods": ["b1-apparel-accessories-and-footwear", "b2-appliance-manufacturing", "b3-building-products-and-furnishings", "b4-e-commerce", "b5-household-and-personal-products", "b6-multiline-and-specialty-retailers-and-distributors"],
    "Extractives and Minerals Processing": ["b7-coal-operations", "b8-construction-materials", "b9-iron-and-steel-producers", "b10-metals-and-mining", "b11-oil-and-gas-exploration-and-production", "b12-oil-and-gas-midstream", "b13-oil-and-gas-refining-and-marketing", "b14-oil-and-gas-services"],
    "Financials": ["b15-asset-management-and-custody-activities", "b16-commercial-banks", "b17-insurance", "b18-investment-banking-and-brokerage", "b19-mortgage-finance"]. Do not use this, rather be creative and invent a name for the {', '.join(industry_names)} industries in the question 
 

        """

        response = client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=4096,
            tools=[cross_industry_qa_schema],
            temperature=0.2,
            tool_choice={"type": "tool", "name": "cross_industry_qa_schema"},
            messages=[{"role": "user", "content": prompt}],
        )

        try:
            new_qa_pairs = response.content[0].input["qa_pairs"]
            existing_questions = [
                qa["question"] for qa in existing_qa_pairs + unique_qa_pairs
            ]

            for qa_pair in new_qa_pairs:
                # if not is_similar(qa_pair['question'], existing_questions):
                qa_pair["industries"] = industries
                qa_pair["pairing_explanation"] = explanation
                qa_pair["qa_type"] = qa_type
                unique_qa_pairs.append(qa_pair)
                existing_questions.append(qa_pair["question"])
                if len(unique_qa_pairs) == 4:
                    break
        except Exception as e:
            print(f"Error accessing qa_pairs for {qa_type}: {e}")
            print(new_qa_pairs)

        attempts += 1

    if len(unique_qa_pairs) < 4:
        print(
            f"Warning: Only generated {len(unique_qa_pairs)} unique questions for {qa_type} after {attempts} attempts."
        )

    return unique_qa_pairs


def process_cross_industry_questions(
    main_folder: str, output_folder: str, industry_pairs_file: str
):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    markdown_files = read_markdown_from_folders(main_folder)

    with open(industry_pairs_file, "r") as f:
        industry_pairs = json.load(f)

    all_qa_pairs = []
    
    for pair in industry_pairs["pairings"][:1]:
        industries = pair["industries"]
        explanation = pair["explanation"]
        print(f"Generating questions for industries: {', '.join(industries)}")

        qa_pairs = generate_qa(industries, markdown_files, explanation)
        all_qa_pairs.extend(qa_pairs)

        # Save questions for this industry pair
        industry_codes = "".join([industry[:3] for industry in industries])
        output_json = os.path.join(
            output_folder, f"cross_industry_mcq_{industry_codes}.json"
        )
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(qa_pairs, f, indent=2)

        # Save to CSV
        df = pd.DataFrame(qa_pairs)
        output_csv = os.path.join(
            output_folder, f"cross_industry_mcq_{industry_codes}.csv"
        )
        df.to_csv(output_csv, index=False)

        print(
            f"Generated {len(qa_pairs)} cross-industry questions and saved to {output_json} and {output_csv}"
        )

    print("Finished processing all industry pairs.")


if __name__ == "__main__":
    main_folder = "./markdowns"
    output_folder = "./generate_qa_traceable/mcq_cross_industry_output_traceable"
    industry_pairs_file = (
        "./generate_qa/industry_pairs_test2.json"
    )
    process_cross_industry_questions(main_folder, output_folder, industry_pairs_file)
