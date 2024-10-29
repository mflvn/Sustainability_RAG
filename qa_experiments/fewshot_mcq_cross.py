import json
import os
from typing import Dict, List, Optional

from openai import OpenAI
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)


class QAPair(BaseModel):
    question: str
    optionA: str
    optionB: str
    optionC: str
    optionD: str
    optionE: str
    answer: str
    reference_text: List[str]
    pages: List[str]
    industries: List[str]
    qa_type: str
    temperature: float


class QAPairResponse(BaseModel):
    qa_pairs: List[QAPair]


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


def load_question_structures(file_path: str) -> Dict:
    with open(file_path, "r") as file:
        return json.load(file)


@retry(
    stop=stop_after_attempt(10),
    wait=wait_exponential(multiplier=1, min=1, max=5),
    retry=retry_if_exception_type(Exception),
)
def generate_qa_for_type(
    industries: List[str],
    markdown_files: List[Dict],
    qa_type: str,
    temperature: float,
    question_structures: Dict,
) -> List[QAPair]:
    combined_content = "\n\n".join(
        [file["content"] for file in markdown_files if file["industry"] in industries]
    )

    qa_type_str = "Single Hop" if qa_type == "single_hop" else "Multi Hop"

    question_structures = question_structures["Cross-industry"][qa_type]
    question_structures_str = "\n".join(
        [f'"{structure}"' for structure in question_structures]
    )

    with open("./generate_qa/industry_dictionary.json", "r") as file:
        industry_dict = json.load(file)
        industry_str = ", ".join([industry_dict.get(ind, ind) for ind in industries])

    prompt = f"""
    Here is the markdown content for two industries:
    {combined_content}

    Based on the markdown content from the {industry_str} industries, generate 2 'Single Best Answer' (SBA) questions that have only one correct answer out of five options. These questions must be comparative, to compare specific information found in the two industry documents. The correct answer should not be obvious and *should really require specific information from both source documents to be able to be answered*. The incorrect answer options should not be so ridiculous or extreme that they are obviously wrong. The questions must be of the type: {qa_type_str}
    
    The questions should include all {len(industries)} given industries ({industry_str}). The questions should be ones that could occur when a human wants information from the chatbot. They should be directly relevant to companies preparing their sustainability reports and reflect real-world scenarios that reporting teams might encounter.
    To generate questions, follow these steps:
    1. Select a list of one or more sentences/snippets of the markdown content that can be used to form an answer to a question. This will form the reference text. Remember this should be relevant to the human for drafting sustainability reports.
    2. Write a question that requires the reader to understand the content of the selected text to answer correctly. The question should be based only on the selected text and should not require any additional information. Remember this should be the type of question a human would ask when drafting sustainability reports.
    3. Write five answer options, one of which is correct and the other four are incorrect. The correct answer should complete and taken verbatim from the selected section(s) of the markdown content.

    Some example {qa_type} question structures are shown below. Please choose 2 structures at random but do not limit yourself to these types only. If some of these structures do not make sense given the content of the document, adapt them to the context as appropriate or choose ones that you think are appropriate.

    Here are some question structures:
    {question_structures_str}  

    VERY IMPORTANT: The correct reference_text MUST BE COMPLETE and taken VERBATIM from the document. Do not paraphrase unless absolutely necessary. The answer should only be a letter from A to E.

    Generate 2 unique and diverse QA pairs for the specified type and return them using the provided schema.
    """

    try:
        response = client.beta.chat.completions.parse(
            model="gpt-4o-2024-08-06",
            response_format=QAPairResponse,
            messages=[
                {
                    "role": "system",
                    "content": "You are a sustainability reporting expert that helps companies draft their corporate sustainability reports using the IFRS reporting standards. You are preparing some questions that a company might ask while preparing its sustainability report, for which the answer can be taken from the context in the markdown given.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
        )

        qa_pairs = response.choices[0].message.parsed.qa_pairs

        for qa_pair in qa_pairs:
            qa_pair.industries = industries
            qa_pair.qa_type = qa_type
            qa_pair.temperature = temperature

        return qa_pairs[:2]

    except Exception as e:
        print(f"Error generating QA pairs for {qa_type}: {e}")
        raise


def process_cross_industry_questions(
    main_folder: str, output_folder: str, industry_pairs_file: str, temperature: float
):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    markdown_files = read_markdown_from_folders(main_folder)
    question_structures = load_question_structures(
        "./generate_qa_traceable/question_structures.json"
    )

    with open(industry_pairs_file, "r") as f:
        industry_pairs = json.load(f)

    all_qa_pairs = []

    for pair in industry_pairs["pairings"][0:3]:
        industries = pair["industries"]
        print(f"Generating questions for industries: {', '.join(industries)}")

        qa_pairs = []
        for qa_type in ["single_hop", "multi_hop"]:
            qa_pairs.extend(
                generate_qa_for_type(
                    industries,
                    markdown_files,
                    qa_type,
                    temperature,
                    question_structures,
                )
            )

        all_qa_pairs.extend(qa_pairs)

        # Save questions for this industry pair
        industry_codes = "".join([industry[:3] for industry in industries])
        output_json = os.path.join(
            output_folder, f"{industry_codes}_mcq_fewshot_cross_temp_{temperature}.json"
        )
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump([qa_pair.dict() for qa_pair in qa_pairs], f, indent=2)

        print(
            f"Generated {len(qa_pairs)} cross-industry questions and saved to {output_json}"
        )


if __name__ == "__main__":
    for temperature in [0,0.2,0.5,1.0]:
        main_folder = "./markdowns"
        output_folder = "./final/fewshot_mcq_cross"
        industry_pairs_file = "./generate_qa/industry_pairs.json"
        process_cross_industry_questions(
            main_folder, output_folder, industry_pairs_file, temperature
        )
