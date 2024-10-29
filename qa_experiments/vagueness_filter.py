import json
from typing import List, Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential
from openai import OpenAI
from pydantic import BaseModel, Field
from tqdm import tqdm
import os

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)

class VaguifierResponse(BaseModel):
    vague_question: str = Field(..., description="The refined version of the original question")

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def make_question_vague(question: str, industry: str) -> VaguifierResponse:
    prompt = f"""
    Refine the following question by ONLY slightly generalizing the name of the industry. 
    The question is from the {industry} industry. 
    Keep the entire question exactly the same, only modifying the industry name to be slightly more general.

    Original question: {question}

    Guidelines:
    1. ONLY change the specific industry name to a slightly more general term.
    2. Keep ALL other parts of the question, including technical terms, metrics, and structure, exactly the same.
    3. The refined question should be identical to the original except for the industry name.
    4. Make the change in industry name as minimal as possible while still generalizing slightly.

    Examples:
    1. Original: What is the code for the 'Gross global Scope 1 emissions, percentage covered under emissions-limiting regulations' metric in the Coal Operations industry?
       Refined: What is the code for the 'Gross global Scope 1 emissions, percentage covered under emissions-limiting regulations' metric in the industry about operating coal?

    2. Original: In the Apparel, Accessories & Footwear industry, what percentage of raw materials should be third-party certified to environmental or social sustainability standards?
       Refined: For clothing companies, what percentage of raw materials should be third-party certified to environmental or social sustainability standards?

    3. Original: What is the reporting metric for water consumption in Oil & Gas Exploration & Production operations?
       Refined: What is the reporting metric for water consumption in fossil fuel extraction operations?

    4. Original: For Electric Utilities, what is the RIF (Recordable Incident Frequency) safety performance indicator?
       Refined: For power utility companies, what is the RIF (Recordable Incident Frequency) safety performance indicator?

    Provide only the revised question without any explanation.
    """

    response = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": "You are an expert at rephrasing questions to make them slightly more vague while maintaining their core meaning."},
            {"role": "user", "content": prompt}
        ],
        response_format=VaguifierResponse,
        temperature=0.1,
    )
    refined = response.choices[0].message.parsed
    return refined.vague_question
 
def process_questions(input_file: str, output_file: str):
    with open(input_file, 'r') as f:
        questions: List[Dict[str, Any]] = json.load(f)

    for question in tqdm(questions):
        original_question = question['question']
        industry_code = question['industry']

        try:
            vague_response = make_question_vague(original_question, industry_code)
            question['refined_question'] = vague_response
        except Exception as e:
            print(f"An error occurred processing question: {original_question}")
            print(f"Error: {e}")
            question['refined_question'] = "Error: Unable to generate refined question"

    with open(output_file, 'w') as f:
        json.dump(questions, f, indent=2)

if __name__ == "__main__":
    input_file = "./qa_experiments/fewshot_mcq_local/b16_fewshot_mcq_temp_0.2.json"
    output_file = "./qa_experiments/fewshot_mcq_local/refined_b16_fewshot_mcq_temp_0.2.json"
    process_questions(input_file, output_file)
    print(f"Processed questions saved to {output_file}")