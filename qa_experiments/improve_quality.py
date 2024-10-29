import json
import os
from typing import List, Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential
from openai import OpenAI
from pydantic import BaseModel, Field
from tqdm import tqdm

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)

class ImprovedQuestionResponse(BaseModel):
    improved_question: str = Field(..., description="The improved version of the original question")
    improved_metric: str = Field(..., description="The metric that was improved")

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

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def improve_question(question: Dict[str, Any], metric_to_improve: str, markdown_content: List[Dict[str, str]]) -> ImprovedQuestionResponse:
    relevant_content = "\n\n".join([
        f"Industry: {md['industry']}\n{md['content'][:500]}..." 
        for md in markdown_content 
        if md['industry'] in question['industries']
    ])

    prompt = f"""
    Improve the following question by focusing on the {metric_to_improve} metric. 
    The question is from the {', '.join(question['industries'])} industry/industries.

    Original question: {question['question']}

    Current metrics:
    Faithfulness: {question['faithfulness_score']}
    Relevancy: {question.get(f"relevancy_score_{question['industries'][0]}", 'N/A')}
    Specificity: {question['specificity_score']}

    Relevant industry content:
    {relevant_content}

    Guidelines for improvement:
    1. If improving faithfulness: Ensure accuracy, verify metrics and concepts, remove misleading information.
    2. If improving relevancy: Focus on industry-specific aspects, use appropriate terminology.
    3. If improving specificity: Add precise details, use exact metric names or values, narrow the scope if needed.

    Maintain the original structure, intent, and difficulty level of the question.
    If it's a multi-choice question, preserve that format.

    Provide only the improved question without any explanation.
    """

    response = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": "You are an expert at improving questions while maintaining their core meaning and structure."},
            {"role": "user", "content": prompt}
        ],
        response_format=ImprovedQuestionResponse,
        temperature=0.1,
    )
    return response.choices[0].message.parsed

def process_questions(input_file: str, output_file: str, markdown_folder: str):
    with open(input_file, 'r') as f:
        questions: List[Dict[str, Any]] = json.load(f)

    markdown_content = read_markdown_from_folders(markdown_folder)
    improved_questions = []

    for question in tqdm(questions):
        metrics_to_check = ['faithfulness_score', 'specificity_score']
        for industry in question['industries']:
            metrics_to_check.append(f"relevancy_score_{industry}")

        threshold = 9 if len(question['industries']) == 1 else 7
        flagged_metrics = [metric for metric in metrics_to_check if question.get(metric, 0) < threshold]

        if len(flagged_metrics) == 1:
            try:
                response = improve_question(question, flagged_metrics[0], markdown_content)
                question['question'] = response.improved_question
                question[flagged_metrics[0]] = threshold
                improved_questions.append(question)
            except Exception as e:
                print(f"An error occurred processing question: {question['question']}")
                print(f"Error: {e}")
        elif len(flagged_metrics) == 0:
            improved_questions.append(question)
        # If more than one metric is flagged, the question is discarded

    with open(output_file, 'w') as f:
        json.dump(improved_questions, f, indent=2)
import json
import os
from typing import List, Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential
from openai import OpenAI
from pydantic import BaseModel, Field
from tqdm import tqdm

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
)

class ImprovedQuestionResponse(BaseModel):
    improved_question: str = Field(..., description="The improved version of the original question")
    improved_metric: str = Field(..., description="The metric that was improved")

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

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def improve_question(question: Dict[str, Any], metric_to_improve: str, markdown_content: List[Dict[str, str]]) -> ImprovedQuestionResponse:
    relevant_content = "\n\n".join([
        f"Industry: {md['industry']}\n{md['content'][:500]}..." 
        for md in markdown_content 
        if md['industry'] in question['industries']
    ])

    prompt = f"""
    Improve the following question by focusing on the {metric_to_improve} metric. 
    The question is from the {', '.join(question['industries'])} industry/industries.

    Original question: {question['question']}

    Current metrics:
    Faithfulness: {question['faithfulness_score']}
    Relevancy: {question.get(f"relevancy_score_{question['industries'][0]}", 'N/A')}
    Specificity: {question['specificity_score']}

    Relevant industry content:
    {relevant_content}

    Guidelines for improvement:
    1. If improving faithfulness: Ensure accuracy, verify metrics and concepts, remove misleading information.
    2. If improving relevancy: Focus on industry-specific aspects, use appropriate terminology.
    3. If improving specificity: Add precise details, use exact metric names or values, narrow the scope if needed.

    Maintain the original structure, intent, and difficulty level of the question.
    If it's a multi-choice question, preserve that format.

    Provide only the improved question without any explanation.
    """

    response = client.beta.chat.completions.parse(
        model="gpt-4o-2024-08-06",
        messages=[
            {"role": "system", "content": "You are an expert at improving questions while maintaining their core meaning and structure."},
            {"role": "user", "content": prompt}
        ],
        response_format=ImprovedQuestionResponse,
        temperature=0.1,
    )
    return response.choices[0].message.parsed

def process_questions(input_file: str, output_file: str, markdown_folder: str):
    with open(input_file, 'r') as f:
        questions: List[Dict[str, Any]] = json.load(f)

    markdown_content = read_markdown_from_folders(markdown_folder)
    improved_questions = []

    for question in tqdm(questions):
        metrics_to_check = ['faithfulness_score', 'specificity_score']
        for industry in question['industries']:
            metrics_to_check.append(f"relevancy_score_{industry}")

        threshold = 9 if len(question['industries']) == 1 else 7
        flagged_metrics = [metric for metric in metrics_to_check if question.get(metric, 0) < threshold]

        if len(flagged_metrics) == 1:
            try:
                response = improve_question(question, flagged_metrics[0], markdown_content)
                question['question'] = response.improved_question
                question[flagged_metrics[0]] = threshold
                improved_questions.append(question)
            except Exception as e:
                print(f"An error occurred processing question: {question['question']}")
                print(f"Error: {e}")
        elif len(flagged_metrics) == 0:
            improved_questions.append(question)
        # If more than one metric is flagged, the question is discarded

    with open(output_file, 'w') as f:
        json.dump(improved_questions, f, indent=2)
