import asyncio
import csv
import json
import os
import random
from datetime import datetime
from glob import glob
from typing import Any, Dict, List, Union

from pydantic import BaseModel, Field
from tqdm.asyncio import tqdm as async_tqdm

from chatbot.prompting import IndustryClassificationRetriever


class MCQAnswer(BaseModel):
    correctOption: str = Field(
        ...,
        description="The correct option for the question, a single character from A to D.",
    )


class FreeTextAnswer(BaseModel):
    text: str = Field(
        ...,
        description="Free text response to the query.",
    )


def load_datasets(folder_path: str) -> List[Dict[str, Any]]:
    datasets = []
    json_files = glob(os.path.join(folder_path, "*.json"))
    for file_path in json_files:
        with open(file_path, "r") as f:
            dataset = json.load(f)
            if isinstance(dataset, list):
                datasets.extend(dataset)
            else:
                print(f"Warning: {file_path} does not contain a JSON array. Skipping.")
    return datasets


def format_question(item: Dict[str, Any]) -> str:
    question = item["question"]
    if "optionA" in item:
        options = [
            f"{option}: {item[f'option{option}']}"
            for option in ["A", "B", "C", "D"]
            if f"option{option}" in item
        ]
        return f"{question}\n\n" + "\n".join(options)
    return question


async def process_question(
    item: Dict[str, Any], retriever: IndustryClassificationRetriever
) -> Dict[str, Any]:
    formatted_question = format_question(item)
    correct_answer = item.get("answer")

    try:
        model_answer, industries, combined = retriever.retrieve(formatted_question)
        # print(f"Inferred industries: {industries}")
        # print(f"True industries: {item['industries']}")
        # find the difference between the two lists
        diffindustries = list(set(industries) - set(item["industries"]))
        print(f"Failed to find industries: {diffindustries}")
        if "optionA" in item:  # This is an MCQ question
            is_correct = model_answer.strip().upper() == correct_answer.strip().upper()
        else:  # This is a free text question
            is_correct = (
                None  # We can't automatically determine correctness for free text
            )
        if is_correct:
            print("Correct!")
        else:
            print("Incorrect.")
            print(f"Correct answer: {correct_answer}")
            print(f"Model answer: {model_answer}")
            print(f"Question: {item}")
            print(f"Combined content: {combined}")

        return {
            "question": item["question"],
            "type": "mcq" if "optionA" in item else "free_text",
            "correct_answer": correct_answer,
            "model_answer": model_answer,
            "is_correct": is_correct,
            "industries": industries,
        }
    except Exception as e:
        print(f"Query failed: {str(e)}")
        return {
            "question": item["question"],
            "type": "mcq" if "optionA" in item else "free_text",
            "correct_answer": correct_answer,
            "model_answer": "Error",
            "is_correct": False,
            "industries": [],
        }


async def evaluate_questions(
    dataset: List[Dict[str, Any]], retriever: IndustryClassificationRetriever
) -> List[Dict[str, Any]]:
    tasks = [process_question(item, retriever) for item in dataset]
    return await async_tqdm.gather(*tasks)


async def run_evaluation(dataset_folder: str, output_dir: str):
    dataset = load_datasets(dataset_folder)
    #  select 10 random questions using random
    dataset = random.sample(dataset, 21)

    retriever = IndustryClassificationRetriever()

    print(f"Evaluating {len(dataset)} questions...")
    evaluation_results = await evaluate_questions(dataset, retriever)

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_json = os.path.join(output_dir, f"evaluation_results_{timestamp}.json")
    output_csv = os.path.join(output_dir, f"evaluation_summary_{timestamp}.csv")

    with open(output_json, "w") as f:
        json.dump(evaluation_results, f, indent=2)

    # Calculate summary statistics
    total_questions = len(evaluation_results)
    mcq_questions = [q for q in evaluation_results if q["type"] == "mcq"]
    correct_mcq_answers = sum(1 for q in mcq_questions if q["is_correct"])
    mcq_accuracy = correct_mcq_answers / len(mcq_questions) if mcq_questions else 0

    with open(output_csv, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(
            ["Total Questions", "MCQ Questions", "Correct MCQ Answers", "MCQ Accuracy"]
        )
        csv_writer.writerow(
            [
                total_questions,
                len(mcq_questions),
                correct_mcq_answers,
                f"{mcq_accuracy:.2%}",
            ]
        )

    print(f"Evaluation complete. Results saved to {output_json} and {output_csv}")
    print(f"MCQ Accuracy: {mcq_accuracy:.2%}")


if __name__ == "__main__":
    dataset_folder = "./final_questions/final_mcq/fewshot_mcq_local"
    output_dir = "./evaluation_results"
    asyncio.run(run_evaluation(dataset_folder, output_dir))
