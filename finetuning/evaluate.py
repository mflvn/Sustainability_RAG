import asyncio
import glob
import json
import os

from together import AsyncTogether

async_together_client = AsyncTogether(
    api_key=os.getenv("TOGETHER_API_KEY"),
)

base_model = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
top_oss_model = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"
finetuned_model = (
    "dfsf/Meta-Llama-3.1-8B-Instruct-Reference-2024-08-11-16-13-33-bbac2fc3"
)
input_directory = "./final_questions/final_mcq/fewshot_mcq_local"
results_file_path = "./finetuning/all_results.json"


async def chatCompletion(model, instruction):
    completion = await async_together_client.chat.completions.create(
        messages=[
            {"role": "user", "content": instruction},
        ],
        model=model,
        max_tokens=3500,
    )
    return completion.choices[0].message.content


def extract_answer(response):
    for letter in ["A", "B", "C", "D", "E"]:
        if (
            f"The correct answer is {letter}" in response
            or f"Option {letter}" in response
            or f"Answer: {letter}" in response
            or letter in response[0]
        ):
            return letter
    return None  # If no clear answer is found


async def process_file(file_path):
    with open(file_path, "r") as file:
        qa_data = json.load(file)

    results = []
    base_correct = 0
    top_oss_correct = 0
    finetuned_correct = 0
    total_questions = len(qa_data)

    for qa_item in qa_data:
        question = qa_item["question"]

        # Skip questions that start with "What is the category..."
        if question.lower().startswith("what is the category"):
            total_questions -= 1
            continue

        options = f"""A: {qa_item['optionA']}
B: {qa_item['optionB']}
C: {qa_item['optionC']}
D: {qa_item['optionD']}
E: {qa_item['optionE']}"""

        full_question = f"{question}\n\n{options}\n\nPlease provide the letter of the correct answer in this format: 'The correct answer is A (or B, C,D,E)' ."

        (
            baseModelCompletion,
            topOSSModelCompletion,
            finetunedModelCompletion,
        ) = await asyncio.gather(
            chatCompletion(base_model, full_question),
            chatCompletion(top_oss_model, full_question),
            chatCompletion(finetuned_model, full_question),
        )

        base_answer = extract_answer(baseModelCompletion)
        top_oss_answer = extract_answer(topOSSModelCompletion)
        finetuned_answer = extract_answer(finetunedModelCompletion)

        correct_answer = qa_item["answer"]

        base_correct += int(base_answer == correct_answer)
        top_oss_correct += int(top_oss_answer == correct_answer)
        finetuned_correct += int(finetuned_answer == correct_answer)

        results.append(
            {
                "question": question,
                "correct_answer": correct_answer,
                "base_model_answer": base_answer,
                "top_oss_model_answer": top_oss_answer,
                "finetuned_model_answer": finetuned_answer,
            }
        )

    return results, total_questions, base_correct, top_oss_correct, finetuned_correct


def calculate_stats_from_results(results):
    total_questions = len(results)
    base_correct = sum(
        1 for r in results if r["base_model_answer"] == r["correct_answer"]
    )
    top_oss_correct = sum(
        1 for r in results if r["top_oss_model_answer"] == r["correct_answer"]
    )
    finetuned_correct = sum(
        1 for r in results if r["finetuned_model_answer"] == r["correct_answer"]
    )

    return total_questions, base_correct, top_oss_correct, finetuned_correct


async def main():
    if os.path.exists(results_file_path):
        print("Results file found. Calculating stats from existing results.")
        with open(results_file_path, "r", encoding="utf-8") as file:
            all_results = json.load(file)

        # Filter out questions starting with "What is the category..."
        filtered_results = [
            r
            for r in all_results
            if not r["question"].lower().startswith("what is the category")
            and "unit of measure" not in r["question"].lower()
        ]

        (
            total_questions,
            total_base_correct,
            total_top_oss_correct,
            total_finetuned_correct,
        ) = calculate_stats_from_results(filtered_results)
    else:
        all_results = []
        total_questions = 0
        total_base_correct = 0
        total_top_oss_correct = 0
        total_finetuned_correct = 0

        json_files = glob.glob(os.path.join(input_directory, "*.json"))

        for file_path in json_files:
            print(f"Processing file: {file_path}")
            (
                results,
                questions,
                base_correct,
                top_oss_correct,
                finetuned_correct,
            ) = await process_file(file_path)
            all_results.extend(results)
            total_questions += questions
            total_base_correct += base_correct
            total_top_oss_correct += top_oss_correct
            total_finetuned_correct += finetuned_correct

        with open(results_file_path, "w", encoding="utf-8") as results_file:
            json.dump(all_results, results_file, indent=4)

    print(f"\nTotal questions processed: {total_questions}")
    print(
        f"Base model (Llama-3-8b) accuracy: {total_base_correct / total_questions * 100:.2f}%"
    )
    print(
        f"Top OSS model (Llama-3-70b) accuracy: {total_top_oss_correct / total_questions * 100:.2f}%"
    )
    print(
        f"Fine-tuned model accuracy: {total_finetuned_correct / total_questions * 100:.2f}%"
    )


if __name__ == "__main__":
    asyncio.run(main())
