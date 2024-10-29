import asyncio
import json
import os
import random
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

import numpy as np
import pandas as pd
from binary_checks import verify_binary_checks
from bleurouge import calculate_text_similarity_metrics
from openai import OpenAI
from quality import verify_question_quality
from sba_check import verify_one_and_only_one_correct_answer
from specificity_check import verify_specificity

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


def load_questions(file_path: str) -> List[Dict]:
    with open(file_path) as f:
        questions = json.load(f)
    for question in questions:
        question["file_source"] = os.path.basename(os.path.dirname(file_path))
    return questions


def load_context(industries: List[str]) -> str:
    contexts = []
    for industry in industries:
        file_path = f"./markdowns/{industry}/full_content.md"
        if os.path.exists(file_path):
            with open(file_path) as f:
                contexts.append(f.read())
    return "\n\n".join(contexts)


def is_multiple_choice(question: Dict) -> bool:
    return all(f"option{chr(65 + i)}" in question for i in range(5))


def should_check_question(question: Dict) -> bool:
    if question["answer"] == "<UNKNOWN>":
        return False

    if not is_multiple_choice(question):
        options = [question.get(f"option{chr(65+i)}", "") for i in range(5)]
        if not all(
            len(opt) == 1 and "A" <= opt <= "E"
            for opt in options + [question["answer"]]
        ):
            return False

    return True


def check_question(question: Dict, context: str) -> Dict:
    print("Checking question")
    combined_result = {
        "industries": question["industries"],
        "qa_type": question["qa_type"],
        "question": question["question"],
        "answer": question["answer"],
        "temperature": question["temperature"],
        "version": question["version"],
    }

    specificity_result = verify_specificity(question, context)
    combined_result["specificity_score"] = specificity_result["specificity_score"]

    if is_multiple_choice(question):
        sba_result = verify_one_and_only_one_correct_answer(question, context)
        combined_result["one_and_only_one_correct_answer"] = sba_result[
            "one_and_only_one_correct_answer"
        ]
        combined_result["correct_answers"] = (
            sba_result["correct_answers"] if sba_result else None
        )

    quality_result = verify_question_quality(question)
    if quality_result:
        combined_result["faithfulness_score"] = quality_result.faithfulness_score
        for industry, metrics in quality_result.industry_metrics.items():
            combined_result[f"relevancy_score_{industry}"] = metrics.relevancy_score

    binary_result = verify_binary_checks(question, context)
    combined_result["negative_question"] = (
        binary_result["negative_question"] if binary_result else None
    )
    combined_result["multihop_check"] = (
        binary_result["multihop_check"] if binary_result else None
    )

    if not is_multiple_choice(question):
        reference = (
            question["reference_text"]
            if isinstance(question["reference_text"], str)
            else " ".join(question["reference_text"])
        )
        candidate = question["answer"]
        similarity_metrics = calculate_text_similarity_metrics(reference, candidate)
        combined_result.update(similarity_metrics)
    else:
        combined_result["optionA"] = question.get("optionA", "")
        combined_result["optionB"] = question.get("optionB", "")
        combined_result["optionC"] = question.get("optionC", "")
        combined_result["optionD"] = question.get("optionD", "")
        combined_result["optionE"] = question.get("optionE", "")

    return combined_result


async def run_checks(questions: List[Dict], batch_size: int = 100) -> List[Dict]:
    results = []
    total_questions = len(questions)

    async def process_batch(batch):
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            batch_results = await asyncio.gather(
                *[
                    loop.run_in_executor(
                        executor,
                        check_question,
                        question,
                        load_context(question["industries"]),
                    )
                    for question in batch
                ],
            )
        return batch_results

    for i in range(0, total_questions, batch_size):
        batch = questions[i : i + batch_size]
        print(f"Processing questions {i+1} to {min(i+batch_size, total_questions)}")
        batch_results = await process_batch(batch)
        results.extend(batch_results)

    return results


def save_results(results: List[Dict], output_file: str):
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved in {output_file}")


def create_combined_csv(results: List[Dict], output_file: str) -> pd.DataFrame:
    data = []
    for result in results:
        relevancy_scores = [
            value for key, value in result.items() if key.startswith("relevancy_score_")
        ]
        avg_relevancy = np.mean(relevancy_scores) if relevancy_scores else np.nan

        row = {
            "Question Type": "MCQ" if is_multiple_choice(result) else "Free-text",
            "Temperature": result.get("temperature", "unknown"),
            "Method": result["version"],
            "Hop Type": result.get("qa_type", "unknown"),
            "Industry Type": result["version"],
            "Specificity": result.get("specificity_score", np.nan),
            "Faithfulness": result.get("faithfulness_score", np.nan),
            "Relevancy": avg_relevancy,
            "Single Best Answer": 1
            if result.get("one_and_only_one_correct_answer", False)
            else 0,
            "Negative Questions": 1 if result.get("negative_question", 0) == 1 else 0,
            "Multi-hop Questions": 1 if result.get("multihop_check", 0) == 1 else 0,
        }
        try:
            row["BLEU"] = result.get("BLEU", np.nan)
            row["ROUGE"] = result.get("ROUGE", np.nan)
        except KeyError:
            pass

        data.append(row)

    df = pd.DataFrame(data)

    column_order = [
        "Question Type",
        "Temperature",
        "Method",
        "Hop Type",
        "Industry Type",
        "Specificity",
        "Faithfulness",
        "Relevancy",
        "Single Best Answer",
        "Negative Questions",
        "Multi-hop Questions",
        "BLEU",
        "ROUGE",
    ]
    df = df.reindex(columns=[col for col in column_order if col in df.columns])

    df.to_csv(output_file, index=False)
    print(f"Combined CSV saved in {output_file}")
    return df


def calculate_stats(df: pd.DataFrame, group_by: str = "Industry Type") -> pd.DataFrame:
    metrics = [
        "Specificity",
        "Faithfulness",
        "Relevancy",
        "Single Best Answer",
        "Negative Questions",
        "Multi-hop Questions",
    ]
    if "BLEU" in df.columns:
        metrics.extend(["BLEU", "ROUGE"])

    stats = []

    for metric in metrics:
        for group in df[group_by].unique():
            row = {"Metric": metric, group_by: group}
            subset = df[df[group_by] == group]

            if metric in [
                "Specificity",
                "Faithfulness",
                "Relevancy",
                "BLEU",
                "ROUGE",
            ]:
                avg = subset[metric].mean()
                row["Value"] = avg
            else:
                percentage = (subset[metric].sum() / len(subset)) * 100
                row["Value"] = percentage

            stats.append(row)

    stats_df = pd.DataFrame(stats)
    return stats_df


def save_stats_to_csv(stats: pd.DataFrame, output_file: str):
    stats.to_csv(output_file, index=False)
    print(f"Statistics saved in {output_file}")


async def process_folder(
    folder_path: str, version: str, batch_size: int = 100, max_files: int = 10
) -> List[Dict]:
    all_results = []
    json_files = [f for f in os.listdir(folder_path) if f.endswith(".json")]

    # Randomly select 10 files if there are more than 10
    if len(json_files) > max_files:
        json_files = random.sample(json_files, max_files)

    for filename in json_files:
        file_path = os.path.join(folder_path, filename)
        questions = load_questions(file_path)[:10]
        for question in questions:
            question["version"] = version
        results = await run_checks(questions, batch_size)
        all_results.extend(results)
    return all_results


if __name__ == "__main__":
    cross_folder = "./final/fewshot_mcq_cross"
    local_folder = "./final/fewshot_mcq_local"
    results_file = "./final/outputs/varied/combined_results.json"
    combined_csv_file = "./final/outputs/varied/combined_results.csv"
    comparison_csv_file = "./final/outputs/varied/comparison_results.csv"
    batch_size = 500
    max_files = 10  # New parameter to limit the number of files processed

    if os.path.exists(results_file):
        print("Results file already exists. Using existing results.")
        with open(results_file) as f:
            results = json.load(f)
    else:
        loop = asyncio.get_event_loop()
        cross_results = loop.run_until_complete(
            process_folder(cross_folder, "cross", batch_size, max_files)
        )
        local_results = loop.run_until_complete(
            process_folder(local_folder, "local", batch_size, max_files)
        )
        results = cross_results + local_results
        save_results(results, results_file)

    combined_df = create_combined_csv(results, combined_csv_file)

    # Create and save the tables
    # 1. Average metrics per technique
    technique_stats = calculate_stats(combined_df, group_by="Method")
    technique_table = technique_stats.pivot(
        index="Metric", columns="Method", values="Value"
    )
    technique_table.to_csv("./final/outputs/varied/technique_comparison.csv")
    print("Average metrics per technique:")
    print(technique_table)
    print("\n")

    # 2. Single/multihop sensitivity table
    hop_stats = calculate_stats(combined_df, group_by="Hop Type")
    hop_table = hop_stats.pivot(index="Metric", columns="Hop Type", values="Value")
    hop_table.to_csv("./final/outputs/varied/hop_sensitivity.csv")
    print("Single/multihop sensitivity table:")
    print(hop_table)
    print("\n")

    # 3. Temperature sensitivity table
    temp_stats = calculate_stats(combined_df, group_by="Temperature")
    temp_table = temp_stats.pivot(index="Metric", columns="Temperature", values="Value")
    temp_table.to_csv("./final/outputs/varied/temperature_sensitivity.csv")
    print("Temperature sensitivity table:")
    print(temp_table)

    print("Results saved and statistics calculated.")
