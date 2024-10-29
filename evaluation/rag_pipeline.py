import asyncio
import csv
import json
import os
from datetime import datetime
from glob import glob

from tqdm.asyncio import tqdm as async_tqdm
from utils.model import ModelWrapper, QueryFailedException
from chatbot.prompting import IndustryClassificationRetriever

def load_datasets(folder_path):
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

def format_question(item):
    return item["question"]

@retry(
    stop=stop_after_attempt(5),
    wait=wait_fixed(0.1),
    retry=retry_if_exception_type(QueryFailedException),
)
async def perform_rag_query(model, formatted_question, filters):
    loop = asyncio.get_running_loop()
    try:
        return await loop.run_in_executor(
            None, model.query_unstructured, formatted_question, filters
        )
    except QueryFailedException as e:
        print(f"Query failed: {str(e)}")
        raise

async def process_question(item, model, industry_retriever):
    formatted_question = format_question(item)
    correct_answer = item["answer"]

    # Use the IndustryClassificationRetriever to filter industries
    relevant_industries = industry_retriever._identify_industries(formatted_question)
    # Create filters based on identified industries
    filters = model._filter_chunks_by_industry(formatted_question)


    try:
        response = await perform_rag_query(model, formatted_question, filters)
    except Exception as e:
        print(f"Query failed: {str(e)}")
        return {
            "question": item["question"],
            "error": "Query failed",
            "reference": item.get("reference_text", ""),
            "page": item.get("pages", ""),
            "industries": item.get("industries", []),
            "qa_type": item.get("qa_type", ""),
        }, None

    similarity_metrics = calculate_text_similarity_metrics(correct_answer, response)

    return {
        "question": item["question"],
        "correct_answer": correct_answer,
        "rag_response": response,
        "bleu_score": similarity_metrics["BLEU"],
        "rouge_l_score": similarity_metrics["ROUGE"],
        "reference": item.get("reference_text", ""),
        "page": item.get("pages", ""),
        "industries": item.get("industries", []),
        "qa_type": item.get("qa_type", ""),
    }, similarity_metrics

async def evaluate_rag(dataset, model, industry_retriever):
    results = {
        "evaluation_date": datetime.now().isoformat(),
        "total_questions": len(dataset),
        "questions": [],
        "average_bleu": 0,
        "average_rouge_l": 0,
    }

    tasks = [process_question(item, model, industry_retriever) for item in dataset]

    total_bleu = 0
    total_rouge_l = 0

    for task in async_tqdm.as_completed(tasks, total=len(tasks)):
        question_result, similarity_metrics = await task
        results["questions"].append(question_result)
        if similarity_metrics:
            total_bleu += similarity_metrics["BLEU"]
            total_rouge_l += similarity_metrics["ROUGE"]

    if results["total_questions"] > 0:
        results["average_bleu"] = total_bleu / results["total_questions"]
        results["average_rouge_l"] = total_rouge_l / results["total_questions"]

    return results

async def run_experiment(experiment_type, dataset, output_dir, csv_writer, dataset_folder):
    if experiment_type == "model_size":
        params = ModelWrapper.MODEL_INFO.keys()
    elif experiment_type == "similarity_top_k":
        params = [1, 5, 10, 20, 50]
    elif experiment_type == "retriever":
        params = ModelWrapper.QUERY_MODES.keys()
    elif experiment_type == "chunking_namespace":
        params = ModelWrapper.CHUNKING_NAMESPACES.keys()
    elif experiment_type == "rag_technique":
        params = ModelWrapper.RAG_TRANSFORMS.keys()
    else:
        raise ValueError("Invalid experiment type")

    for param in params:
        print(f"Running experiment with {experiment_type}: {param}")

        if experiment_type == "model_size":
            model = ModelWrapper(model_size=param)
        elif experiment_type == "similarity_top_k":
            model = ModelWrapper(similarity_top_k=param)
        elif experiment_type == "retriever":
            model = ModelWrapper(vector_store_query_mode=param)
        elif experiment_type == "chunking_namespace":
            model = ModelWrapper(chunking_namespace=param)
        else:  # rag_technique
            model = ModelWrapper(rag_transform=param)

        industry_retriever = IndustryClassificationRetriever()
        evaluation_results = await evaluate_rag(dataset, model, industry_retriever)

        output_filename = f"{experiment_type}_{param}_{dataset_folder.split('_')[-1]}_eval.json"
        output_file = os.path.join(output_dir, output_filename)

        with open(output_file, "w") as f:
            json.dump(evaluation_results, f, indent=2)

        print(f"Evaluation complete for {experiment_type}: {param}")
        print(f"Results saved to {output_file}")
        print(f"Average BLEU score: {evaluation_results['average_bleu']:.4f}")
        print(f"Average ROUGE-L score: {evaluation_results['average_rouge_l']:.4f}")
        print("---")

        # Write results to CSV
        csv_writer.writerow(
            [
                experiment_type,
                param,
                model.get_model_info()["model_name"],
                model.get_model_info()["model_size"],
                evaluation_results["average_bleu"],
                evaluation_results["average_rouge_l"],
                evaluation_results["total_questions"],
            ]
        )

async def main():
    dataset_folder = "./final_questions/final_free/fewshot_free_local"
    dataset = load_datasets(dataset_folder)

    base_output_dir = "./final_experiment_results"
    os.makedirs(base_output_dir, exist_ok=True)

    experiment_types = [
        "model_size",
        # "similarity_top_k",
        # "retriever",
        # "chunking_namespace",
        # "rag_technique",
    ]

    for experiment_type in experiment_types:
        print(f"\nStarting experiment: {experiment_type}")
        output_dir = os.path.join(base_output_dir, experiment_type)
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file = os.path.join(
            output_dir,
            f"{experiment_type}_results_{timestamp}_{dataset_folder.split('_')[-1]}.csv",
        )

        with open(csv_file, "w", newline="") as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(
                [
                    "Experiment Type",
                    "Parameter",
                    "Model Name",
                    "Model Size (B)",
                    "Average BLEU Score",
                    "Average ROUGE-L Score",
                    "Total Questions",
                ]
            )

            await run_experiment(experiment_type, dataset, output_dir, csv_writer, dataset_folder)

        print(f"Results for {experiment_type} have been saved to {csv_file}")

    print("\nAll experiments completed.")

if __name__ == "__main__":
    asyncio.run(main())