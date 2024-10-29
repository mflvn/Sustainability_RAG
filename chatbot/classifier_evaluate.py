import json
import os
import random
from typing import Dict, List, Set

from prompting import IndustryClassificationRetriever
from sklearn.metrics import f1_score, hamming_loss, precision_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer


def load_test_data(folder_path: str, num_questions: int = 20) -> List[Dict]:
    all_questions = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r") as f:
                questions_data = json.load(f)
                all_questions.extend(questions_data)

    # Randomly select num_questions from all_questions
    return random.sample(all_questions, min(num_questions, len(all_questions)))


def evaluate_retriever(
    retriever: IndustryClassificationRetriever, test_data: List[Dict]
) -> Dict:
    true_labels = []
    predicted_labels = []

    for item in test_data:
        query = item["question"]
        true_industries = set(item["industries"])

        retrieved_nodes = retriever.retrieve(query)
        predicted_industries = set(
            [node.node.metadata["industry"] for node in retrieved_nodes]
        )

        true_labels.append(list(true_industries))
        predicted_labels.append(list(predicted_industries))

    return calculate_metrics(true_labels, predicted_labels)


def calculate_metrics(
    true_labels: List[List[str]], predicted_labels: List[List[str]]
) -> Dict:
    mlb = MultiLabelBinarizer()
    true_labels_bin = mlb.fit_transform(true_labels)
    predicted_labels_bin = mlb.transform(predicted_labels)

    # Check if all predictions are empty
    if predicted_labels_bin.sum() == 0:
        return {
            metric: 0.0
            for metric in [
                "hamming_loss",
                "precision_micro",
                "recall_micro",
                "f1_micro",
                "precision_macro",
                "recall_macro",
                "f1_macro",
            ]
        }

    return {
        "hamming_loss": hamming_loss(true_labels_bin, predicted_labels_bin),
        "precision_micro": precision_score(
            true_labels_bin, predicted_labels_bin, average="micro", zero_division=0
        ),
        "recall_micro": recall_score(
            true_labels_bin, predicted_labels_bin, average="micro", zero_division=0
        ),
        "f1_micro": f1_score(
            true_labels_bin, predicted_labels_bin, average="micro", zero_division=0
        ),
        "precision_macro": precision_score(
            true_labels_bin, predicted_labels_bin, average="macro", zero_division=0
        ),
        "recall_macro": recall_score(
            true_labels_bin, predicted_labels_bin, average="macro", zero_division=0
        ),
        "f1_macro": f1_score(
            true_labels_bin, predicted_labels_bin, average="macro", zero_division=0
        ),
    }


def main():
    test_data = load_test_data("./final_questions/final_mcq/fewshot_free_cross", num_questions=25)
    models = ["gpt-4o-mini", "gpt-4o-2024-08-06"]

    results = {}

    for model in models:
        print(f"Evaluating with model: {model}")
        retriever = IndustryClassificationRetriever(model=model)
        metrics = evaluate_retriever(retriever, test_data)
        results[model] = metrics
        print(f"Results for {model}:")
        print(json.dumps(metrics, indent=2))
        print("\n" + "=" * 50 + "\n")

    # Save overall results
    with open("evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
