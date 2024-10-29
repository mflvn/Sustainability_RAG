import json
import os
import re
from typing import Dict, List

from sklearn.metrics import f1_score, hamming_loss, precision_score, recall_score
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm

from utils.model import ModelWrapper


def normalize_industry_name(name: str) -> str:
    """Normalize industry name by removing special characters and converting to lowercase."""
    name = name.replace(" ", "-").lower()
    return name


def map_industry_to_name(industry: str, label_mapping: Dict[str, str]) -> str:
    normalized_industry = normalize_industry_name(industry)
    for code, name in label_mapping.items():
        if normalized_industry in normalize_industry_name(name):
            return name
    return None  # Return None if no match is found


def compute_metrics(true_labels, predicted_labels):
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


def load_questions_data(questions_folder: str) -> List[Dict]:
    all_questions = []
    for filename in os.listdir(questions_folder):
        if filename.endswith(".json"):
            file_path = os.path.join(questions_folder, filename)
            with open(file_path, "r") as f:
                questions_data = json.load(f)
                all_questions.extend(questions_data)
    return all_questions


# Load label mapping
with open("label_mapping.json", "r") as f:
    label_mapping = json.load(f)

# Load and preprocess data
questions_data = load_questions_data("./final_questions/final_mcq/fewshot_mcq_cross")

# Initialize model
model = ModelWrapper()

all_true_labels = []
all_predicted_industries = []

# select 20 random questiins
import random

random.seed(42)
questions_data = random.sample(questions_data, 20)

for data in tqdm(questions_data):
    question = data["question"]
    true_industries = data["industries"]

    # Query the model
    predicted_industries = model.query_unstructured(question)

    # Map predicted industries to full names
    predicted_names = [
        map_industry_to_name(industry, label_mapping)
        for industry in predicted_industries
    ]
    predicted_names = [name for name in predicted_names if name is not None]

    all_true_labels.append(true_industries)
    all_predicted_industries.append(predicted_names)

    print(f"\nQuestion: {question}")
    print(f"Predicted Industries: {predicted_industries}")
    print(f"Predicted Names: {predicted_names}")
    print(f"True Industries: {true_industries}")

# Compute metrics
metrics = compute_metrics(all_true_labels, all_predicted_industries)

print("\nEvaluation Metrics:")
for key, value in metrics.items():
    print(f"{key}: {value:.4f}")
