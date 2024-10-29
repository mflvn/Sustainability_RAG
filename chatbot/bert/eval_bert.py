import json
import os
from typing import Dict, List

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Set up device
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS device")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA device")
else:
    device = torch.device("cpu")
    print("Using CPU")


def load_questions_data(questions_folder: str) -> List[Dict]:
    all_questions = []
    for filename in os.listdir(questions_folder):
        if filename.endswith(".json"):
            file_path = os.path.join(questions_folder, filename)
            with open(file_path, "r") as f:
                questions_data = json.load(f)
                all_questions.extend(questions_data)
    return all_questions


def load_model_and_tokenizer(model_path: str):
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer


def process_questions(questions_data: List[Dict], label2id: Dict[str, int]):
    processed_data = []
    for item in questions_data:
        labels = [0] * len(label2id)
        for industry in item["industries"]:
            if industry in label2id:
                labels[label2id[industry]] = 1
        processed_data.append({"text": item["question"], "labels": labels})
    return processed_data


def tokenize_and_encode(example, tokenizer):
    encoded = tokenizer(
        example["text"], padding="max_length", truncation=True, max_length=512
    )
    encoded["labels"] = example["labels"]
    return encoded


def find_optimal_threshold(model, dataset, tokenizer):
    model.eval()
    all_logits = []
    all_labels = []

    for batch in dataset:
        inputs = tokenizer(
            batch["text"],
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        all_logits.append(outputs.logits.cpu().numpy())
        all_labels.append(batch["labels"])

    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.array(all_labels)

    best_threshold = 0
    best_f1 = 0
    for threshold in np.arange(0.1, 0.9, 0.05):
        predictions = (sigmoid(all_logits) > threshold).astype(int)
        f1 = f1_score(all_labels, predictions, average="micro")
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    print(f"Optimal threshold: {best_threshold:.2f}")
    return best_threshold


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def predict(model, tokenizer, texts, threshold):
    model.eval()
    predictions = []
    for text in texts:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding="max_length",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        predicted_labels = sigmoid(logits.cpu().numpy()) > threshold
        predictions.append(predicted_labels[0])
    return np.array(predictions)


def compute_metrics(predictions, labels):
    return {
        "accuracy": accuracy_score(labels, predictions),
        "micro_f1": f1_score(labels, predictions, average="micro", zero_division=0),
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
        "weighted_f1": f1_score(
            labels, predictions, average="weighted", zero_division=0
        ),
        "micro_precision": precision_score(
            labels, predictions, average="micro", zero_division=0
        ),
        "micro_recall": recall_score(
            labels, predictions, average="micro", zero_division=0
        ),
    }


def main():
    # Load the model and tokenizer
    model_path = "./simple_industry_classifier"
    model, tokenizer = load_model_and_tokenizer(model_path)

    # Load label mapping
    with open("label_mapping.json", "r") as f:
        label_mapping = json.load(f)

    # Create id2label and label2id mappings
    id2label = {int(k): v for k, v in label_mapping.items()}
    label2id = {v: int(k) for k, v in label_mapping.items()}

    # Load and process questions from a specific folder
    questions_folder = (
        "./final_questions/final_mcq/fewshot_mcq_cross"  # Update this path
    )
    questions_data = load_questions_data(questions_folder)
    processed_data = process_questions(questions_data, label2id)

    # Create dataset
    dataset = Dataset.from_list(processed_data)
    tokenized_dataset = dataset.map(
        lambda x: tokenize_and_encode(x, tokenizer), remove_columns=["text"]
    )

    # Split dataset into validation and test sets
    val_test_split = tokenized_dataset.train_test_split(
        test_size=0.5, shuffle=True, seed=42
    )
    val_dataset = val_test_split["train"]
    test_dataset = val_test_split["test"]

    # Find optimal threshold using validation set
    threshold = find_optimal_threshold(model, val_dataset, tokenizer)

    # Make predictions on test set
    test_texts = [item["text"] for item in test_dataset]
    predictions = predict(model, tokenizer, test_texts, threshold)

    # Calculate metrics
    true_labels = np.array([item["labels"] for item in test_dataset])

    metrics = compute_metrics(predictions, true_labels)

    # Print results
    print("\nEvaluation Metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    # Example predictions
    print("\nExample Predictions:")
    for i in range(min(5, len(test_texts))):
        print(f"Text: {test_texts[i]}")
        print(
            f"True Labels: {[id2label[j] for j, label in enumerate(true_labels[i]) if label == 1]}"
        )
        print(
            f"Predicted Labels: {[id2label[j] for j, pred in enumerate(predictions[i]) if pred == 1]}"
        )
        print()


if __name__ == "__main__":
    main()
