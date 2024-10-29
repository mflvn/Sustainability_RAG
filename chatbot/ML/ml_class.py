import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import xgboost as xgb
from openai import OpenAI
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score,
    hamming_loss,
    jaccard_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm

# Set up OpenAI client
client = OpenAI()


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

# Create id2label and label2id mappings
id2label = {int(k): v for k, v in label_mapping.items()}
label2id = {v: int(k) for k, v in label_mapping.items()}

# Load and preprocess data
questions_data = load_questions_data("./final_questions/final_mcq/fewshot_mcq_cross")
questions_data.extend(
    load_questions_data("./final_questions/final_mcq/fewshot_mcq_local")
)


def process_questions(questions_data: List[Dict], label2id: Dict[str, int]):
    processed_data = []
    label_counts = [0] * len(label2id)
    for item in questions_data:
        labels = [
            label2id[industry]
            for industry in item["industries"]
            if industry in label2id
        ]
        for label in labels:
            label_counts[label] += 1
        processed_data.append({"text": item["question"], "labels": labels})

    # print("Label distribution:")
    # for i, count in enumerate(label_counts):
    #     print(f"{id2label[i]}: {count}")

    return processed_data


# Preprocess data
classification_data = process_questions(questions_data, label2id)

# Create DataFrame
df = pd.DataFrame(classification_data)

# Split the data
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Set up OpenAI client
client = OpenAI()

# ... (previous code for loading and processing data remains the same)


def get_embeddings(
    texts: List[str], batch_size=200, cache_file="embeddings_cache.npz"
) -> np.ndarray:
    if os.path.exists(cache_file):
        print(f"Loading embeddings from cache: {cache_file}")
        with np.load(cache_file) as data:
            return data["embeddings"]

    print("Generating new embeddings...")
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(model="text-embedding-ada-002", input=batch)
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

    embeddings = np.array(all_embeddings)

    print(f"Saving embeddings to cache: {cache_file}")
    np.savez_compressed(cache_file, embeddings=embeddings)

    return embeddings


# Create DataFrame
df = pd.DataFrame(classification_data)

# Split the data
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Generate embeddings for train and test data
print("Processing train data embeddings...")
train_embeddings = get_embeddings(
    train_df["text"].tolist(), cache_file="train_embeddings_cache.npz"
)
print("Processing test data embeddings...")
test_embeddings = get_embeddings(
    test_df["text"].tolist(), cache_file="test_embeddings_cache.npz"
)

# Prepare multi-label binarizer
mlb = MultiLabelBinarizer()
train_labels = mlb.fit_transform(train_df["labels"])
test_labels = mlb.transform(test_df["labels"])


def compute_metrics(true_labels, pred_probs, threshold=0.06):
    predictions = (pred_probs >= threshold).astype(int)
    return {
        "micro_f1": f1_score(
            true_labels, predictions, average="micro", zero_division=0
        ),
        "macro_f1": f1_score(
            true_labels, predictions, average="macro", zero_division=0
        ),
        "weighted_f1": f1_score(
            true_labels, predictions, average="weighted", zero_division=0
        ),
        "micro_precision": precision_score(
            true_labels, predictions, average="micro", zero_division=0
        ),
        "micro_recall": recall_score(
            true_labels, predictions, average="micro", zero_division=0
        ),
        "hamming_loss": hamming_loss(true_labels, predictions),
    }


# Train and evaluate Random Forest model
rf_model = RandomForestClassifier(
    n_estimators=200,
    n_jobs=-1,
    random_state=42,
)
rf_model.fit(train_embeddings, train_labels)

rf_train_pred_probs = rf_model.predict_proba(train_embeddings)
rf_test_pred_probs = rf_model.predict_proba(test_embeddings)

rf_train_predictions = np.column_stack([probs[:, 1] for probs in rf_train_pred_probs])
rf_test_predictions = np.column_stack([probs[:, 1] for probs in rf_test_pred_probs])

print("\nRandom Forest Metrics:")
print("Training Metrics:")
rf_train_metrics = compute_metrics(train_labels, rf_train_predictions)
for key, value in rf_train_metrics.items():
    print(f"{key}: {value:.4f}")

print("\nTest Metrics:")
rf_test_metrics = compute_metrics(test_labels, rf_test_predictions)
for key, value in rf_test_metrics.items():
    print(f"{key}: {value:.4f}")

# Train and evaluate XGBoost model
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_delta_step=1,
    gamma=1,
    random_state=42,
    use_label_encoder=False,
)

xgb_model.fit(train_embeddings, train_labels)

xgb_train_predictions = xgb_model.predict_proba(train_embeddings)
xgb_test_predictions = xgb_model.predict_proba(test_embeddings)

print("\nXGBoost Metrics:")
print("Training Metrics:")
xgb_train_metrics = compute_metrics(train_labels, xgb_train_predictions)
for key, value in xgb_train_metrics.items():
    print(f"{key}: {value:.4f}")

print("\nTest Metrics:")
xgb_test_metrics = compute_metrics(test_labels, xgb_test_predictions)
for key, value in xgb_test_metrics.items():
    print(f"{key}: {value:.4f}")

# Train and evaluate MLP model
mlp_model = MLPClassifier(hidden_layer_sizes=(200, 100), max_iter=500, random_state=42)
mlp_model.fit(train_embeddings, train_labels)

mlp_train_predictions = mlp_model.predict_proba(train_embeddings)
mlp_test_predictions = mlp_model.predict_proba(test_embeddings)

print("\nMLP Metrics:")
print("Training Metrics:")
mlp_train_metrics = compute_metrics(train_labels, mlp_train_predictions)
for key, value in mlp_train_metrics.items():
    print(f"{key}: {value:.4f}")

print("\nTest Metrics:")
mlp_test_metrics = compute_metrics(test_labels, mlp_test_predictions)
for key, value in mlp_test_metrics.items():
    print(f"{key}: {value:.4f}")


def predict_industries(questions: List[str], models, mlb, threshold=0.3):
    embeddings = get_embeddings(questions, cache_file="prediction_embeddings_cache.npz")

    all_predictions = []
    for model in models:
        pred_probs = model.predict_proba(embeddings)
        if isinstance(model, RandomForestClassifier):
            predictions = np.column_stack([probs[:, 1] for probs in pred_probs])
        else:
            predictions = pred_probs
        all_predictions.append(predictions)

    # Average predictions from all models
    avg_predictions = np.mean(all_predictions, axis=0)

    predicted_industries = []
    for pred in avg_predictions:
        predicted_labels = (pred >= threshold).astype(int)
        industries = mlb.inverse_transform(predicted_labels.reshape(1, -1))[0]

        if not industries:
            top_industries = np.argsort(pred)[-3:]
            industries = mlb.classes_[top_industries]

        predicted_industries.append(list(industries))

    return predicted_industries


# Example usage
new_questions = [
    "What are the environmental impacts of water utilities?",
    "How do cruise lines manage their carbon emissions?",
]

