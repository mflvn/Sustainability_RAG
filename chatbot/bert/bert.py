import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

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
        labels = [0] * len(label2id)
        for industry in item["industries"]:
            if industry in label2id:
                labels[label2id[industry]] = 1
                label_counts[label2id[industry]] += 1
        processed_data.append({"text": item["question"], "labels": labels})

    print("Label distribution:")
    for i, count in enumerate(label_counts):
        print(f"{id2label[i]}: {count}")

    return processed_data


# Preprocess data
classification_data = process_questions(questions_data, label2id)

# Print sample data for debugging
print("Sample data:")
print(classification_data[0])
print("Label type:", type(classification_data[0]["labels"][0]))

# Tokenizer and model
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)


def tokenize_and_encode(example):
    encoded = tokenizer(
        example["text"], padding="max_length", truncation=True, max_length=512
    )
    encoded["labels"] = example["labels"]
    return encoded


# Create dataset
dataset = Dataset.from_pandas(pd.DataFrame(classification_data))
tokenized_dataset = dataset.map(tokenize_and_encode, remove_columns=["text"])

# Print sample tokenized data for debugging
print("Sample tokenized data:")
print(tokenized_dataset[0])
print("Tokenized label type:", type(tokenized_dataset[0]["labels"][0]))

# Split dataset
train_val_test = tokenized_dataset.train_test_split(test_size=0.3)
train_dataset = train_val_test["train"]
val_test_dataset = train_val_test["test"].train_test_split(test_size=0.5)
val_dataset = val_test_dataset["train"]
test_dataset = val_test_dataset["test"]


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Lower the threshold for positive predictions
    predictions = (predictions > -0).astype(int)  # Changed from 0 to -0.5
    return {
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


# Model
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(label2id),
    problem_type="multi_label_classification",
    id2label=id2label,
    label2id=label2id,
).to(device)

# Training arguments
training_args = TrainingArguments(
    output_dir="simple_industry_classifier",
    learning_rate=2e-5,  # Slightly lower learning rate
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=4,  # Increase number of epochs
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="micro_f1",
    push_to_hub=False,
)


# Custom data collator
class MultiLabelDataCollator:
    def __call__(self, features):
        batch = {}
        for key in features[0].keys():
            if key != "labels":
                batch[key] = torch.tensor([f[key] for f in features])
            else:
                batch[key] = torch.tensor([f[key] for f in features], dtype=torch.float)
        return batch


# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    data_collator=MultiLabelDataCollator(),
)

# Train and evaluate
print("Starting training...")
trainer.train()
print("Training completed.")

print("\nEvaluating on validation set...")
val_metrics = trainer.evaluate()

print("\nValidation Metrics:")
for key, value in val_metrics.items():
    print(f"{key}: {value:.4f}")

print("\nEvaluating on test set...")
test_metrics = trainer.evaluate(test_dataset)

print("\nTest Metrics:")
for key, value in test_metrics.items():
    print(f"{key}: {value:.4f}")

# Save the model
print("\nSaving model...")
trainer.save_model("./simple_final_model")
print("Model saved.")


# Function to predict industries for new questions
def predict_industries(questions: List[str], model, tokenizer, id2label):
    model.eval()
    predictions = []
    for question in questions:
        inputs = tokenizer(
            question,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding="max_length",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        # Lower the threshold for predictions
        predicted_labels = torch.sigmoid(logits) > 0.35
        predicted_industries = [
            id2label[i] for i, pred in enumerate(predicted_labels[0]) if pred
        ]

        # If no industries are predicted, take the top 3
        if not predicted_industries:
            top_industries = torch.topk(logits, k=3).indices[0]
            predicted_industries = [id2label[i.item()] for i in top_industries]

        predictions.append(predicted_industries)
    return predictions


# Example usage
new_questions = [
    "What are the environmental impacts of water utilities?",
    "How do cruise lines manage their carbon emissions?",
]

predicted_industries = predict_industries(new_questions, model, tokenizer, id2label)

print("\nPredictions for new questions:")
for question, industries in zip(new_questions, predicted_industries):
    print(f"Question: {question}")
    print(f"Predicted Industries: {', '.join(industries)}")
    print()

print("Process completed!")
