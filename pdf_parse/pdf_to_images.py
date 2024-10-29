import json
import os
from statistics import mean, median, stdev
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from openai import OpenAI
from scipy.spatial.distance import cosine
from sklearn.manifold import TSNE
from tqdm import tqdm

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Configuration dictionary
config = {
    "color_by": "qa_type",  # Options: "qa_type", "temperature", "industry"
    "n_questions": 1000,  # Number of questions to analyze
    "similarity_threshold": 0.9,  # Threshold for deleting similar questions
    "embeddings_file": "./qa_check_agents/similarity/question_embeddings.json",
    "output_csv": "./qa_check_agents/similarity/question_analysis_results.csv",
    "output_plot": "./qa_check_agents/similarity/question_embeddings_tsne_{}.png",
    "questions_file": "./qa_check_agents/similarity/filtered_questions.json",
    "batch_size": 300,  # Number of questions to embed in a single API call
}


def load_data(file_path: str, n_questions: int) -> pd.DataFrame:
    with open(file_path, "r") as f:
        questions = json.load(f)
    df = pd.DataFrame(questions).head(n_questions)
    return df.explode("industries")


def get_embeddings_batch(texts: List[str]) -> List[List[float]]:
    response = client.embeddings.create(input=texts, model="text-embedding-3-small")
    return [data.embedding for data in response.data]


def load_or_generate_embeddings(df: pd.DataFrame, embeddings_file: str) -> pd.DataFrame:
    if os.path.exists(embeddings_file):
        with open(embeddings_file, "r") as f:
            embeddings = json.load(f)
        df["embedding"] = [np.array(emb) for emb in embeddings]
    else:
        all_embeddings = []
        for i in tqdm(
            range(0, len(df), config["batch_size"]), desc="Generating embeddings"
        ):
            batch = df["question"].iloc[i : i + config["batch_size"]].tolist()
            batch_embeddings = get_embeddings_batch(batch)
            all_embeddings.extend(batch_embeddings)

        df["embedding"] = all_embeddings
        with open(embeddings_file, "w") as f:
            json.dump(all_embeddings, f)
    return df


def find_similar_questions(
    df: pd.DataFrame, threshold: float
) -> Tuple[pd.DataFrame, List[int], Dict[str, List[int]]]:
    to_delete = set()
    similar_questions = {}
    identical_questions = {}

    for i in range(len(df)):
        if i in to_delete:
            continue

        question1 = df.iloc[i]["question"]
        embedding1 = df.iloc[i]["embedding"]

        if question1 not in identical_questions:
            identical_questions[question1] = [i]
        else:
            identical_questions[question1].append(i)
            continue

        for j in range(i + 1, len(df)):
            if j in to_delete:
                continue

            question2 = df.iloc[j]["question"]
            embedding2 = df.iloc[j]["embedding"]

            if question1 == question2:
                identical_questions[question1].append(j)
                to_delete.add(j)
            else:
                similarity = 1 - cosine(embedding1, embedding2)
                if similarity > threshold:
                    to_delete.add(j)
                    if question1 not in similar_questions:
                        similar_questions[question1] = []
                    similar_questions[question1].append(question2)

    kept_indices = [i for i in range(len(df)) if i not in to_delete]
    return (
        df.iloc[kept_indices].reset_index(drop=True),
        list(to_delete),
        identical_questions,
        similar_questions,
    )


def save_filtered_json(
    original_file: str, indices_to_delete: List[int], output_file: str
):
    with open(original_file, "r") as f:
        original_data = json.load(f)

    filtered_data = [
        q for i, q in enumerate(original_data) if i not in indices_to_delete
    ]

    with open(output_file, "w") as f:
        json.dump(filtered_data, f, indent=2)


def perform_tsne(matrix: np.ndarray) -> np.ndarray:
    print("Performing t-SNE...")
    tsne = TSNE(
        n_components=2, perplexity=15, random_state=42, init="random", learning_rate=200
    )
    return tsne.fit_transform(matrix)


def plot_tsne(df: pd.DataFrame, color_by: str, output_file: str):
    print(f"Generating plot colored by {color_by}...")
    plt.figure(figsize=(12, 8))

    if color_by in ["qa_type", "industries"]:
        categories = df[color_by].unique()
        colors = plt.cm.rainbow(np.linspace(0, 1, len(categories)))
        for category, color in zip(categories, colors):
            mask = df[color_by] == category
            plt.scatter(
                df.loc[mask, "tsne_1"],
                df.loc[mask, "tsne_2"],
                c=[color],
                label=category,
                alpha=0.6,
            )
        plt.legend()
    elif color_by == "temperature":
        scatter = plt.scatter(
            df["tsne_1"], df["tsne_2"], c=df["temperature"], cmap="viridis", alpha=0.6
        )
        plt.colorbar(scatter, label="Temperature")
    else:
        raise ValueError(f"Invalid color_by option: {color_by}")

    plt.title(f"t-SNE visualization of question embeddings (colored by {color_by})")
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    plt.savefig(output_file)
    plt.close()


def calculate_statistics(data: List[float]) -> Dict[str, float]:
    return {
        "mean": mean(data),
        "median": median(data),
        "std_dev": stdev(data) if len(data) > 1 else 0,
    }


def main():
    # Load data
    df = load_data(config["questions_file"], config["n_questions"])

    # Load or generate embeddings
    df = load_or_generate_embeddings(df, config["embeddings_file"])

    # Find similar questions and delete them
    original_count = len(df)
    print(f"Loaded {original_count} questions.")
    df, deleted_indices, identical_questions, similar_questions = (
        find_similar_questions(df, config["similarity_threshold"])
    )
    deleted_count = original_count - len(df)

    print(f"\nDeleted {deleted_count} questions:")
    print(
        f"- Similar questions (threshold: {config['similarity_threshold']}): {len(similar_questions)}"
    )
    print(
        f"- Identical questions: {sum(len(group) - 1 for group in identical_questions.values() if len(group) > 1)}"
    )

    # Print some examples of similar questions
    print("\nExamples of similar questions:")
    for original, similar in list(similar_questions.items())[
        :5
    ]:  # Show first 5 examples
        print(f"Original: {original}")
        print(f"Similar: {similar[0]}")
        print()

    # Save filtered JSON
    filtered_json_output = "./qa_check_agents/similarity/filtered_questions.json"
    save_filtered_json(config["questions_file"], deleted_indices, filtered_json_output)
    print(f"\nSaved filtered questions to {filtered_json_output}")

    # Perform t-SNE for visualization
    matrix = np.vstack(df.embedding.values)
    vis_dims = perform_tsne(matrix)
    df["tsne_1"] = vis_dims[:, 0]
    df["tsne_2"] = vis_dims[:, 1]

    # Plot t-SNE results
    plot_tsne(df, config["color_by"], config["output_plot"].format(config["color_by"]))

    # Calculate statistics
    temperature_stats = calculate_statistics(df["temperature"].tolist())
    embedding_stats = calculate_statistics(
        [np.array(e).mean() for e in df["embedding"]]
    )

    print("\nTemperature Statistics:")
    print(json.dumps(temperature_stats, indent=2))
    print("\nEmbedding Statistics:")
    print(json.dumps(embedding_stats, indent=2))

    # Save results to CSV
    df.to_csv(config["output_csv"], index=False)

    print(
        f"\nAnalysis complete. Results saved to {config['output_csv']} and {config['output_plot'].format(config['color_by'])}"
    )


if __name__ == "__main__":
    main()
