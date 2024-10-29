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
    "color_by": "industry_group",  # Updated to use industry_group
    "n_questions": 10000,
    "similarity_threshold": 0.99,
    "embeddings_file": "./qa_check_agents/similarity/question_embeddings.json",
    "output_csv": "./qa_check_agents/similarity/question_analysis_results.csv",
    "output_plot": "./qa_check_agents/similarity/question_embeddings_tsne_sample.pdf",
    "questions_directory": "./qa_experiments/prompt_free_local/all_qas.json",
    "batch_size": 10000,
}

# Industry groups mapping
INDUSTRY_GROUPS = {
    "Consumer Goods": [
        "b1-apparel-accessories-and-footwear",
        "b2-appliance-manufacturing",
        "b3-building-products-and-furnishings",
        "b4-e-commerce",
        "b5-household-and-personal-products",
        "b6-multiline-and-specialty-retailers-and-distributors",
    ],
    "Extractives and Minerals Processing": [
        "b7-coal-operations",
        "b8-construction-materials",
        "b9-iron-and-steel-producers",
        "b10-metals-and-mining",
        "b11-oil-and-gas-exploration-and-production",
        "b12-oil-and-gas-midstream",
        "b13-oil-and-gas-refining-and-marketing",
        "b14-oil-and-gas-services",
    ],
    "Financials": [
        "b15-asset-management-and-custody-activities",
        "b16-commercial-banks",
        "b17-insurance",
        "b18-investment-banking-and-brokerage",
        "b19-mortgage-finance",
    ],
    "Food and Beverage": [
        "b20-agricultural-products",
        "b21-alcoholic-beverages",
        "b22-food-retailers-and-distributors",
        "b23-meat-poultry-and-dairy",
        "b24-non-alcoholic-beverages",
        "b25-processed-foods",
        "b26-restaurants",
    ],
    "Health Care": [
        "b27-drug-retailers",
        "b28-health-care-delivery",
        "b29-health-care-distributors",
        "b30-managed-care",
        "b31-medical-equipment-and-supplies",
    ],
    "Infrastructure": [
        "b32-electric-utilities-and-power-generators",
        "b33-engineering-and-construction-services",
        "b34-gas-utilities-and-distributors",
        "b35-home-builders",
        "b36-real-estate",
        "b37-real-estate-services",
        "b38-waste-management",
        "b39-water-utilities-and-services",
    ],
    "Renewable Resources and Alternative Energy": [
        "b40-biofuels",
        "b41-forestry-management",
        "b42-fuel-cells-and-industrial-batteries",
        "b43-pulp-and-paper-products",
        "b44-solar-technology-and-project-developers",
        "b45-wind-technology-and-project-developers",
    ],
    "Resource Transformation": [
        "b46-aerospace-and-defense",
        "b47-chemicals",
        "b48-containers-and-packaging",
        "b49-electrical-and-electronic-equipment",
        "b50-industrial-machinery-and-goods",
    ],
    "Services": [
        "b51-casinos-and-gaming",
        "b52-hotels-and-lodging",
        "b53-leisure-facilities",
    ],
    "Technology and Communications": [
        "b54-electronic-manufacturing-services-and-original-design",
        "b55-hardware",
        "b56-internet-media-and-services",
        "b57-semiconductors",
        "b58-software-and-it-services",
        "b59-telecommunication-services",
    ],
    "Transportation": [
        "b60-air-freight-and-logistics",
        "b61-airlines",
        "b62-auto-parts",
        "b63-automobiles",
        "b64-car-rental-and-leasing",
        "b65-cruise-lines",
        "b66-marine-transportation",
        "b67-rail-transportation",
        "b68-road-transportation",
    ],
}


def load_data(directory_path: str, n_questions: int) -> pd.DataFrame:
    all_questions = []
    if not os.path.isdir(directory_path):
        # then its a file
        with open(directory_path, "r") as f:
            questions = json.load(f)
            all_questions.extend(questions)
    else:
        for filename in os.listdir(directory_path):
            if filename.endswith(".json"):
                with open(os.path.join(directory_path, filename), "r") as f:
                    questions = json.load(f)
                    all_questions.extend(questions)

    df = pd.DataFrame(all_questions).head(n_questions)
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


def perform_tsne(matrix: np.ndarray) -> np.ndarray:
    print("Performing t-SNE...")
    tsne = TSNE(
        n_components=2, perplexity=15, random_state=42, init="random", learning_rate=200
    )
    return tsne.fit_transform(matrix)


def remove_tsne_outliers(
    df: pd.DataFrame, outlier_percent: float = 0.05
) -> pd.DataFrame:
    df["distance"] = np.sqrt(df["tsne_1"] ** 2 + df["tsne_2"] ** 2)
    lower_threshold = df["distance"].quantile(outlier_percent)
    upper_threshold = df["distance"].quantile(1 - outlier_percent)
    df_filtered = df[
        (df["distance"] >= lower_threshold) & (df["distance"] <= upper_threshold)
    ]
    df_filtered = df_filtered.drop(columns=["distance"])
    print(
        f"Removed {len(df) - len(df_filtered)} outliers ({outlier_percent*200}% of data)"
    )
    return df_filtered.reset_index(drop=True)


def get_industry_group(industry: str) -> str:
    for group, industries in INDUSTRY_GROUPS.items():
        if industry in industries:
            return group
    return "Other"


def plot_tsne(df: pd.DataFrame, color_by: str, output_file: str):
    print(f"Generating plot colored by {color_by}...")
    plt.figure(figsize=(12, 8))

    if color_by == "industry_group":
        df["industry_group"] = df["industries"].apply(get_industry_group)
        categories = df["industry_group"].unique()
        colors = plt.cm.rainbow(np.linspace(0, 1, len(categories)))
        for category, color in zip(categories, colors):
            mask = df["industry_group"] == category
            plt.scatter(
                df.loc[mask, "tsne_1"],
                df.loc[mask, "tsne_2"],
                c=[color],
                label=category,
                alpha=0.6,
            )
        plt.legend(loc="best")
    elif color_by == "qa_type":
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

    plt.title(
        f"samplet-SNE visualization of question embeddings (colored by {color_by})"
    )
    plt.xlabel("t-SNE dimension 1")
    plt.ylabel("t-SNE dimension 2")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()


def calculate_statistics(data: List[float]) -> Dict[str, float]:
    return {
        "mean": mean(data),
        "median": median(data),
        "std_dev": stdev(data) if len(data) > 1 else 0,
    }


def main():
    df = load_data(config["questions_directory"], config["n_questions"])
    df = load_or_generate_embeddings(df, config["embeddings_file"])

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

    print("\nExamples of similar questions:")
    for original, similar in list(similar_questions.items())[:5]:
        print(f"Original: {original}")
        print(f"Similar: {similar[0]}")
        print()

    matrix = np.vstack(df.embedding.values)
    vis_dims = perform_tsne(matrix)
    df["tsne_1"] = vis_dims[:, 0]
    df["tsne_2"] = vis_dims[:, 1]

    df = remove_tsne_outliers(df)

    plot_tsne(df, config["color_by"], config["output_plot"].format(config["color_by"]))

    temperature_stats = calculate_statistics(df["temperature"].tolist())
    embedding_stats = calculate_statistics(
        [np.array(e).mean() for e in df["embedding"]]
    )

    print("\nTemperature Statistics:")
    print(json.dumps(temperature_stats, indent=2))
    print("\nEmbedding Statistics:")
    print(json.dumps(embedding_stats, indent=2))

    df.to_csv(config["output_csv"], index=False)

    print(
        f"\nAnalysis complete. Results saved to {config['output_csv']} and {config['output_plot'].format(config['color_by'])}"
    )


if __name__ == "__main__":
    main()
