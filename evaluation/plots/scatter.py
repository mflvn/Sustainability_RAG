import glob
import os
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def get_latest_csv(folder_path: str) -> str:
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {folder_path}")
    return max(csv_files, key=os.path.getctime)


def create_scatter_plot(experiment_type: str):
    experiment_types = {
        "1": "model_size",
        "2": "similarity_top_k",
        "3": "retriever",
        "4": "chunking_namespace",
    }

    if experiment_type in experiment_types:
        experiment_type = experiment_types[experiment_type]
    elif experiment_type not in experiment_types.values():
        raise ValueError(f"Invalid experiment type: {experiment_type}")

    # Define paths
    base_dir = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    input_folder = os.path.join(base_dir, "final_experiment_results", experiment_type)
    output_folder = os.path.join(base_dir, "evaluation", "plots", "output")

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Get the latest CSV file
    csv_file_path = get_latest_csv(input_folder)

    # Generate output file name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file_path = os.path.join(
        output_folder, f"{experiment_type}_accuracy_scatter_{timestamp}.png"
    )

    # Read the CSV file
    df = pd.read_csv(csv_file_path)

    # Filter for the specified experiment type
    df_filtered = df[df["Experiment Type"] == experiment_type]

    if df_filtered.empty:
        raise ValueError(f"No data found for experiment type: {experiment_type}")

    # Create the scatter plot
    plt.figure(figsize=(12, 8))
    scatter = sns.scatterplot(
        data=df_filtered,
        x="Parameter",
        y="Accuracy",
        hue="Model Name",
        size="Model Size (B)",
        sizes=(20, 200),
        alpha=0.7,
    )

    # Customize the plot
    plt.title(
        f"{experiment_type.capitalize()} Experiment: Accuracy vs Parameter", fontsize=16
    )
    plt.xlabel("Parameter", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)

    # Adjust x-axis based on experiment type
    if experiment_type == "model_size":
        plt.xscale("log")
        plt.xlabel("Model Size (Billion Parameters)", fontsize=12)
    elif experiment_type == "similarity_top_k":
        plt.xscale("linear")
        plt.xlabel("Top-K Value", fontsize=12)
    elif experiment_type == "retriever":
        plt.xlabel("Retriever Technique", fontsize=12)
    elif experiment_type == "chunking_namespace":
        plt.xlabel("Chunking Namespace", fontsize=12)

    # Adjust legend
    plt.legend(title="Model", bbox_to_anchor=(1.05, 1), loc="upper left")

    # Add labels for each point
    for _, row in df_filtered.iterrows():
        plt.annotate(
            str(row["Parameter"]),
            (row["Parameter"], row["Accuracy"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=10,
            alpha=0.8,
        )

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_file_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Scatter plot saved to {output_file_path}")


# Usage examples
if __name__ == "__main__":
    print("Choose an experiment to plot:")
    print("1. Model ")
    print("2. Similarity Top-K")
    print("3. Retriever Technique")
    print("4. Chunking Namespace")

    choice = input("Enter your choice (1, 2, 3, or 4): ")
    create_scatter_plot(choice)
