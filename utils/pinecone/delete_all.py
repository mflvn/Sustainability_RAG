import os

from dotenv import load_dotenv
from pinecone.grpc import PineconeGRPC as Pinecone


def delete_all_namespaces():
    load_dotenv()
    pine_key = os.getenv("PINECONE_API_KEY")
    pc = Pinecone(api_key=pine_key)

    index = pc.Index("sustain")

    # Get all namespaces
    stats = index.describe_index_stats()
    namespaces = ["sentence_window", "markdown", "hierarchical", "original"]

    # Delete vectors in each namespace
    for namespace in namespaces:
        print(f"Deleting vectors in namespace: {namespace}")
        index.delete(delete_all=True, namespace=namespace)

    # Print final stats
    print("Final index stats:")
    print(index.describe_index_stats())


if __name__ == "__main__":
    delete_all_namespaces()
