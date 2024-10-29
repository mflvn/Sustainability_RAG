import os

from dotenv import load_dotenv
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone

pine_key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=pine_key)
from llama_index.core.vector_stores import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)

index = pc.Index("sustain")


def print_all_vectors_metadata():
    load_dotenv()

    # Get all namespaces
    stats = index.describe_index_stats()
    namespaces = stats.namespaces.keys()

    # Fetch and print metadata for vectors in each namespace
    for namespace in namespaces:
        print(f"\nNamespace: {namespace}")
        # vector must be 1536 dimension
    a = index.query(
        namespace=namespace, vector=[0.1] * 1536, top_k=100, include_values=False
    )
    print(a)


def queryTest():
    index = pc.Index("sustain")

    # Create PineconeVectorStore
    vector_store = PineconeVectorStore(pinecone_index=index, namespace="semantic")

    # Create VectorStoreIndex from the existing vector store
    index = VectorStoreIndex.from_vector_store(vector_store)


    # answer = retriever.retrieve("What is inception about?")

    # print([i.get_content()[:22] for i in answer])
    # print(len(answer))

    base_unstructured_query_engine = index.as_query_engine(
        
    )
    print(base_unstructured_query_engine.query("What is inception about?"))


if __name__ == "__main__":
    queryTest()
