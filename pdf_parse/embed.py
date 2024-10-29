import json
import os
import asyncio
import pickle
from typing import Dict, List
from dotenv import load_dotenv
from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.extractors import TitleExtractor
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.node_parser import (
    HierarchicalNodeParser,
    MarkdownNodeParser,
    SentenceWindowNodeParser,
)
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

load_dotenv()
pine_key = os.getenv("PINECONE_API_KEY")
openai_key = os.getenv("OPENAI_API_KEY")
pc = Pinecone(api_key=pine_key)

# Create or get the Pinecone index
index_name = "sustain"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="dotproduct",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

pinecone_index = pc.Index(index_name)

# Initialize embedding model
embed_model = OpenAIEmbedding(api_key=openai_key)

# Create different IngestionPipelines with various splitters
pipelines = {
    "sentence_window": IngestionPipeline(
        transformations=[
            SentenceWindowNodeParser(chunk_size=1024,),
            TitleExtractor(),
            embed_model,
        ]
    ),
    "markdown": IngestionPipeline(
        transformations=[
            MarkdownNodeParser(),
            TitleExtractor(),
            embed_model,
        ],
    ),
    "hierarchical": IngestionPipeline(
        transformations=[
            HierarchicalNodeParser.from_defaults(chunk_sizes=[2048, 512, 128]),
            TitleExtractor(),
            embed_model,
        ]
    ),
}

async def process_document(doc: Document, pipeline_name: str) -> List[Document]:
    nodes = await asyncio.to_thread(pipelines[pipeline_name].run, documents=[doc])
    return [
        Document(
            text=node.text,
            metadata={
                **doc.metadata,
                **node.metadata,
                "chunk_type": pipeline_name,
                "namespace": pipeline_name,
            },
        )
        for node in nodes
    ]

async def process_json_file(file_path: str, first_json: dict, first_page: int) -> Dict[str, List[Document]]:
    with open(file_path, "r") as file:
        json_data = json.load(file)

    documents: Dict[str, List[Document]] = {name: [] for name in pipelines.keys()}
    documents["original"] = []

    if file_path.endswith("tables.json"):
        for table_type in ["sustainability_metrics_table", "activity_metrics_table", "other_text"]:
            if table_type in json_data:
                documents["original"].append(
                    Document(
                        text=json_data[table_type],
                        metadata={
                            "type": table_type.replace("_table", ""),
                            "namespace": "original",
                        },
                    )
                )
    else:
        doc = Document(
            text=json_data["text_content"],
            metadata={
                "filename": os.path.basename(file_path),
                "page_number": json_data["page_number"],
                "type": "text",
                "namespace": "original",
            },
        )
        documents["original"].append(doc)

        # Process the document content for chunked versions
        tasks = [process_document(doc, name) for name in pipelines.keys()]
        results = await asyncio.gather(*tasks)
        
        for name, result in zip(pipelines.keys(), results):
            documents[name].extend(result)

    # Add general metadata to all documents
    for doc_list in documents.values():
        for doc in doc_list:
            doc.metadata.update({
                "report_title": first_json.get("report_title", ""),
                "industry": first_json.get("industry", ""),
            })
            if doc.metadata["type"] in ["sustainability_metrics", "activity_metrics"]:
                doc.metadata["page_number"] = first_page

    return documents

async def process_folder(folder_path: str) -> Dict[str, List[Document]]:
    all_documents: Dict[str, List[Document]] = {name: [] for name in pipelines.keys()}
    all_documents["original"] = []

    json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    json_files.sort()  # Ensure consistent processing order

    # Process tables.json first to get the first_json data
    tables_json_path = os.path.join(folder_path, "tables.json")
    with open(tables_json_path, "r") as file:
        first_json = json.load(file)

    # Find the first page number
    first_page = min(
        json.load(open(os.path.join(folder_path, f), "r"))["page_number"]
        for f in json_files
        if f != "tables.json"
    )

    # Process all JSON files concurrently
    tasks = [process_json_file(os.path.join(folder_path, f), first_json, first_page) for f in json_files]
    results = await asyncio.gather(*tasks)

    # Combine results
    for result in results:
        for key in all_documents.keys():
            all_documents[key].extend(result[key])

    return all_documents

async def save_documents(documents: Dict[str, List[Document]], folder_name: str):
    os.makedirs("processed_documents", exist_ok=True)
    file_path = os.path.join("processed_documents", f"{folder_name}.pkl")
    with open(file_path, "wb") as f:
        pickle.dump(documents, f)

async def load_documents(folder_name: str) -> Dict[str, List[Document]]:
    file_path = os.path.join("processed_documents", f"{folder_name}.pkl")
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            return pickle.load(f)
    return None

async def process_and_save_folder(folder_name: str, folder_path: str) -> Dict[str, List[Document]]:
    documents = await load_documents(folder_name)
    if documents is None:
        print(f"Processing folder: {folder_name}")
        documents = await process_folder(folder_path)
        await save_documents(documents, folder_name)
    return documents

async def process_industry_folders(output_folder: str, max_concurrent: int = 5):
    all_documents: Dict[str, List[Document]] = {name: [] for name in pipelines.keys()}
    all_documents["original"] = []

    # Create a semaphore to limit concurrent processing
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_folder_with_semaphore(folder_name: str, folder_path: str):
        async with semaphore:
            return await process_and_save_folder(folder_name, folder_path)

    tasks = []
    for folder_name in os.listdir(output_folder):
        folder_path = os.path.join(output_folder, folder_name)
        if os.path.isdir(folder_path):
            tasks.append(process_folder_with_semaphore(folder_name, folder_path))

    results = await asyncio.gather(*tasks)

    for folder_documents in results:
        for key in all_documents.keys():
            all_documents[key].extend(folder_documents[key])

    return all_documents

async def main():
    output_folder = "./markdowns_copy"
    
    print("Processing industry folders...")
    all_documents = await process_industry_folders(output_folder)

    # Create Pinecone vector store and index for each namespace
    for namespace, documents in all_documents.items():
        print(f"Processing namespace: {namespace}")
        vector_store = PineconeVectorStore(
            pinecone_index=pinecone_index,
            add_sparse_vector=True,
            namespace=namespace,
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
        print(f"Created index for namespace: {namespace}")

if __name__ == "__main__":
    asyncio.run(main())