import os
from typing import Dict, List, Optional, Tuple, Union

from llama_index.core import VectorStoreIndex
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.indices.query.query_transform.base import (
    StepDecomposeQueryTransform,
)
from llama_index.core.query_engine import (
    MultiStepQueryEngine,
    RetrieverQueryEngine,
    TransformQueryEngine,
)
from llama_index.core.vector_stores import (
    FilterOperator,
    MetadataFilter,
    MetadataFilters,
)
from llama_index.core.vector_stores.types import VectorStoreQueryMode
from llama_index.llms.together import TogetherLLM
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone
from pydantic.v1 import BaseModel, Field

from chatbot.prompting import IndustryClassificationRetriever


def map_industry_to_code(industry: str, label_mapping: Dict[str, str]) -> str:
    for code, label in label_mapping.items():
        if industry.lower() in label.replace("-", " "):  # Simple matching for now
            return code
    return None  # Return None if not found


class QueryFailedException(Exception):
    pass


class CorrectOptionOutput(BaseModel):
    correctOption: str = Field(
        ...,
        description="The correct option for the question, a single character from A to D.",
    )


class FreeTextOutput(BaseModel):
    text: str = Field(
        ...,
        description="Free text response to the query.",
    )


class ModelWrapper:
    MODEL_INFO: Dict[str, Tuple[str, int]] = {
        # "2b": ("google/gemma-2b-it", 2),
        "8B": ("meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo", 8),
        # "13B": ("meta-llama/Llama-2-13b-chat-hf", 13),
        # "70B": ("meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo", 70),
        # "mixtral-8x7B": ("mistralai/Mixtral-8x7B-Instruct-v0.1", 56),  # 8
        "finetuned-8B": (
            "dfsf/Meta-Llama-3.1-8B-Instruct-Reference-2024-08-11-16-13-33-bbac2fc3",
            8,
        ),
    }

    QUERY_MODES = {
        "default": VectorStoreQueryMode.DEFAULT,
        "hybrid": VectorStoreQueryMode.HYBRID,
        "semantic_hybrid": VectorStoreQueryMode.SEMANTIC_HYBRID,
        "svm": VectorStoreQueryMode.SVM,
        "linear_regression": VectorStoreQueryMode.LINEAR_REGRESSION,
        "mmr": VectorStoreQueryMode.MMR,
    }

    RAG_TRANSFORMS = {
        "none": None,
        "hyde": "hyde",
        "multi": "multi",
    }

    CHUNKING_NAMESPACES = {
        "sentence_512": "sentence_512",
        "sentence_1024": "sentence_1024",
        "semantic": "semantic",
        "original": "original",
        "sentence_window": "sentence_window",
        "hierarchical": "hierarchical",
        "markdown": "markdown",
        "sentence_256": "sentence_256",
    }

    def __init__(
        self,
        model_size: str = "8B",
        output_type: str = "correctOption",
        similarity_top_k: int = 5,
        vector_store_query_mode: str = "default",
        chunking_namespace: str = "sentence_512",
        rag_transform: str = "none",
        return_industries_only: bool = False,
    ):
        self.TOGETHER_API_KEY: Optional[str] = os.getenv("TOGETHER_API_KEY")
        self.model_size = model_size
        self.output_type = output_type
        self.similarity_top_k = similarity_top_k
        self.vector_store_query_mode = vector_store_query_mode
        self.model_name, self.model_size_b = self.MODEL_INFO.get(
            model_size, self.MODEL_INFO["8B"]
        )
        self.llm = self._initialize_llm()
        self.chunking_namespace = self.CHUNKING_NAMESPACES.get(
            chunking_namespace, "sentence_512"
        )
        self.rag_transform = self.RAG_TRANSFORMS.get(rag_transform, None)

        self.index = self._initialize_index()
        self.structured_query_engine, self.unstructured_query_engine = (
            self._initialize_query_engines()
        )
        self.return_industries_only = return_industries_only
        self.industry_retriever = IndustryClassificationRetriever()

    def _initialize_llm(self):
        return TogetherLLM(model=self.model_name, api_key=self.TOGETHER_API_KEY)

    def _initialize_index(self):
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index_name = "sustain"
        pinecone_index = pc.Index(
            index_name,
        )
        vector_store = PineconeVectorStore(
            pinecone_index=pinecone_index, namespace=self.chunking_namespace
        )
        return VectorStoreIndex.from_vector_store(vector_store)

    def _initialize_query_engines(self):
        output_cls = (
            CorrectOptionOutput
            if self.output_type == "correctOption"
            else FreeTextOutput
        )
        sllm = self.llm.as_structured_llm(output_cls=output_cls)

        query_mode = self.QUERY_MODES.get(
            self.vector_store_query_mode, VectorStoreQueryMode.DEFAULT
        )

        base_structured_query_engine = self.index.as_query_engine(
            llm=sllm,
            similarity_top_k=self.similarity_top_k,
            vector_store_query_mode=query_mode,
        )
        base_unstructured_query_engine = self.index.as_query_engine(
            llm=self.llm,
            similarity_top_k=self.similarity_top_k,
            vector_store_query_mode=query_mode,
        )

        if self.rag_transform == "hyde":
            hyde = HyDEQueryTransform(include_original=True)
            structured_query_engine = TransformQueryEngine(
                base_structured_query_engine, hyde
            )
            unstructured_query_engine = TransformQueryEngine(
                base_unstructured_query_engine, hyde
            )
        elif self.rag_transform == "multi":
            step_decompose_transform = StepDecomposeQueryTransform(
                llm=self.llm, verbose=True
            )
            index_summary = "Used to answer questions about the given topic"
            structured_query_engine = MultiStepQueryEngine(
                query_engine=base_structured_query_engine,
                query_transform=step_decompose_transform,
                index_summary=index_summary,
            )
            unstructured_query_engine = MultiStepQueryEngine(
                query_engine=base_unstructured_query_engine,
                query_transform=step_decompose_transform,
                index_summary=index_summary,
            )
        else:
            structured_query_engine = base_structured_query_engine
            unstructured_query_engine = base_unstructured_query_engine

        return structured_query_engine, unstructured_query_engine

    def _filter_chunks_by_industry(self, query: str):
        relevant_industries = self.industry_retriever._identify_industries(query)
        # replace dashes with spaces and capitalize
        relevant_industries = [industry.replace("-", " ").capitalize() for industry in relevant_industries]
        # Create metadata filters
        filters = MetadataFilters(
            filters=[
                MetadataFilter(
                    key="industry",
                    operator=FilterOperator.IN,
                    value=relevant_industries
                )
            ]
        )
        return filters

    def query_unstructured(
        self, query_text: str, filters: Optional[MetadataFilters] = None
    ) -> Union[Tuple[str, List[str]], List[str]]:
        try:
            # Create filters based on identified industries
            # filters = self._filter_chunks_by_industry(query_text)

            # Create a retriever with the filters
            retriever = self.index.as_retriever(
                # filters=filters,
                llm=self.llm,
                similarity_top_k=self.similarity_top_k,
            )

            # Create a query engine with the filtered retriever
            query_engine = RetrieverQueryEngine(
                retriever=retriever,
            )

            # Apply RAG transform if necessary
            if self.rag_transform == "hyde":
                hyde = HyDEQueryTransform(include_original=True)
                query_engine = TransformQueryEngine(query_engine, hyde)
            elif self.rag_transform == "multi":
                step_decompose_transform = StepDecomposeQueryTransform(
                    llm=self.llm, verbose=True
                )
                index_summary = "Used to answer questions about the given topic"
                query_engine = MultiStepQueryEngine(
                    query_engine=query_engine,
                    query_transform=step_decompose_transform,
                    index_summary=index_summary,
                )

            # Query using the engine
            result = query_engine.query(query_text)

            predicted_industries = []

            # Extract industries from metadata of result nodes
            try:
                for node in result.source_nodes:
                    industries = node.metadata.get("industries", [])
                    predicted_industries.extend(industries)
            except Exception as e:
                print(f"Error extracting metadata: {e}")

            # Remove duplicates
            predicted_industries = list(set(predicted_industries))

            if self.return_industries_only:
                return predicted_industries
            else:
                return result.response.strip(), predicted_industries
        except Exception as e:
            raise QueryFailedException(f"Unstructured query failed: {str(e)}")

    def query_structured(self, query_text: str) -> Tuple[str, List[str]]:
        try:
            # Create filters based on identified industries
            # filters = self._filter_chunks_by_industry(query_text)

            # Create a retriever with the filters
            retriever = self.index.as_retriever(
                # filters=filters,
                similarity_top_k=self.similarity_top_k,
                vector_store_query_mode=self.QUERY_MODES.get(
                    self.vector_store_query_mode, VectorStoreQueryMode.DEFAULT
                ),
            )

            # Create a structured query engine with the filtered retriever
            query_engine = self.index.as_query_engine(
                retriever=retriever,
                llm=self.llm.as_structured_llm(output_cls=CorrectOptionOutput),
            )

            # Apply RAG transform if necessary
            if self.rag_transform == "hyde":
                hyde = HyDEQueryTransform(include_original=True)
                query_engine = TransformQueryEngine(query_engine, hyde)
            elif self.rag_transform == "multi":
                step_decompose_transform = StepDecomposeQueryTransform(
                    llm=self.llm, verbose=True
                )
                index_summary = "Used to answer questions about the given topic"
                query_engine = MultiStepQueryEngine(
                    query_engine=query_engine,
                    query_transform=step_decompose_transform,
                    index_summary=index_summary,
                )

            # Query using the engine
            result = query_engine.query(query_text)

            industries = []
            try:
                for node in result.source_nodes:
                    industries.extend(node.metadata.get("industries", []))
            except Exception as e:
                print(f"Error extracting metadata: {e}")

            industries = list(set(industries))  # Remove duplicates
            return result.correctOption, industries
        except Exception as e:
            raise QueryFailedException(f"Structured query failed: {str(e)}")

    def get_model_info(self) -> Dict[str, Union[str, int]]:
        return {"model_name": self.model_name, "model_size": self.model_size_b}
