"""
RAG Retriever Module.

Provides high-level retrieval interface for RAG pipelines,
wrapping STIndex HybridRetriever with additional features:
- Query expansion and rewriting
- Multi-hop retrieval for complex questions
- Evidence aggregation and deduplication
- Context formatting for LLM input

Usage:
    from rag.retriever import RAGRetriever, RAGRetrieverConfig

    retriever = RAGRetriever(config=RAGRetrieverConfig(
        vector_index_path="data/vector/rag",
        warehouse_path="data/warehouse/rag",
    ))

    # Single-hop retrieval
    context = retriever.retrieve("What is the capital of France?", k=5)

    # Multi-hop retrieval
    context = retriever.retrieve_multihop(
        question="Who is the director of the movie that won Best Picture in 2020?",
        max_hops=2,
    )
"""

from .retriever import RAGRetriever, RAGRetrieverConfig, RetrievalContext
from .query_processor import QueryProcessor
from .rrf_retriever import RRFRetriever, RetrievalResult as RRFRetrievalResult
from .three_stage_retriever import ThreeStageRetriever, RetrievalResult

__all__ = [
    "RAGRetriever",
    "RAGRetrieverConfig",
    "RetrievalContext",
    "QueryProcessor",
    "RRFRetriever",
    "RRFRetrievalResult",
    "ThreeStageRetriever",
    "RetrievalResult",
]
