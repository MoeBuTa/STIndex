"""
STIndex Retrieval Module.

Provides hybrid retrieval combining vector similarity search with
dimensional filtering (temporal, spatial, categorical).

Components:
- HybridRetriever: Main retriever combining FAISS + STIndex filters
- DimensionalFilter: Filter chunks by temporal/spatial dimensions
- RetrievalResult: Unified result format

Usage:
    from stindex.retrieval import HybridRetriever

    retriever = HybridRetriever(
        vector_index_path="data/vector/rag",
        stindex_warehouse_path="data/warehouse/rag",
    )

    # Pure vector search
    results = retriever.search("What happened in 2022?", k=10)

    # Filtered search (temporal + vector)
    results = retriever.search(
        "What happened in Perth?",
        k=10,
        temporal_filter={"year": 2022},
        spatial_filter={"region": "Australia"},
    )
"""

from .hybrid_retriever import HybridRetriever, RetrievalResult, RetrievalResponse
from .dimensional_filter import DimensionalFilter, TemporalFilter, SpatialFilter

__all__ = [
    "HybridRetriever",
    "RetrievalResult",
    "RetrievalResponse",
    "DimensionalFilter",
    "TemporalFilter",
    "SpatialFilter",
]
