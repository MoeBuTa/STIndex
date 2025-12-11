"""
Ingestion module for RAG corpus preprocessing.

This module provides two types of ingestion:
1. STIndex Ingestion: Dimensional entity extraction (temporal, spatial)
2. Vector Ingestion: FAISS-based embedding storage for semantic retrieval

Usage:
    # STIndex ingestion
    from rag.preprocess.ingestion import STIndexCorpusIngester

    ingester = STIndexCorpusIngester(config_path="extract")
    stats = ingester.ingest("data/corpus/merged/chunks.jsonl")

    # Vector ingestion
    from rag.preprocess.ingestion import VectorIngester

    ingester = VectorIngester(config_path="rag/cfg/ingestion.yaml")
    ingester.ingest("data/corpus/merged/chunks.jsonl")
"""

from .stindex_ingest import STIndexCorpusIngester
from .vector_ingest import VectorIngester

__all__ = [
    "STIndexCorpusIngester",
    "VectorIngester",
]
