"""
Corpus preprocessing module for RAG datasets.

This module provides tools for extracting and deduplicating documents from QA datasets.

Usage:
    # Extract documents from formatted datasets (train only, merged corpus)
    python -m rag.preprocess.corpus.extract_documents
"""

from .extract_documents import (
    generate_doc_id,
    process_dataset,
    merge_corpus,
)

__all__ = [
    "generate_doc_id",
    "process_dataset",
    "merge_corpus",
]
