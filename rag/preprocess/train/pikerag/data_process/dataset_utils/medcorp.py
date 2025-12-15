# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

"""
MedCorp dataset loader - Textbooks + StatPearls corpus
Download medical textbooks and StatPearls snippets from HuggingFace for RAG.

Corpus composition:
- Textbooks: 125,847 snippets (~1.2 GB)
- StatPearls: 301,202 snippets (~6.2 GB)
Total: 427,049 snippets (~7.4 GB)

Note: StatPearls requires accepting license terms on HuggingFace.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import uuid

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("Warning: 'datasets' package not found. Install with: pip install datasets")

from loguru import logger


def load_raw_data(dataset_dir: str, split: str) -> List[dict]:
    """
    Load MedCorp corpus (Textbooks + StatPearls)

    Strategy:
    - Textbooks: Load from HuggingFace (pre-chunked, available)
    - StatPearls: Load from local preprocessed file (NCBI source, manual download required)

    Args:
        dataset_dir: Directory to cache downloaded data
        split: Split name (only 'train' is supported)

    Returns:
        List of document dictionaries
    """
    logger.info("Loading MedCorp corpus (Textbooks + StatPearls)...")

    # Create cache directory
    cache_dir = os.path.join(dataset_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    all_documents = []

    # ========== Load Textbooks from HuggingFace ==========
    logger.info("  [1/2] Loading Textbooks corpus from HuggingFace...")
    if not DATASETS_AVAILABLE:
        logger.warning("      ✗ 'datasets' package not available")
        logger.warning("      Install with: pip install datasets")
    else:
        try:
            textbooks_dataset = load_dataset(
                "MedRAG/textbooks",
                split="train",
                cache_dir=cache_dir,
                trust_remote_code=True
            )
            logger.info(f"      ✓ Loaded {len(textbooks_dataset)} textbook snippets")

            for doc in textbooks_dataset:
                doc_entry = doc.copy()
                doc_entry["source_corpus"] = "textbooks"
                all_documents.append(doc_entry)

        except Exception as e:
            logger.error(f"      ✗ Failed to load Textbooks: {e}")
            logger.error(f"      Check: https://huggingface.co/datasets/MedRAG/textbooks")

    # ========== Load StatPearls from local file ==========
    logger.info("  [2/2] Loading StatPearls corpus from local file...")
    statpearls_path = Path(dataset_dir) / "raw" / "statpearls_processed.jsonl"

    if statpearls_path.exists():
        logger.info(f"      Found: {statpearls_path}")
        try:
            statpearls_count = 0
            with open(statpearls_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        doc = json.loads(line)
                        # doc already has source_corpus="statpearls" from preprocessing
                        all_documents.append(doc)
                        statpearls_count += 1

            logger.info(f"      ✓ Loaded {statpearls_count} StatPearls snippets")

        except Exception as e:
            logger.error(f"      ✗ Failed to load StatPearls from {statpearls_path}: {e}")

    else:
        logger.warning(f"      ✗ StatPearls not found: {statpearls_path}")
        logger.warning("      To download and preprocess StatPearls:")
        logger.warning("        1. python -m rag.preprocess.corpus.download_statpearls")
        logger.warning("        2. python -m rag.preprocess.corpus.preprocess_statpearls")
        logger.warning("")
        logger.warning("      StatPearls cannot be distributed via HuggingFace due to NCBI policy.")
        logger.warning("      See: https://huggingface.co/datasets/MedRAG/statpearls")

    logger.info("=" * 80)
    logger.info(f"✓ MedCorp corpus loaded: {len(all_documents)} total snippets")

    # Show breakdown
    textbooks_count = sum(1 for d in all_documents if d.get("source_corpus") == "textbooks")
    statpearls_count = sum(1 for d in all_documents if d.get("source_corpus") == "statpearls")
    logger.info(f"  Textbooks: {textbooks_count:,} snippets")
    logger.info(f"  StatPearls: {statpearls_count:,} snippets")
    logger.info("=" * 80)

    return all_documents


def format_raw_data(raw: dict) -> Optional[dict]:
    """
    Format MedCorp document to match pikerag protocol

    Input format (MedRAG):
    {
        "id": "snippet_id",
        "title": "Document title",
        "content": "Snippet text",
        "contents": "Title + content combined",
        "source_corpus": "textbooks" or "statpearls"
    }

    Output format (pikerag protocol):
    {
        "doc_id": "uuid",
        "title": "...",
        "contents": "...",
        "metadata": {
            "source_corpus": "textbooks",
            "original_id": "...",
        }
    }
    """
    # Extract fields
    original_id = raw.get("id", "")
    title = raw.get("title", "")
    contents = raw.get("contents", raw.get("content", ""))
    source_corpus = raw.get("source_corpus", "medcorp")

    if not contents:
        # Skip empty documents
        return None

    # Format according to pikerag protocol
    formatted_data = {
        "doc_id": uuid.uuid4().hex,
        "title": title,
        "contents": contents,
        "metadata": {
            "source_corpus": source_corpus,
            "original_id": original_id,
            "type": "medical_corpus"
        }
    }

    return formatted_data
