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

import os
from typing import Dict, List, Optional

import uuid

try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print("Warning: 'datasets' package not found. Install with: pip install datasets")


def load_raw_data(dataset_dir: str, split: str) -> List[dict]:
    """
    Load MedCorp corpus (Textbooks + StatPearls) from HuggingFace

    Args:
        dataset_dir: Directory to cache downloaded data
        split: Split name (only 'train' is supported)

    Returns:
        List of document dictionaries
    """
    if not DATASETS_AVAILABLE:
        raise ImportError("'datasets' package required. Install with: pip install datasets")

    print(f"Loading MedCorp corpus (Textbooks + StatPearls)...")

    # Create cache directory
    cache_dir = os.path.join(dataset_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    all_documents = []

    # Load Textbooks corpus
    print("  [1/2] Loading Textbooks corpus from HuggingFace...")
    try:
        textbooks_dataset = load_dataset(
            "MedRAG/textbooks",
            split="train",
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        print(f"      ✓ Loaded {len(textbooks_dataset)} textbook snippets")

        for doc in textbooks_dataset:
            doc_entry = doc.copy()
            doc_entry["source_corpus"] = "textbooks"
            all_documents.append(doc_entry)

    except Exception as e:
        print(f"      ✗ Failed to load Textbooks: {e}")
        print(f"      Note: Check HuggingFace access at https://huggingface.co/datasets/MedRAG/textbooks")

    # Load StatPearls corpus
    print("  [2/2] Loading StatPearls corpus from HuggingFace...")
    print("      Note: StatPearls requires accepting license terms on HuggingFace")
    print("      Visit: https://huggingface.co/datasets/MedRAG/statpearls")
    try:
        statpearls_dataset = load_dataset(
            "MedRAG/statpearls",
            split="train",
            cache_dir=cache_dir,
            trust_remote_code=True
        )
        print(f"      ✓ Loaded {len(statpearls_dataset)} StatPearls snippets")

        for doc in statpearls_dataset:
            doc_entry = doc.copy()
            doc_entry["source_corpus"] = "statpearls"
            all_documents.append(doc_entry)

    except Exception as e:
        print(f"      ✗ Failed to load StatPearls: {e}")
        print(f"      This is likely due to license requirements.")
        print(f"      Please:")
        print(f"        1. Visit https://huggingface.co/datasets/MedRAG/statpearls")
        print(f"        2. Accept the license terms")
        print(f"        3. Login with: huggingface-cli login")

    print(f"\n✓ MedCorp corpus loaded: {len(all_documents)} total snippets")
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
