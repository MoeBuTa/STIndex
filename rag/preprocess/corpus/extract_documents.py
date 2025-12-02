#!/usr/bin/env python3
"""
Extract and deduplicate documents from RAG datasets (HotpotQA, 2WikiMQA, MuSiQue).

This script processes the formatted QA datasets and extracts:
1. All unique documents (supporting_facts and retrieval_contexts)
2. Assigns document IDs based on (type, title) pairs
3. Creates mappings from QA pairs to document IDs
4. Merges all datasets into a unified corpus

Output structure:
    data/corpus/
    ├── hotpotqa/train/       # Per-dataset documents
    ├── two_wiki/train/
    ├── musique/train/
    └── documents.jsonl       # Merged corpus (all datasets)
"""

import argparse
import hashlib
import json
import os
from typing import Any, Dict, List, Set, Tuple

import jsonlines
from tqdm import tqdm


def generate_doc_id(contents: str) -> str:
    """Generate a unique document ID from content hash."""
    # Use first 12 chars of hash for shorter IDs
    return hashlib.md5(contents.encode()).hexdigest()[:12]


def extract_documents_from_sample(
    sample: Dict[str, Any],
    seen_docs: Dict[str, Dict[str, Any]],
) -> Tuple[List[str], List[str]]:
    """
    Extract documents from a single QA sample.

    Args:
        sample: The QA sample dict
        seen_docs: Dict of doc_id -> document (for deduplication)

    Returns:
        Tuple of (supporting_doc_ids, retrieval_doc_ids)
    """
    metadata = sample.get("metadata", {})

    supporting_doc_ids = []
    retrieval_doc_ids = []

    # Extract supporting facts (gold documents)
    supporting_facts = metadata.get("supporting_facts", [])
    for doc in supporting_facts:
        doc_type = doc.get("type", "unknown")
        title = doc.get("title", "")
        contents = doc.get("contents", "")

        if not contents:
            continue

        doc_id = generate_doc_id(contents)

        if doc_id not in seen_docs:
            seen_docs[doc_id] = {
                "doc_id": doc_id,
                "type": doc_type,
                "title": title,
                "contents": contents,
            }

        supporting_doc_ids.append(doc_id)

    # Extract retrieval contexts (all candidate documents)
    retrieval_contexts = metadata.get("retrieval_contexts", [])
    for doc in retrieval_contexts:
        doc_type = doc.get("type", "unknown")
        title = doc.get("title", "")
        contents = doc.get("contents", "")

        if not contents:
            continue

        doc_id = generate_doc_id(contents)

        if doc_id not in seen_docs:
            seen_docs[doc_id] = {
                "doc_id": doc_id,
                "type": doc_type,
                "title": title,
                "contents": contents,
            }

        retrieval_doc_ids.append(doc_id)

    return supporting_doc_ids, retrieval_doc_ids


def process_dataset(
    input_path: str,
    output_dir: str,
    dataset_name: str,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """
    Process a single dataset file and extract documents.

    Args:
        input_path: Path to input JSONL file
        output_dir: Directory to save output files
        dataset_name: Name of the dataset (for logging)

    Returns:
        Tuple of (statistics dict, seen_docs dict)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Storage for unique documents
    seen_docs: Dict[str, Dict[str, Any]] = {}

    # QA to document mappings
    qa_mappings: List[Dict[str, Any]] = []

    # Read input file
    print(f"Processing {dataset_name} from {input_path}...")

    with jsonlines.open(input_path, "r") as reader:
        samples = list(reader)

    # Extract documents from each sample
    for sample in tqdm(samples, desc=f"Extracting docs from {dataset_name}"):
        qa_id = sample.get("id", "")

        supporting_ids, retrieval_ids = extract_documents_from_sample(
            sample, seen_docs
        )

        qa_mappings.append({
            "qa_id": qa_id,
            "question": sample.get("question", ""),
            "answer_labels": sample.get("answer_labels", []),
            "supporting_doc_ids": supporting_ids,
            "retrieval_doc_ids": list(set(retrieval_ids)),  # dedupe within sample
        })

    # Write documents.jsonl (all unique documents)
    documents_path = os.path.join(output_dir, "documents.jsonl")
    with jsonlines.open(documents_path, "w") as writer:
        for doc_id in sorted(seen_docs.keys()):
            writer.write(seen_docs[doc_id])

    # Write qa_doc_mapping.jsonl
    mapping_path = os.path.join(output_dir, "qa_doc_mapping.jsonl")
    with jsonlines.open(mapping_path, "w") as writer:
        for mapping in qa_mappings:
            writer.write(mapping)

    # Calculate statistics
    stats = {
        "dataset": dataset_name,
        "total_qa_samples": len(samples),
        "unique_documents": len(seen_docs),
        "avg_supporting_per_qa": sum(len(m["supporting_doc_ids"]) for m in qa_mappings) / len(qa_mappings) if qa_mappings else 0,
        "avg_retrieval_per_qa": sum(len(m["retrieval_doc_ids"]) for m in qa_mappings) / len(qa_mappings) if qa_mappings else 0,
    }

    # Write stats
    stats_path = os.path.join(output_dir, "stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"  ✓ {stats['unique_documents']} unique documents extracted")
    print(f"  ✓ Avg {stats['avg_supporting_per_qa']:.2f} supporting docs per QA")

    return stats, seen_docs


def merge_corpus(
    all_docs_by_dataset: Dict[str, Dict[str, Dict[str, Any]]],
    output_dir: str,
) -> Dict[str, Any]:
    """
    Merge documents from multiple datasets into a unified corpus.

    Documents with the same (type, title) are deduplicated.

    Args:
        all_docs_by_dataset: Dict of dataset_name -> {doc_id -> doc}
        output_dir: Directory to save merged corpus

    Returns:
        Statistics dict
    """
    os.makedirs(output_dir, exist_ok=True)

    # Collect all unique documents
    merged_docs: Dict[str, Dict[str, Any]] = {}

    for dataset_name, docs in all_docs_by_dataset.items():
        for doc_id, doc in docs.items():
            if doc_id not in merged_docs:
                merged_docs[doc_id] = doc

    # Write merged documents
    merged_path = os.path.join(output_dir, "documents.jsonl")
    with jsonlines.open(merged_path, "w") as writer:
        for doc_id in sorted(merged_docs.keys()):
            writer.write(merged_docs[doc_id])

    stats = {
        "total_unique_documents": len(merged_docs),
        "documents_per_dataset": {
            name: len(docs) for name, docs in all_docs_by_dataset.items()
        },
    }

    stats_path = os.path.join(output_dir, "stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n=== Merged Corpus ===")
    print(f"  ✓ {stats['total_unique_documents']} total unique documents")
    for name, count in stats["documents_per_dataset"].items():
        print(f"    - {name}: {count}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Extract and deduplicate documents from RAG datasets"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["hotpotqa", "two_wiki", "musique"],
        help="Datasets to process",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/original",
        help="Input directory containing formatted datasets",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/corpus",
        help="Output directory for extracted documents",
    )

    args = parser.parse_args()

    # Process each dataset (train split only)
    all_docs_by_dataset: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for dataset in args.datasets:
        input_path = os.path.join(args.input_dir, dataset, "train.jsonl")

        if not os.path.exists(input_path):
            print(f"Warning: {input_path} not found, skipping...")
            continue

        output_dir = os.path.join(args.output_dir, dataset, "train")
        stats, seen_docs = process_dataset(input_path, output_dir, dataset)
        all_docs_by_dataset[dataset] = seen_docs

    # Merge all datasets into unified corpus
    if all_docs_by_dataset:
        merge_corpus(all_docs_by_dataset, args.output_dir)


if __name__ == "__main__":
    main()
