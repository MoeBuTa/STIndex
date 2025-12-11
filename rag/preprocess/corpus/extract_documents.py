#!/usr/bin/env python3
"""
Extract documents and questions from RAG datasets (HotpotQA, 2WikiMQA, MuSiQue).

This script processes the formatted QA datasets and extracts:
1. All unique documents (deduplicated by content hash)
2. All questions with references to supporting documents

Output structure:
    data/corpus/
    ├── documents.jsonl           # Merged corpus (all datasets)
    ├── hotpotqa/train/
    ├── two_wiki/train/
    └── musique/train/

    data/questions/
    ├── questions.jsonl           # Merged questions (all datasets)
    ├── hotpotqa/train/
    ├── two_wiki/train/
    └── musique/train/
"""

import argparse
import hashlib
import json
import os
from typing import Any, Dict, List, Tuple

import jsonlines
from tqdm import tqdm


def generate_doc_id(contents: str) -> str:
    """Generate a unique document ID from content hash."""
    return hashlib.md5(contents.encode()).hexdigest()[:12]


def process_dataset(
    input_path: str,
    corpus_output_dir: str,
    questions_output_dir: str,
    dataset_name: str,
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Process a single dataset file and extract documents and questions.

    Args:
        input_path: Path to input JSONL file
        corpus_output_dir: Directory to save corpus output files
        questions_output_dir: Directory to save questions output files
        dataset_name: Name of the dataset (for logging)

    Returns:
        Tuple of (statistics dict, seen_docs dict, questions list)
    """
    os.makedirs(corpus_output_dir, exist_ok=True)
    os.makedirs(questions_output_dir, exist_ok=True)

    # Storage for unique documents
    seen_docs: Dict[str, Dict[str, Any]] = {}
    # Storage for questions
    questions: List[Dict[str, Any]] = []

    print(f"Processing {dataset_name} from {input_path}...")

    with jsonlines.open(input_path, "r") as reader:
        samples = list(reader)

    for sample in tqdm(samples, desc=f"Extracting from {dataset_name}"):
        metadata = sample.get("metadata", {})

        # Extract supporting facts (gold documents)
        for doc in metadata.get("supporting_facts", []):
            title = doc.get("title", "")
            contents = doc.get("contents", "")

            if not contents:
                continue

            doc_id = generate_doc_id(contents)

            if doc_id not in seen_docs:
                seen_docs[doc_id] = {
                    "doc_id": doc_id,
                    "title": title,
                    "contents": contents,
                }

        # Extract retrieval contexts (all candidate documents)
        for doc in metadata.get("retrieval_contexts", []):
            title = doc.get("title", "")
            contents = doc.get("contents", "")

            if not contents:
                continue

            doc_id = generate_doc_id(contents)

            if doc_id not in seen_docs:
                seen_docs[doc_id] = {
                    "doc_id": doc_id,
                    "title": title,
                    "contents": contents,
                }

        # Extract question
        questions.append({
            "question_id": sample.get("id", ""),
            "question": sample.get("question", ""),
        })

    # Write documents.jsonl
    documents_path = os.path.join(corpus_output_dir, "documents.jsonl")
    with jsonlines.open(documents_path, "w") as writer:
        for doc_id in sorted(seen_docs.keys()):
            writer.write(seen_docs[doc_id])

    # Write questions.jsonl
    questions_path = os.path.join(questions_output_dir, "questions.jsonl")
    with jsonlines.open(questions_path, "w") as writer:
        for q in questions:
            writer.write(q)

    # Calculate statistics
    stats = {
        "dataset": dataset_name,
        "unique_documents": len(seen_docs),
        "total_questions": len(questions),
    }

    # Write stats to both directories
    for output_dir in [corpus_output_dir, questions_output_dir]:
        stats_path = os.path.join(output_dir, "stats.json")
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

    print(f"  ✓ {stats['unique_documents']} unique documents")
    print(f"  ✓ {stats['total_questions']} questions")

    return stats, seen_docs, questions


def merge_corpus(
    all_docs_by_dataset: Dict[str, Dict[str, Dict[str, Any]]],
    output_dir: str,
) -> Dict[str, Any]:
    """Merge documents from multiple datasets into a unified corpus."""
    os.makedirs(output_dir, exist_ok=True)

    merged_docs: Dict[str, Dict[str, Any]] = {}
    for dataset_name, docs in all_docs_by_dataset.items():
        for doc_id, doc in docs.items():
            if doc_id not in merged_docs:
                merged_docs[doc_id] = doc

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


def merge_questions(
    all_questions_by_dataset: Dict[str, List[Dict[str, Any]]],
    output_dir: str,
) -> Dict[str, Any]:
    """Merge questions from multiple datasets."""
    os.makedirs(output_dir, exist_ok=True)

    all_questions: List[Dict[str, Any]] = []
    for questions in all_questions_by_dataset.values():
        all_questions.extend(questions)

    merged_path = os.path.join(output_dir, "questions.jsonl")
    with jsonlines.open(merged_path, "w") as writer:
        for q in all_questions:
            writer.write(q)

    stats = {
        "total_questions": len(all_questions),
        "questions_per_dataset": {
            name: len(questions) for name, questions in all_questions_by_dataset.items()
        },
    }

    stats_path = os.path.join(output_dir, "stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n=== Merged Questions ===")
    print(f"  ✓ {stats['total_questions']} total questions")
    for name, count in stats["questions_per_dataset"].items():
        print(f"    - {name}: {count}")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Extract documents and questions from RAG datasets"
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
        "--corpus-dir",
        type=str,
        default="data/corpus",
        help="Output directory for extracted documents",
    )
    parser.add_argument(
        "--questions-dir",
        type=str,
        default="data/questions",
        help="Output directory for extracted questions",
    )

    args = parser.parse_args()

    all_docs_by_dataset: Dict[str, Dict[str, Dict[str, Any]]] = {}
    all_questions_by_dataset: Dict[str, List[Dict[str, Any]]] = {}

    for dataset in args.datasets:
        input_path = os.path.join(args.input_dir, dataset, "train.jsonl")

        if not os.path.exists(input_path):
            print(f"Warning: {input_path} not found, skipping...")
            continue

        corpus_output_dir = os.path.join(args.corpus_dir, dataset, "train")
        questions_output_dir = os.path.join(args.questions_dir, dataset, "train")

        stats, seen_docs, questions = process_dataset(
            input_path, corpus_output_dir, questions_output_dir, dataset
        )
        all_docs_by_dataset[dataset] = seen_docs
        all_questions_by_dataset[dataset] = questions

    # Merge all datasets
    if all_docs_by_dataset:
        merge_corpus(all_docs_by_dataset, args.corpus_dir)
    if all_questions_by_dataset:
        merge_questions(all_questions_by_dataset, args.questions_dir)


if __name__ == "__main__":
    main()
