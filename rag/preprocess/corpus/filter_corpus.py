#!/usr/bin/env python3
"""
Filter corpus to documents relevant to specific questions using BM25.

Unlike extract_documents.py which processes multi-hop QA datasets with
supporting_facts, this module handles evaluation-only datasets (like MIRAGE)
that need retrieval-based corpus filtering.

Usage:
    python -m rag.preprocess.corpus.filter_corpus \
        --questions data/original/mirage/train.jsonl \
        --corpus data/original/medcorp/train.jsonl \
        --output data/corpus/mirage_filtered.jsonl \
        --top-k 10
"""

import argparse
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Set
from tqdm import tqdm
from rank_bm25 import BM25Okapi


def generate_doc_id(contents: str) -> str:
    """Generate a unique document ID from content hash."""
    return hashlib.md5(contents.encode()).hexdigest()[:12]


def load_questions(questions_path: str) -> List[str]:
    """Load questions from JSONL file."""
    questions = []
    with open(questions_path) as f:
        for line in f:
            data = json.loads(line)
            questions.append(data.get('question', ''))
    return [q for q in questions if q]  # Filter empty


def load_corpus(corpus_path: str) -> List[Dict[str, Any]]:
    """Load corpus from JSONL file."""
    corpus = []
    with open(corpus_path) as f:
        for line in f:
            data = json.loads(line)
            # Ensure doc_id exists
            if 'doc_id' not in data:
                data['doc_id'] = generate_doc_id(data.get('contents', ''))
            corpus.append(data)
    return corpus


def filter_corpus_bm25(
    questions: List[str],
    corpus: List[Dict[str, Any]],
    top_k: int = 10
) -> Set[int]:
    """
    Filter corpus using BM25 retrieval.

    Args:
        questions: List of question strings
        corpus: List of corpus documents
        top_k: Number of top documents to retrieve per question

    Returns:
        Set of corpus indices that are relevant to questions
    """
    # Build BM25 index
    tokenized_corpus = [doc.get('contents', '').split() for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)

    # Retrieve relevant docs for each question
    relevant_indices = set()
    for question in tqdm(questions, desc="Retrieving relevant documents"):
        scores = bm25.get_scores(question.split())
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]
        relevant_indices.update(top_indices)

    return relevant_indices


def main():
    parser = argparse.ArgumentParser(description="Filter corpus for evaluation questions")
    parser.add_argument("--questions", required=True, help="Path to questions JSONL")
    parser.add_argument("--corpus", required=True, help="Path to corpus JSONL")
    parser.add_argument("--output", required=True, help="Path to output filtered corpus")
    parser.add_argument("--top-k", type=int, default=10,
                       help="Top-k documents per question (default: 10)")
    parser.add_argument("--stats-output", help="Path to save statistics JSON")
    args = parser.parse_args()

    # Load data
    print(f"Loading questions from {args.questions}...")
    questions = load_questions(args.questions)
    print(f"  Loaded {len(questions):,} questions")

    print(f"\nLoading corpus from {args.corpus}...")
    corpus = load_corpus(args.corpus)
    print(f"  Loaded {len(corpus):,} documents")

    # Filter corpus
    print(f"\nFiltering corpus (top-{args.top_k} per question)...")
    relevant_indices = filter_corpus_bm25(questions, corpus, args.top_k)
    print(f"  Found {len(relevant_indices):,} relevant documents")
    print(f"  Reduction: {len(corpus):,} → {len(relevant_indices):,} " +
          f"({100 * len(relevant_indices) / len(corpus):.1f}%)")

    # Save filtered corpus
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for idx in sorted(relevant_indices):
            f.write(json.dumps(corpus[idx]) + '\n')

    print(f"\n✓ Filtered corpus saved to: {output_path}")

    # Save statistics
    stats = {
        "original_corpus_size": len(corpus),
        "filtered_corpus_size": len(relevant_indices),
        "num_questions": len(questions),
        "top_k_per_question": args.top_k,
        "reduction_ratio": len(relevant_indices) / len(corpus),
    }

    if args.stats_output:
        stats_path = Path(args.stats_output)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"✓ Statistics saved to: {stats_path}")

    print("\nCorpus filtering complete!")


if __name__ == "__main__":
    main()
