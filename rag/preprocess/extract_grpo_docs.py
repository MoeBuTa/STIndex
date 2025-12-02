#!/usr/bin/env python3
"""
Extract unique documents from GRPO dataset for RAG ingestion.

This script extracts all unique documents embedded in the GRPO training data
and outputs them as chunks for STIndex extraction and vector ingestion.

Usage:
    python -m rag.preprocess.extract_grpo_docs \
        --input data/data_train/grpo/grpo_25000.jsonl \
        --output data/corpus/grpo/chunks.jsonl
"""

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

from loguru import logger
from tqdm import tqdm


def parse_documents_from_content(content: str) -> List[Tuple[int, str, str]]:
    """
    Parse documents from user message content.

    Returns:
        List of (doc_num, title, text) tuples
    """
    documents = []

    if '<documents>' not in content:
        return documents

    docs_section = content.split('<documents>')[1].split('</documents>')[0]

    # Match [N] Document Title: Content pattern
    # Pattern: [num] Title: Content (until next [num] or end)
    pattern = r'\[(\d+)\]\s*([^:]+):\s*(.+?)(?=\[\d+\]|$)'
    matches = re.findall(pattern, docs_section, re.DOTALL)

    for doc_num, title, text in matches:
        title = title.strip()
        text = text.strip()
        if text:
            documents.append((int(doc_num), title, text))

    return documents


def generate_doc_id(title: str, text: str) -> str:
    """Generate unique document ID from title and text prefix."""
    key = f"{title}:{text[:100]}"
    return hashlib.md5(key.encode()).hexdigest()[:12]


def extract_documents(
    input_path: str,
    output_path: str,
    limit: int = None,
) -> Dict:
    """
    Extract unique documents from GRPO dataset.

    Args:
        input_path: Path to GRPO JSONL file
        output_path: Path to output chunks JSONL
        limit: Limit number of samples to process

    Returns:
        Statistics dict
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    unique_docs = {}  # doc_id -> doc_data
    total_samples = 0
    total_doc_refs = 0

    logger.info(f"Reading from {input_path}")

    with open(input_path, 'r') as f:
        for i, line in enumerate(tqdm(f, desc="Extracting documents")):
            if limit and i >= limit:
                break

            sample = json.loads(line)
            total_samples += 1

            # Get user message content
            user_content = sample['messages'][1]['content']

            # Parse documents
            docs = parse_documents_from_content(user_content)

            for doc_num, title, text in docs:
                total_doc_refs += 1
                doc_id = generate_doc_id(title, text)

                if doc_id not in unique_docs:
                    unique_docs[doc_id] = {
                        'chunk_id': f"{doc_id}_c0000",
                        'doc_id': doc_id,
                        'chunk_index': 0,
                        'total_chunks': 1,
                        'text': text,
                        'doc_type': 'wikipedia',
                        'doc_title': title,
                    }

    # Write unique documents
    logger.info(f"Writing {len(unique_docs):,} unique documents to {output_path}")

    with open(output_path, 'w') as f:
        for doc_data in unique_docs.values():
            f.write(json.dumps(doc_data, ensure_ascii=False) + '\n')

    stats = {
        'total_samples': total_samples,
        'total_doc_refs': total_doc_refs,
        'unique_documents': len(unique_docs),
        'output_path': str(output_path),
    }

    # Save stats
    stats_path = output_path.parent / 'extraction_stats.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    logger.success(f"Extracted {len(unique_docs):,} unique documents from {total_samples:,} samples")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Extract documents from GRPO dataset")
    parser.add_argument(
        "--input",
        type=str,
        default="data/data_train/grpo/grpo_25000.jsonl",
        help="Input GRPO JSONL file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/corpus/grpo/chunks.jsonl",
        help="Output chunks JSONL file",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples to process",
    )

    args = parser.parse_args()

    stats = extract_documents(
        input_path=args.input,
        output_path=args.output,
        limit=args.limit,
    )

    print(f"\n=== Extraction Complete ===")
    print(f"Samples processed: {stats['total_samples']:,}")
    print(f"Document references: {stats['total_doc_refs']:,}")
    print(f"Unique documents: {stats['unique_documents']:,}")
    print(f"Output: {stats['output_path']}")


if __name__ == "__main__":
    main()
