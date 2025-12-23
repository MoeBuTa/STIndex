#!/usr/bin/env python3
"""
Build RAG-compatible chunks_metadata.jsonl from extraction results.

Converts extraction output format to the format expected by:
- HybridRetriever (stindex/retrieval/hybrid_retriever.py)
- DimensionalFilter (stindex/retrieval/dimensional_filter.py)
- RAGRetriever (rag/retriever/retriever.py)

Usage:
    python -m scripts.rag.build_dimensional_metadata \
        --extraction data/extraction_results_parallel/corpus_extraction_worker1.jsonl \
        --output data/indices/medcorp/chunks_metadata.jsonl
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List
from tqdm import tqdm


def normalize_dimension_name(name: str) -> str:
    """
    Normalize dimension names for consistent indexing.

    'diagnostic standard' -> 'diagnostic_standard'
    'Organ System' -> 'organ_system'
    """
    return name.strip().lower().replace(" ", "_")


def convert_extraction_to_metadata(doc: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a single extraction result to chunks_metadata format.

    Input format:
    {
        "doc_id": "abc123",
        "doc_metadata": {"title": "...", "source": "...", "corpus": "..."},
        "text": "...",
        "schema_metadata": {
            "diagnostic standard": [["radiography", "before interpretation"]],
            "drug": [["aspirin", "NSAID"]]
        }
    }

    Output format:
    {
        "doc_id": "abc123",
        "doc_title": "...",
        "text": "...",
        "dimensions": {
            "diagnostic_standard": [["radiography", "before interpretation"]],
            "drug": [["aspirin", "NSAID"]]
        },
        "temporal_labels": [],
        "spatial_labels": []
    }
    """
    doc_id = doc.get("doc_id", "")
    doc_metadata = doc.get("doc_metadata", {})
    schema_metadata = doc.get("schema_metadata", {})

    # Normalize dimension names and filter empty values
    dimensions = {}
    for dim_name, entities in schema_metadata.items():
        if entities:  # Only include non-empty dimensions
            normalized_name = normalize_dimension_name(dim_name)
            dimensions[normalized_name] = entities

    return {
        "doc_id": doc_id,
        "doc_title": doc_metadata.get("title", ""),
        "text": doc.get("text", ""),
        "dimensions": dimensions,
        "temporal_labels": [],  # Empty - temporal disabled
        "spatial_labels": [],   # Empty - spatial disabled
        "source": doc_metadata.get("source", "unknown"),
        "corpus": doc_metadata.get("corpus", "unknown")
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build RAG-compatible chunks_metadata.jsonl from extraction results"
    )
    parser.add_argument(
        "--extraction",
        required=True,
        help="Path to extraction results JSONL file"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output chunks_metadata.jsonl"
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="Limit number of documents to process (for testing)"
    )

    args = parser.parse_args()

    extraction_path = Path(args.extraction)
    output_path = Path(args.output)

    if not extraction_path.exists():
        print(f"Error: Extraction file not found: {extraction_path}")
        return 1

    # Create output directory
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Count lines for progress bar
    print(f"Counting documents in {extraction_path}...")
    with open(extraction_path) as f:
        total_lines = sum(1 for _ in f)

    if args.sample_limit:
        total_lines = min(total_lines, args.sample_limit)

    print(f"Processing {total_lines:,} documents...")

    # Track statistics
    stats = {
        "total_docs": 0,
        "docs_with_dimensions": 0,
        "dimension_counts": {},
        "unique_labels_per_dimension": {}
    }

    with open(extraction_path) as infile, open(output_path, 'w') as outfile:
        for i, line in enumerate(tqdm(infile, total=total_lines, desc="Converting")):
            if args.sample_limit and i >= args.sample_limit:
                break

            doc = json.loads(line)
            metadata = convert_extraction_to_metadata(doc)

            # Update statistics
            stats["total_docs"] += 1
            if metadata["dimensions"]:
                stats["docs_with_dimensions"] += 1

                for dim_name, entities in metadata["dimensions"].items():
                    if dim_name not in stats["dimension_counts"]:
                        stats["dimension_counts"][dim_name] = 0
                        stats["unique_labels_per_dimension"][dim_name] = set()

                    stats["dimension_counts"][dim_name] += len(entities)

                    # Collect unique labels (all hierarchy levels)
                    for hierarchy in entities:
                        if isinstance(hierarchy, list):
                            for label in hierarchy:
                                stats["unique_labels_per_dimension"][dim_name].add(label)

            outfile.write(json.dumps(metadata) + '\n')

    # Print statistics
    print(f"\n{'=' * 60}")
    print(f"Conversion Complete!")
    print(f"{'=' * 60}")
    print(f"Total documents: {stats['total_docs']:,}")
    print(f"Documents with dimensions: {stats['docs_with_dimensions']:,}")
    print(f"Coverage: {100 * stats['docs_with_dimensions'] / stats['total_docs']:.1f}%")
    print(f"\nDimension statistics:")

    # Sort by count
    sorted_dims = sorted(
        stats["dimension_counts"].items(),
        key=lambda x: x[1],
        reverse=True
    )

    for dim_name, count in sorted_dims[:20]:  # Top 20
        unique_labels = len(stats["unique_labels_per_dimension"].get(dim_name, set()))
        print(f"  {dim_name}: {count:,} entities, {unique_labels:,} unique labels")

    if len(sorted_dims) > 20:
        print(f"  ... and {len(sorted_dims) - 20} more dimensions")

    print(f"\nOutput: {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
