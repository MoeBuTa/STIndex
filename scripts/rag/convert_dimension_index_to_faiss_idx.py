#!/usr/bin/env python3
"""
Convert dimensional index from doc_id (string UUID) to faiss_idx (integer).

This enables efficient FAISS pre-filtering using IDSelectorArray for
dimensional retrieval.

Before:
    dimension_index[dim]["labels"][label] = ["doc_uuid_1", "doc_uuid_2", ...]

After:
    dimension_index[dim]["labels"][label] = [0, 42, 103, ...]  # faiss indices

Usage:
    python -m scripts.rag.convert_dimension_index_to_faiss_idx \
        --passages-metadata data/indices/textbook_bgem3/passages_metadata.jsonl \
        --dimension-index data/indices/medcorp/indexes/dimension_index.json \
        --output data/indices/medcorp/indexes/dimension_index_faiss.json
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set

from tqdm import tqdm


def build_doc_id_to_faiss_idx(passages_metadata_path: Path) -> Dict[str, int]:
    """
    Build mapping from doc_id (string UUID) to faiss_idx (line number).

    Args:
        passages_metadata_path: Path to passages_metadata.jsonl

    Returns:
        Dict mapping doc_id -> faiss_idx
    """
    doc_id_to_idx = {}

    print(f"Building doc_id → faiss_idx mapping from {passages_metadata_path}...")

    with open(passages_metadata_path) as f:
        for idx, line in enumerate(tqdm(f, desc="Loading passages")):
            doc = json.loads(line)
            doc_id = doc.get("doc_id", doc.get("chunk_id", str(idx)))
            doc_id_to_idx[doc_id] = idx

    print(f"  Loaded {len(doc_id_to_idx):,} doc_id → faiss_idx mappings")
    return doc_id_to_idx


def convert_dimension_index(
    dimension_index: Dict,
    doc_id_to_idx: Dict[str, int],
) -> Dict:
    """
    Convert dimension index from doc_id to faiss_idx.

    Args:
        dimension_index: Original dimension index with doc_id strings
        doc_id_to_idx: Mapping from doc_id to faiss_idx

    Returns:
        Converted dimension index with faiss_idx integers
    """
    converted = {}
    stats = {
        "total_refs": 0,
        "matched_refs": 0,
        "missing_refs": 0,
        "missing_doc_ids": set(),
    }

    print(f"Converting {len(dimension_index)} dimensions...")

    for dim_name, dim_data in tqdm(dimension_index.items(), desc="Converting dimensions"):
        converted[dim_name] = {
            "labels": {},
            "paths": {},
            # Store reverse mapping for debugging
            "faiss_to_doc_id": {},
        }

        # Convert labels index
        labels = dim_data.get("labels", {})
        for label, doc_ids in labels.items():
            faiss_indices = []
            for doc_id in doc_ids:
                stats["total_refs"] += 1
                if doc_id in doc_id_to_idx:
                    faiss_idx = doc_id_to_idx[doc_id]
                    faiss_indices.append(faiss_idx)
                    stats["matched_refs"] += 1
                else:
                    stats["missing_refs"] += 1
                    stats["missing_doc_ids"].add(doc_id)

            if faiss_indices:
                # Sort for efficient binary search
                converted[dim_name]["labels"][label] = sorted(faiss_indices)

        # Convert paths index
        paths = dim_data.get("paths", {})
        for path, doc_ids in paths.items():
            faiss_indices = []
            for doc_id in doc_ids:
                if doc_id in doc_id_to_idx:
                    faiss_indices.append(doc_id_to_idx[doc_id])

            if faiss_indices:
                converted[dim_name]["paths"][path] = sorted(faiss_indices)

    # Print stats
    print(f"\nConversion Statistics:")
    print(f"  Total references: {stats['total_refs']:,}")
    print(f"  Matched: {stats['matched_refs']:,} ({100*stats['matched_refs']/max(1,stats['total_refs']):.1f}%)")
    print(f"  Missing: {stats['missing_refs']:,} ({100*stats['missing_refs']/max(1,stats['total_refs']):.1f}%)")

    if stats["missing_doc_ids"]:
        print(f"  Missing doc_ids (first 5): {list(stats['missing_doc_ids'])[:5]}")

    return converted


def main():
    parser = argparse.ArgumentParser(
        description="Convert dimensional index from doc_id to faiss_idx"
    )
    parser.add_argument(
        "--passages-metadata",
        required=True,
        help="Path to passages_metadata.jsonl (defines faiss_idx order)"
    )
    parser.add_argument(
        "--dimension-index",
        required=True,
        help="Path to dimension_index.json (input)"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output dimension_index_faiss.json"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate conversion by sampling"
    )

    args = parser.parse_args()

    passages_path = Path(args.passages_metadata)
    dimension_path = Path(args.dimension_index)
    output_path = Path(args.output)

    if not passages_path.exists():
        print(f"Error: Passages metadata not found: {passages_path}")
        return 1

    if not dimension_path.exists():
        print(f"Error: Dimension index not found: {dimension_path}")
        return 1

    # Build mapping
    doc_id_to_idx = build_doc_id_to_faiss_idx(passages_path)

    # Load dimension index
    print(f"Loading dimension index from {dimension_path}...")
    with open(dimension_path) as f:
        dimension_index = json.load(f)

    # Convert
    converted_index = convert_dimension_index(dimension_index, doc_id_to_idx)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nWriting converted index to {output_path}...")
    with open(output_path, "w") as f:
        json.dump(converted_index, f)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"  Output size: {size_mb:.2f} MB")

    # Validation
    if args.validate:
        print("\nValidating conversion...")
        # Load both indexes
        with open(dimension_path) as f:
            orig = json.load(f)
        with open(output_path) as f:
            conv = json.load(f)

        # Check a sample
        for dim_name in list(orig.keys())[:3]:
            labels = list(orig[dim_name].get("labels", {}).keys())[:2]
            for label in labels:
                orig_doc_ids = orig[dim_name]["labels"].get(label, [])
                conv_faiss_ids = conv[dim_name]["labels"].get(label, [])

                print(f"  {dim_name}/{label}:")
                print(f"    Original: {len(orig_doc_ids)} doc_ids")
                print(f"    Converted: {len(conv_faiss_ids)} faiss_ids")

                # Verify first few
                for i, doc_id in enumerate(orig_doc_ids[:3]):
                    expected_idx = doc_id_to_idx.get(doc_id)
                    if expected_idx is not None and expected_idx in conv_faiss_ids:
                        print(f"    ✓ {doc_id} → {expected_idx}")
                    else:
                        print(f"    ✗ {doc_id} → {expected_idx} (not in converted)")

    print("\n✓ Conversion complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
