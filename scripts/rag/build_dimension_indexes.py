#!/usr/bin/env python3
"""
Build hierarchical inverted indexes for dimensional filtering.

Creates indexes used by DimensionalFilter for pre/post-filtering
during hybrid retrieval.

Hierarchy Indexing Strategy:
For each entity with hierarchy ["radiography", "before interpretation", "plain radiographs"]:
- Index at level 0: "radiography" → doc_id
- Index at level 1: "before interpretation" → doc_id
- Index at level 2: "plain radiographs" → doc_id
- Also index the full path: "radiography > before interpretation > plain radiographs" → doc_id

Usage:
    python -m scripts.rag.build_dimension_indexes \
        --chunks-metadata data/indices/medcorp/chunks_metadata.jsonl \
        --output-dir data/indices/medcorp/indexes
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Set, List
from tqdm import tqdm


def build_hierarchy_path(hierarchy: List[str]) -> str:
    """Build a path string from hierarchy list."""
    return " > ".join(hierarchy)


def index_hierarchy(
    hierarchy: List[str],
    doc_id: str,
    label_index: Dict[str, Set[str]],
    path_index: Dict[str, Set[str]]
) -> None:
    """
    Index a single hierarchy at all levels.

    Args:
        hierarchy: List of labels from general to specific, e.g., ["radiography", "CT scan"]
        doc_id: Document ID to index
        label_index: Dict mapping label -> set of doc_ids
        path_index: Dict mapping full path -> set of doc_ids
    """
    # Index each individual label
    for label in hierarchy:
        if label:  # Skip empty labels
            label_lower = label.lower().strip()
            label_index[label_lower].add(doc_id)

    # Index the full path
    if hierarchy:
        full_path = build_hierarchy_path([l.lower().strip() for l in hierarchy if l])
        if full_path:
            path_index[full_path].add(doc_id)


def main():
    parser = argparse.ArgumentParser(
        description="Build hierarchical inverted indexes for dimensional filtering"
    )
    parser.add_argument(
        "--chunks-metadata",
        required=True,
        help="Path to chunks_metadata.jsonl"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write index files"
    )

    args = parser.parse_args()

    metadata_path = Path(args.chunks_metadata)
    output_dir = Path(args.output_dir)

    if not metadata_path.exists():
        print(f"Error: Metadata file not found: {metadata_path}")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    # Count lines for progress bar
    print(f"Counting documents in {metadata_path}...")
    with open(metadata_path) as f:
        total_lines = sum(1 for _ in f)

    print(f"Building indexes from {total_lines:,} documents...")

    # Dimension index structure:
    # {
    #   "dimension_name": {
    #     "labels": {"label": ["doc_id1", "doc_id2"]},
    #     "paths": {"path > to > label": ["doc_id1"]}
    #   }
    # }
    dimension_indexes = defaultdict(lambda: {
        "labels": defaultdict(set),
        "paths": defaultdict(set)
    })

    # Process each document
    with open(metadata_path) as f:
        for line in tqdm(f, total=total_lines, desc="Indexing"):
            doc = json.loads(line)
            doc_id = doc.get("doc_id", doc.get("chunk_id", ""))
            dimensions = doc.get("dimensions", {})

            for dim_name, entities in dimensions.items():
                if not entities:
                    continue

                for entity in entities:
                    if isinstance(entity, list):
                        # Hierarchy format: ["level1", "level2", "level3"]
                        index_hierarchy(
                            entity,
                            doc_id,
                            dimension_indexes[dim_name]["labels"],
                            dimension_indexes[dim_name]["paths"]
                        )
                    elif isinstance(entity, dict):
                        # Dict format with text/category fields
                        text = entity.get("text", entity.get("category", ""))
                        if text:
                            dimension_indexes[dim_name]["labels"][text.lower().strip()].add(doc_id)
                    elif isinstance(entity, str):
                        # Simple string format
                        dimension_indexes[dim_name]["labels"][entity.lower().strip()].add(doc_id)

    # Convert sets to lists for JSON serialization
    print("Converting to JSON format...")
    dimension_index_json = {}
    for dim_name, index_data in dimension_indexes.items():
        dimension_index_json[dim_name] = {
            "labels": {k: sorted(list(v)) for k, v in index_data["labels"].items()},
            "paths": {k: sorted(list(v)) for k, v in index_data["paths"].items()}
        }

    # Write dimension index
    dimension_index_path = output_dir / "dimension_index.json"
    print(f"Writing dimension index to {dimension_index_path}...")
    with open(dimension_index_path, 'w') as f:
        json.dump(dimension_index_json, f, indent=2)

    # Write empty temporal and spatial indexes (since they're disabled)
    temporal_index_path = output_dir / "temporal_index.json"
    spatial_index_path = output_dir / "spatial_index.json"

    with open(temporal_index_path, 'w') as f:
        json.dump({}, f)

    with open(spatial_index_path, 'w') as f:
        json.dump({}, f)

    # Print statistics
    print(f"\n{'=' * 60}")
    print(f"Index Building Complete!")
    print(f"{'=' * 60}")
    print(f"Total dimensions indexed: {len(dimension_indexes)}")

    # Sort by number of unique labels
    sorted_dims = sorted(
        dimension_index_json.items(),
        key=lambda x: len(x[1]["labels"]),
        reverse=True
    )

    print(f"\nDimension index statistics:")
    for dim_name, index_data in sorted_dims[:20]:
        num_labels = len(index_data["labels"])
        num_paths = len(index_data["paths"])
        total_docs = sum(len(v) for v in index_data["labels"].values())
        print(f"  {dim_name}: {num_labels:,} labels, {num_paths:,} paths, {total_docs:,} doc refs")

    if len(sorted_dims) > 20:
        print(f"  ... and {len(sorted_dims) - 20} more dimensions")

    # File sizes
    print(f"\nOutput files:")
    for path in [dimension_index_path, temporal_index_path, spatial_index_path]:
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"  {path.name}: {size_mb:.2f} MB")

    return 0


if __name__ == "__main__":
    sys.exit(main())
