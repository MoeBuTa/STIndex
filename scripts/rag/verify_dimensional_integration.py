#!/usr/bin/env python3
"""
Verify dimensional integration with RAG retrieval.

Tests:
1. Load chunks_metadata.jsonl and verify format
2. Load dimension_index.json and verify structure
3. Initialize RRFRetriever with dimensions
4. Test retrieval with/without dimensional filtering
5. Verify hierarchical filtering works correctly

Usage:
    python -m scripts.rag.verify_dimensional_integration \
        --chunks-metadata data/indices/medcorp/chunks_metadata.jsonl \
        --indexes-dir data/indices/medcorp/indexes \
        --corpus data/original/medcorp/train.jsonl
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def verify_chunks_metadata(metadata_path: Path) -> bool:
    """Verify chunks_metadata.jsonl format."""
    print(f"\n{'=' * 60}")
    print("1. Verifying chunks_metadata.jsonl")
    print(f"{'=' * 60}")

    if not metadata_path.exists():
        print(f"  ✗ File not found: {metadata_path}")
        return False

    total = 0
    with_dimensions = 0
    sample_doc = None

    with open(metadata_path) as f:
        for i, line in enumerate(f):
            doc = json.loads(line)
            total += 1

            # Verify required fields
            if i == 0:
                required_fields = ["doc_id", "dimensions"]
                for field in required_fields:
                    if field not in doc:
                        print(f"  ✗ Missing required field: {field}")
                        return False
                sample_doc = doc

            if doc.get("dimensions"):
                with_dimensions += 1

    print(f"  ✓ Total documents: {total:,}")
    print(f"  ✓ Documents with dimensions: {with_dimensions:,} ({100*with_dimensions/total:.1f}%)")
    print(f"  ✓ Sample fields: {list(sample_doc.keys())}")
    print(f"  ✓ Sample dimensions: {list(sample_doc.get('dimensions', {}).keys())[:5]}...")

    return True


def verify_dimension_index(indexes_dir: Path) -> bool:
    """Verify dimension_index.json structure."""
    print(f"\n{'=' * 60}")
    print("2. Verifying dimension_index.json")
    print(f"{'=' * 60}")

    index_path = indexes_dir / "dimension_index.json"
    if not index_path.exists():
        print(f"  ✗ File not found: {index_path}")
        return False

    with open(index_path) as f:
        index = json.load(f)

    print(f"  ✓ Total dimensions: {len(index)}")

    # Check structure
    sample_dim = list(index.keys())[0]
    sample_data = index[sample_dim]

    if "labels" not in sample_data or "paths" not in sample_data:
        print(f"  ✗ Missing 'labels' or 'paths' in dimension index")
        return False

    print(f"  ✓ Index structure: labels + paths (hierarchical)")

    # Show top dimensions
    sorted_dims = sorted(
        index.items(),
        key=lambda x: len(x[1].get("labels", {})),
        reverse=True
    )

    print(f"\n  Top 10 dimensions by label count:")
    for dim_name, dim_data in sorted_dims[:10]:
        num_labels = len(dim_data.get("labels", {}))
        num_paths = len(dim_data.get("paths", {}))
        print(f"    • {dim_name}: {num_labels:,} labels, {num_paths:,} paths")

    return True


def verify_retriever_integration(
    corpus_path: Path,
    indices_dir: Path
) -> bool:
    """Verify RRFRetriever works with dimensional filtering."""
    print(f"\n{'=' * 60}")
    print("3. Verifying RRFRetriever integration")
    print(f"{'=' * 60}")

    try:
        from rag.retriever.rrf_retriever import RRFRetriever

        print("  Initializing RRFRetriever with dimensions...")
        retriever = RRFRetriever(
            corpus_path=str(corpus_path),
            indices_dir=str(indices_dir),
            use_bm25=True,
            use_contriever=False,  # Skip for faster testing
            use_specter=False,
            use_medcpt=False,
            use_dimensions=True
        )

        print(f"  ✓ Retriever initialized")
        print(f"  ✓ Dimension index loaded: {len(retriever.dimension_index or {})} dimensions")

        # Test basic retrieval
        print("\n  Testing basic retrieval (no filters)...")
        results = retriever.retrieve("What causes diabetes?", k=5)
        print(f"  ✓ Retrieved {len(results)} results")
        if results:
            print(f"    Top result: {results[0].title[:50]}...")

        # Test dimensional filtering
        print("\n  Testing dimensional filtering...")
        results_filtered = retriever.retrieve(
            "What causes diabetes?",
            k=5,
            dimension_filters=[{"dimension": "drug", "values": ["insulin"]}]
        )
        print(f"  ✓ Retrieved {len(results_filtered)} filtered results")
        if results_filtered:
            print(f"    Top result: {results_filtered[0].title[:50]}...")

        # Test hierarchical filtering
        print("\n  Testing hierarchical filtering...")

        # Get a sample label from the index
        if retriever.dimension_index:
            sample_dim = "symptom"
            if sample_dim in retriever.dimension_index:
                sample_labels = list(retriever.dimension_index[sample_dim]["labels"].keys())[:3]
                print(f"    Using dimension '{sample_dim}' with values: {sample_labels}")

                results_hier = retriever.retrieve(
                    "What are the symptoms?",
                    k=5,
                    dimension_filters=[{"dimension": sample_dim, "values": sample_labels}],
                    hierarchical=True
                )
                print(f"  ✓ Hierarchical filter matched {len(results_hier)} results")

        return True

    except Exception as e:
        print(f"  ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Verify dimensional integration with RAG retrieval"
    )
    parser.add_argument(
        "--chunks-metadata",
        default="data/indices/medcorp/chunks_metadata.jsonl",
        help="Path to chunks_metadata.jsonl"
    )
    parser.add_argument(
        "--indexes-dir",
        default="data/indices/medcorp/indexes",
        help="Directory containing dimension indexes"
    )
    parser.add_argument(
        "--corpus",
        default="data/original/medcorp/train.jsonl",
        help="Path to corpus JSONL"
    )
    parser.add_argument(
        "--skip-retriever",
        action="store_true",
        help="Skip retriever test (requires loading large models)"
    )

    args = parser.parse_args()

    metadata_path = Path(args.chunks_metadata)
    indexes_dir = Path(args.indexes_dir)
    corpus_path = Path(args.corpus)

    print("=" * 60)
    print("Dimensional Integration Verification")
    print("=" * 60)

    all_passed = True

    # Test 1: Verify metadata
    if not verify_chunks_metadata(metadata_path):
        all_passed = False

    # Test 2: Verify index
    if not verify_dimension_index(indexes_dir):
        all_passed = False

    # Test 3: Verify retriever (optional)
    if not args.skip_retriever:
        if not verify_retriever_integration(corpus_path, indexes_dir):
            all_passed = False
    else:
        print("\n  ⏭ Skipping retriever test")

    # Summary
    print(f"\n{'=' * 60}")
    if all_passed:
        print("✅ All verification tests PASSED!")
    else:
        print("❌ Some verification tests FAILED!")
    print(f"{'=' * 60}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
