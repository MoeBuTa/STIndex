#!/usr/bin/env python3
"""
Test script to verify chunk attribute merging in pipeline.

Usage:
    python scripts/test_chunk_merge_pipeline.py
"""

from pathlib import Path
from stindex.pipeline.pipeline import STIndexPipeline

def test_chunk_merge():
    """Test chunk attribute merging method in pipeline."""

    # Initialize pipeline with chunk merge enabled
    pipeline = STIndexPipeline(
        output_dir="case_studies/public_health/data",
        enable_chunk_merge=True,
        chunk_merge_dir="data/chunks/public_health"
    )

    # Define paths
    extraction_results_path = Path("case_studies/public_health/data/analysis/extraction_results.json")
    chunks_dir = Path("data/chunks/public_health")

    print("Testing chunk attribute merge...")
    print(f"  Extraction results: {extraction_results_path}")
    print(f"  Chunks directory: {chunks_dir}")
    print()

    # Run merge
    pipeline.merge_chunk_attributes(
        extraction_results_path=extraction_results_path,
        chunks_dir=chunks_dir
    )

    print("\n✓ Test complete!")


if __name__ == "__main__":
    test_chunk_merge()
