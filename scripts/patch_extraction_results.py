#!/usr/bin/env python3
"""
Patch extraction_results.json to add missing 'source' and 'text' fields.

This script reads extraction_results.json, looks up source metadata from
the chunks file, and adds the missing fields without re-running the pipeline.

Usage:
    python scripts/patch_extraction_results.py <extraction_results.json> <chunks.json>

Example:
    python scripts/patch_extraction_results.py \
        case_studies/public_health/data/results/extraction_results.json \
        case_studies/public_health/data/chunks/preprocessed_chunks.json
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any

from loguru import logger


def load_chunks_metadata(chunks_file: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load chunks and build a mapping from chunk_id to metadata.

    Returns:
        Dict mapping chunk_id to {source, text, document_metadata}
    """
    logger.info(f"Loading chunks from: {chunks_file}")

    with open(chunks_file, 'r') as f:
        chunks = json.load(f)

    chunk_metadata = {}
    for chunk in chunks:
        chunk_id = chunk.get('chunk_id')
        if not chunk_id:
            continue

        chunk_metadata[chunk_id] = {
            'source': chunk.get('document_metadata', {}).get('source'),
            'text': chunk.get('text', ''),
            'document_metadata': chunk.get('document_metadata', {})
        }

    logger.info(f"✓ Loaded metadata for {len(chunk_metadata)} chunks")
    return chunk_metadata


def patch_extraction_results(
    results_file: Path,
    chunks_file: Path,
    output_file: Path = None
) -> None:
    """
    Patch extraction results with source and text fields.

    Args:
        results_file: Path to extraction_results.json
        chunks_file: Path to preprocessed_chunks.json
        output_file: Output path (default: overwrite input)
    """
    # Load chunks metadata
    chunk_metadata = load_chunks_metadata(chunks_file)

    # Load extraction results
    logger.info(f"Loading extraction results from: {results_file}")
    with open(results_file, 'r') as f:
        results = json.load(f)

    logger.info(f"✓ Loaded {len(results)} extraction results")

    # Patch results
    patched_count = 0
    missing_chunks = []

    for i, result in enumerate(results):
        chunk_id = result.get('chunk_id')

        if not chunk_id:
            logger.warning(f"Result {i} has no chunk_id, skipping")
            continue

        if chunk_id not in chunk_metadata:
            missing_chunks.append(chunk_id)
            continue

        metadata = chunk_metadata[chunk_id]

        # Add source if missing
        if 'source' not in result or result['source'] is None:
            result['source'] = metadata['source']
            patched_count += 1

        # Add text if missing
        if 'text' not in result or not result['text']:
            result['text'] = metadata['text']

    if missing_chunks:
        logger.warning(f"Could not find metadata for {len(missing_chunks)} chunks:")
        for chunk_id in missing_chunks[:5]:
            logger.warning(f"  - {chunk_id}")
        if len(missing_chunks) > 5:
            logger.warning(f"  ... and {len(missing_chunks) - 5} more")

    # Save patched results
    output_path = output_file or results_file
    logger.info(f"Saving patched results to: {output_path}")

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.success(f"✓ Patched {patched_count} results")
    logger.success(f"✓ Saved to: {output_path}")


def patch_case_study(case_study_dir: Path) -> None:
    """
    Automatically patch all extraction_results.json in a case study directory.

    Looks for:
    - data/results/**/extraction_results.json
    - data/chunks/preprocessed_chunks.json
    """
    case_study_dir = Path(case_study_dir)

    logger.info(f"Patching case study: {case_study_dir}")

    # Find chunks file
    chunks_file = case_study_dir / "data" / "chunks" / "preprocessed_chunks.json"
    if not chunks_file.exists():
        logger.error(f"Chunks file not found: {chunks_file}")
        return

    # Find all extraction_results.json files
    results_dir = case_study_dir / "data" / "results"
    if not results_dir.exists():
        logger.error(f"Results directory not found: {results_dir}")
        return

    results_files = list(results_dir.rglob("extraction_results.json"))

    if not results_files:
        logger.error(f"No extraction_results.json files found in {results_dir}")
        return

    logger.info(f"Found {len(results_files)} extraction_results.json files")

    for results_file in results_files:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Patching: {results_file.relative_to(case_study_dir)}")
        logger.info(f"{'=' * 60}")

        patch_extraction_results(
            results_file=results_file,
            chunks_file=chunks_file,
            output_file=results_file  # Overwrite
        )


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        logger.error("Usage: python patch_extraction_results.py <case_study_dir>")
        logger.error("   or: python patch_extraction_results.py <results.json> <chunks.json>")
        logger.error("")
        logger.error("Examples:")
        logger.error("  # Patch entire case study")
        logger.error("  python scripts/patch_extraction_results.py case_studies/public_health")
        logger.error("")
        logger.error("  # Patch specific files")
        logger.error("  python scripts/patch_extraction_results.py \\")
        logger.error("      case_studies/public_health/data/results/extraction_results.json \\")
        logger.error("      case_studies/public_health/data/chunks/preprocessed_chunks.json")
        sys.exit(1)

    if len(sys.argv) == 2:
        # Patch entire case study
        case_study_dir = Path(sys.argv[1])
        patch_case_study(case_study_dir)
    else:
        # Patch specific files
        results_file = Path(sys.argv[1])
        chunks_file = Path(sys.argv[2])
        output_file = Path(sys.argv[3]) if len(sys.argv) > 3 else None

        patch_extraction_results(
            results_file=results_file,
            chunks_file=chunks_file,
            output_file=output_file
        )


if __name__ == "__main__":
    main()
