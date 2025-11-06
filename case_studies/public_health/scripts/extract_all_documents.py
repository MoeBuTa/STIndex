"""
Batch extraction script for all processed health alert documents.

Reads chunked documents from case_studies/public_health/data/processed/
and runs multi-dimensional extraction on each chunk.
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from loguru import logger
from tqdm import tqdm

from stindex import DimensionalExtractor


def load_chunked_documents(processed_dir: str = "case_studies/public_health/data/processed") -> List[Dict[str, Any]]:
    """
    Load all chunked documents from processed directory.

    Args:
        processed_dir: Directory with chunked JSON files

    Returns:
        List of document chunks with metadata
    """
    processed_path = Path(processed_dir)
    chunked_files = list(processed_path.glob("chunked_*.json"))

    all_chunks = []
    for chunked_file in chunked_files:
        logger.info(f"Loading {chunked_file.name}...")
        with open(chunked_file, 'r') as f:
            chunks = json.load(f)
            all_chunks.extend(chunks)

    logger.info(f"✓ Loaded {len(all_chunks)} document chunks from {len(chunked_files)} files")
    return all_chunks


def extract_from_all_documents(
    output_file: str = "case_studies/public_health/data/results/batch_extraction_results.json",
    dimension_config: str = "case_studies/public_health/extraction/config/health_dimensions",
    sample_limit: int = None,
    save_frequency: int = 5
):
    """
    Run extraction on all chunked documents.

    Args:
        output_file: Path to save extraction results
        dimension_config: Path to dimension configuration
        sample_limit: Limit number of documents to process (for testing)
        save_frequency: Save checkpoint every N chunks (default: 5)
    """
    logger.info("=" * 80)
    logger.info("Batch Extraction: Health Surveillance Documents")
    logger.info("=" * 80)

    # Load chunked documents
    chunks = load_chunked_documents()

    if sample_limit:
        chunks = chunks[:sample_limit]
        logger.info(f"Processing first {sample_limit} chunks (sample mode)")

    # Initialize extractor
    logger.info(f"\nInitializing DimensionalExtractor with config: {dimension_config}")
    extractor = DimensionalExtractor(
        config_path="extract",
        dimension_config_path=dimension_config
    )

    # Check for existing results (resume capability)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    processed_chunk_ids = set()

    if output_path.exists():
        logger.info(f"\n✓ Found existing results file: {output_path}")
        with open(output_path, 'r') as f:
            results = json.load(f)
        processed_chunk_ids = {r['chunk_id'] for r in results if 'chunk_id' in r}
        logger.info(f"  Loaded {len(results)} existing results")
        logger.info(f"  Resuming from chunk {len(results)}/{len(chunks)}")

    # Extract from each chunk
    logger.info(f"\nProcessing {len(chunks)} document chunks...")

    for i, chunk in enumerate(tqdm(chunks, desc="Extracting")):
        chunk_id = chunk.get('chunk_id')

        # Skip if already processed
        if chunk_id in processed_chunk_ids:
            continue

        try:
            # Prepare document metadata from chunk
            doc_metadata = chunk.get('document_metadata', {})

            # Build metadata dict for extraction
            extraction_metadata = {
                "publication_date": doc_metadata.get('publication_date'),
                "source_location": doc_metadata.get('region'),
                "source_url": doc_metadata.get('url'),
                "source": doc_metadata.get('source'),
            }

            # Extract dimensions
            result = extractor.extract(
                text=chunk['text'],
                document_metadata=extraction_metadata
            )

            # Store result with chunk info
            result_data = {
                "chunk_id": chunk_id,
                "chunk_index": chunk.get('chunk_index'),
                "document_title": chunk.get('document_title'),
                "source": doc_metadata.get('source'),
                "extraction": result.model_dump(),
            }

            results.append(result_data)
            processed_chunk_ids.add(chunk_id)

            # Save checkpoint every N chunks
            if len(results) % save_frequency == 0:
                with open(output_path, 'w') as f:
                    json.dump(results, f, indent=2)
                logger.info(f"💾 Checkpoint saved ({len(results)}/{len(chunks)} chunks)")

            # Log progress every 10 chunks
            if (i + 1) % 10 == 0:
                success_count = sum(1 for r in results if r.get('extraction', {}).get('success'))
                logger.info(f"Processed {i + 1}/{len(chunks)} chunks ({success_count} successful)")

        except Exception as e:
            logger.error(f"Failed to process chunk {i}: {e}")
            result_data = {
                "chunk_id": chunk_id,
                "chunk_index": chunk.get('chunk_index'),
                "document_title": chunk.get('document_title'),
                "error": str(e)
            }
            results.append(result_data)
            processed_chunk_ids.add(chunk_id)

            # Save checkpoint on error too
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)

    # Save final results
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("Extraction Summary")
    logger.info("=" * 80)

    success_count = sum(1 for r in results if r.get('extraction', {}).get('success'))
    error_count = len(results) - success_count

    logger.info(f"Total chunks processed: {len(results)}")
    logger.info(f"Successful extractions: {success_count}")
    logger.info(f"Failed extractions: {error_count}")

    # Dimension statistics
    dimension_counts = {}
    for r in results:
        if r.get('extraction', {}).get('success'):
            entities = r['extraction'].get('entities', {})
            for dim_name, dim_entities in entities.items():
                if dim_entities:
                    dimension_counts[dim_name] = dimension_counts.get(dim_name, 0) + len(dim_entities)

    if dimension_counts:
        logger.info("\nDimensions extracted:")
        for dim_name, count in sorted(dimension_counts.items()):
            logger.info(f"  - {dim_name}: {count} entities")

    logger.info(f"\n✓ Results saved to: {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Batch extraction for health surveillance documents"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="case_studies/public_health/data/results/batch_extraction_results.json",
        help="Output JSON file for results"
    )
    parser.add_argument(
        "--dimension-config",
        type=str,
        default="case_studies/public_health/extraction/config/health_dimensions",
        help="Path to dimension configuration"
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="Limit number of documents to process (for testing)"
    )
    parser.add_argument(
        "--save-frequency",
        type=int,
        default=5,
        help="Save checkpoint every N chunks (default: 5)"
    )

    args = parser.parse_args()

    try:
        extract_from_all_documents(
            output_file=args.output,
            dimension_config=args.dimension_config,
            sample_limit=args.sample_limit,
            save_frequency=args.save_frequency
        )
    except KeyboardInterrupt:
        logger.info("\n⚠️  Extraction interrupted by user")
        logger.info(f"✓ Progress has been saved to checkpoint file")
        logger.info(f"  Resume by running the same command again")
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        import traceback
        traceback.print_exc()
