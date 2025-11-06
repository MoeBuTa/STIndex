"""
End-to-end data collection and preprocessing pipeline.

Runs the complete pipeline:
1. Scrape health alerts from web sources
2. Parse HTML into structured documents
3. Chunk long documents for extraction
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from loguru import logger

from case_studies.public_health.preprocessing.scrapers import scrape_all_health_alerts
from case_studies.public_health.preprocessing.parsers import parse_all_health_alerts
from case_studies.public_health.preprocessing.chunkers import chunk_all_parsed_documents


def run_data_collection_pipeline(
    raw_dir: str = "case_studies/public_health/data/raw",
    processed_dir: str = "case_studies/public_health/data/processed",
    max_chunk_size: int = 2000,
    chunking_strategy: str = "paragraph"
):
    """
    Run the complete data collection and preprocessing pipeline.

    Args:
        raw_dir: Directory to save raw scraped data
        processed_dir: Directory to save processed data
        max_chunk_size: Maximum chunk size in characters
        chunking_strategy: Chunking strategy ("sliding_window", "paragraph", "semantic")
    """
    logger.info("=" * 80)
    logger.info("Public Health Surveillance - Data Collection Pipeline")
    logger.info("=" * 80)

    # Step 1: Scrape data
    logger.info("\n[1/3] Scraping health alerts from web sources...")
    try:
        scrape_all_health_alerts(output_dir=raw_dir)
    except Exception as e:
        logger.error(f"Scraping failed: {e}")
        logger.info("Continuing with existing data if available...")

    # Step 2: Parse documents
    logger.info("\n[2/3] Parsing scraped documents...")
    try:
        parse_all_health_alerts(raw_dir=raw_dir, output_dir=processed_dir)
    except Exception as e:
        logger.error(f"Parsing failed: {e}")
        raise

    # Step 3: Chunk documents
    logger.info(f"\n[3/3] Chunking documents (max_size={max_chunk_size}, strategy={chunking_strategy})...")
    try:
        chunk_all_parsed_documents(
            processed_dir=processed_dir,
            max_chunk_size=max_chunk_size,
            strategy=chunking_strategy
        )
    except Exception as e:
        logger.error(f"Chunking failed: {e}")
        raise

    logger.info("\n" + "=" * 80)
    logger.info("✓ Data collection pipeline complete!")
    logger.info("=" * 80)
    logger.info(f"\nRaw data: {raw_dir}")
    logger.info(f"Processed data: {processed_dir}")
    logger.info("\nNext steps:")
    logger.info("  1. Review processed documents in data/processed/")
    logger.info("  2. Run extraction: python case_studies/public_health/scripts/run_measles_demo.py")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Public health surveillance data collection pipeline"
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="case_studies/public_health/data/raw",
        help="Directory to save raw scraped data"
    )
    parser.add_argument(
        "--processed-dir",
        type=str,
        default="case_studies/public_health/data/processed",
        help="Directory to save processed data"
    )
    parser.add_argument(
        "--max-chunk-size",
        type=int,
        default=2000,
        help="Maximum chunk size in characters"
    )
    parser.add_argument(
        "--chunking-strategy",
        type=str,
        default="paragraph",
        choices=["sliding_window", "paragraph", "semantic"],
        help="Chunking strategy"
    )

    args = parser.parse_args()

    run_data_collection_pipeline(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        max_chunk_size=args.max_chunk_size,
        chunking_strategy=args.chunking_strategy
    )
