#!/usr/bin/env python3
"""
Western Australia News Case Study

Demonstrates STIndex's end-to-end pipeline with real WA news articles and government documents.

Sources:
- UWA news articles (web)
- WA Government PDF reports
"""
import sys
from pathlib import Path

# Add STIndex to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from stindex import InputDocument, STIndexPipeline
from loguru import logger


def create_wa_news_sources():
    """Create input documents from Western Australia news sources."""

    sources = [
        # UWA News Article 1: Space Research
        InputDocument.from_url(
            url="https://www.uwa.edu.au/news/article/2025/september/poland-and-australia-partner-to-track-space-junk",
            metadata={
                "source": "UWA News",
                "category": "Research",
                "topic": "Space Science",
                "year": 2025
            },
            document_id="uwa_space_tracking_2025",
            title="Poland and Australia Partner to Track Space Junk"
        ),

        # UWA News Article 2: Leadership
        InputDocument.from_url(
            url="https://www.uwa.edu.au/news/article/2025/february/uwa-welcomes-first-female-chancellor",
            metadata={
                "source": "UWA News",
                "category": "Leadership",
                "topic": "Governance",
                "year": 2025
            },
            document_id="uwa_chancellor_2025",
            title="UWA Welcomes First Female Chancellor"
        ),

        # WA Government PDF: Financial Report
        InputDocument.from_url(
            url="https://www.wa.gov.au/system/files/2024-12/2024-25-myr.pdf",
            metadata={
                "source": "WA Government",
                "category": "Finance",
                "topic": "Economic Outlook",
                "year": 2024,
                "document_type": "PDF"
            },
            document_id="wa_myr_2024",
            title="WA Mid-Year Financial Projections 2024-25"
        )
    ]

    return sources


def run_full_pipeline():
    """Run complete pipeline: preprocessing → extraction → visualization."""

    logger.info("=" * 80)
    logger.info("Western Australia News Case Study")
    logger.info("=" * 80)

    # Create input sources
    logger.info("\nCreating input sources...")
    sources = create_wa_news_sources()

    logger.info(f"✓ Created {len(sources)} input documents:")
    for doc in sources:
        logger.info(f"  - {doc.title} ({doc.input_type.value})")

    # Initialize pipeline
    case_study_dir = Path(__file__).parent.parent
    config_path = case_study_dir / "config" / "wa_dimensions.yml"
    output_dir = case_study_dir / "data"

    logger.info(f"\nConfiguration:")
    logger.info(f"  - Dimension config: {config_path}")
    logger.info(f"  - Output directory: {output_dir}")

    pipeline = STIndexPipeline(
        dimension_config=str(config_path),
        output_dir=str(output_dir),
        max_chunk_size=2000,
        chunk_overlap=200,
        chunking_strategy="sliding_window",
        save_intermediate=True
    )

    # Run full pipeline
    logger.info("\n" + "=" * 80)
    logger.info("Running Full Pipeline")
    logger.info("=" * 80)

    results = pipeline.run_pipeline(
        input_docs=sources,
        save_results=True,
        visualize=True
    )

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("Case Study Complete!")
    logger.info("=" * 80)

    logger.info(f"\n📊 Results Summary:")
    logger.info(f"  - Input documents: {len(sources)}")
    logger.info(f"  - Extraction results: {len(results)}")

    successful = sum(1 for r in results if r.get('extraction', {}).get('success'))
    logger.info(f"  - Successful extractions: {successful}/{len(results)}")

    # Count entities by dimension
    dimension_counts = {}
    for result in results:
        if result.get('extraction', {}).get('success'):
            entities = result['extraction'].get('entities', {})
            for dim_name, dim_entities in entities.items():
                if dim_entities:
                    dimension_counts[dim_name] = dimension_counts.get(dim_name, 0) + len(dim_entities)

    if dimension_counts:
        logger.info(f"\n  📍 Entities Extracted:")
        for dim, count in sorted(dimension_counts.items()):
            logger.info(f"    - {dim}: {count} entities")

    logger.info(f"\n📁 Output Files:")
    logger.info(f"  - Chunks: {output_dir / 'chunks' / 'preprocessed_chunks.json'}")
    logger.info(f"  - Results: {output_dir / 'results' / 'extraction_results.json'}")
    logger.info(f"  - Visualizations: {output_dir / 'visualizations' / '*.html'}")

    return results


if __name__ == "__main__":
    try:
        results = run_full_pipeline()
        logger.success("\n✓ Pipeline completed successfully!")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\n✗ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
