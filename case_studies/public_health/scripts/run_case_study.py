#!/usr/bin/env python3
"""
Public Health Surveillance Case Study

Demonstrates STIndex's end-to-end pipeline with health surveillance documents.

Sources:
- WA Health alerts
- Washington State DOH measles cases
- Australian Influenza statistics
"""
import sys
from pathlib import Path

# Add STIndex to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from stindex import InputDocument, STIndexPipeline
from loguru import logger


def create_health_sources():
    """Create input documents from health surveillance sources."""

    sources = [
        # WA Health Australia - Measles Alert
        InputDocument.from_url(
            url="https://www.health.wa.gov.au/news/2025/measles-alert",
            metadata={
                "source": "WA Health",
                "category": "Disease Alert",
                "topic": "Measles",
                "year": 2025
            },
            document_id="wa_health_measles_2025",
            title="Measles Alert - WA Health"
        ),

        # Washington State DOH - Measles Cases
        InputDocument.from_url(
            url="https://doh.wa.gov/you-and-your-family/illness-and-disease-z/measles/measles-cases-washington-state-2025",
            metadata={
                "source": "WA State DOH",
                "category": "Case Data",
                "topic": "Measles",
                "year": 2025
            },
            document_id="wa_doh_measles_2025",
            title="Measles Cases - Washington State 2025"
        ),

        # Australian Influenza Statistics
        InputDocument.from_url(
            url="https://immunisationcoalition.org.au/influenza-statistics/",
            metadata={
                "source": "Immunisation Coalition",
                "category": "Statistics",
                "topic": "Influenza",
                "year": 2025
            },
            document_id="australia_influenza_stats",
            title="Australian Influenza Statistics"
        ),
    ]

    return sources


def run_full_pipeline():
    """Run complete pipeline: preprocessing → extraction → visualization."""

    logger.info("=" * 80)
    logger.info("Public Health Surveillance Case Study")
    logger.info("=" * 80)

    # Create input sources
    logger.info("\nCreating input sources...")
    sources = create_health_sources()

    logger.info(f"✓ Created {len(sources)} input documents:")
    for doc in sources:
        logger.info(f"  - {doc.title} ({doc.input_type.value})")

    # Initialize pipeline
    case_study_dir = Path(__file__).parent.parent
    config_path = case_study_dir / "config" / "health_dimensions.yml"
    output_dir = case_study_dir / "data"

    logger.info(f"\nConfiguration:")
    logger.info(f"  - Dimension config: {config_path}")
    logger.info(f"  - Output directory: {output_dir}")
    logger.info(f"  - Preprocessing: Loaded from cfg/preprocess/*.yml")
    logger.info(f"  - Visualization: Loaded from cfg/visualization.yml")

    pipeline = STIndexPipeline(
        dimension_config=str(config_path),
        output_dir=str(output_dir),
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
