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
    """Create input documents from health surveillance sources.

    Total: 10 valid documents (all >5000 chars parsed)
    - US sources: Washington State DOH, CDC (1 document)
    - International: WHO (7 documents)
    - Australian sources: Immunisation Coalition, Victoria Health (2 documents)
    """

    sources = [
        # 1. Washington State DOH - Measles Cases
        InputDocument.from_url(
            url="https://doh.wa.gov/you-and-your-family/illness-and-disease-z/measles/measles-cases-washington-state-2025",
            metadata={
                "source": "WA State DOH",
                "category": "Case Data",
                "topic": "Measles",
                "jurisdiction": "Washington",
                "country": "USA",
                "year": 2025
            },
            document_id="wa_doh_measles_2025",
            title="Measles Cases - Washington State 2025"
        ),

        # 2. Australian Influenza Statistics
        InputDocument.from_url(
            url="https://immunisationcoalition.org.au/influenza-statistics/",
            metadata={
                "source": "Immunisation Coalition",
                "category": "Statistics",
                "topic": "Influenza",
                "jurisdiction": "National",
                "country": "Australia",
                "year": 2025
            },
            document_id="australia_influenza_stats",
            title="Australian Influenza Statistics"
        ),

        # 3. CDC - Measles Cases and Outbreaks
        InputDocument.from_url(
            url="https://www.cdc.gov/measles/data-research/index.html",
            metadata={
                "source": "CDC",
                "category": "Case Data",
                "topic": "Measles",
                "jurisdiction": "National",
                "country": "USA",
                "year": 2025
            },
            document_id="cdc_measles_outbreaks",
            title="Measles Cases and Outbreaks - CDC"
        ),

        # 4. WHO - Measles Fact Sheet
        InputDocument.from_url(
            url="https://www.who.int/news-room/fact-sheets/detail/measles",
            metadata={
                "source": "WHO",
                "category": "Fact Sheet",
                "topic": "Measles",
                "jurisdiction": "Global",
                "country": "International",
                "year": 2025
            },
            document_id="who_measles_factsheet",
            title="Measles Fact Sheet - WHO"
        ),

        # 5. WHO - Influenza (Seasonal) Fact Sheet
        InputDocument.from_url(
            url="https://www.who.int/news-room/fact-sheets/detail/influenza-(seasonal)",
            metadata={
                "source": "WHO",
                "category": "Fact Sheet",
                "topic": "Influenza",
                "jurisdiction": "Global",
                "country": "International",
                "year": 2025
            },
            document_id="who_flu_factsheet",
            title="Influenza (Seasonal) Fact Sheet - WHO"
        ),

        # 6. WHO - Immunization Coverage
        InputDocument.from_url(
            url="https://www.who.int/news-room/fact-sheets/detail/immunization-coverage",
            metadata={
                "source": "WHO",
                "category": "Fact Sheet",
                "topic": "Immunization",
                "jurisdiction": "Global",
                "country": "International",
                "year": 2025
            },
            document_id="who_immunization_coverage",
            title="Immunization Coverage - WHO"
        ),

        # 7. WHO - Vaccines and Immunization
        InputDocument.from_url(
            url="https://www.who.int/health-topics/vaccines-and-immunization",
            metadata={
                "source": "WHO",
                "category": "Health Topics",
                "topic": "Vaccines",
                "jurisdiction": "Global",
                "country": "International",
                "year": 2025
            },
            document_id="who_vaccines_immunization",
            title="Vaccines and Immunization - WHO"
        ),

        # 8. Victoria Health - Measles
        InputDocument.from_url(
            url="https://www.health.vic.gov.au/infectious-diseases/measles",
            metadata={
                "source": "Victoria Health",
                "category": "Infectious Diseases",
                "topic": "Measles",
                "jurisdiction": "Victoria",
                "country": "Australia",
                "year": 2025
            },
            document_id="vic_health_measles",
            title="Measles - Victoria Health"
        ),

        # 9. WHO - Poliomyelitis
        InputDocument.from_url(
            url="https://www.who.int/news-room/fact-sheets/detail/poliomyelitis",
            metadata={
                "source": "WHO",
                "category": "Fact Sheet",
                "topic": "Polio",
                "jurisdiction": "Global",
                "country": "International",
                "year": 2025
            },
            document_id="who_polio",
            title="Poliomyelitis - WHO"
        ),

        # 10. WHO - Dengue and Severe Dengue
        InputDocument.from_url(
            url="https://www.who.int/news-room/fact-sheets/detail/dengue-and-severe-dengue",
            metadata={
                "source": "WHO",
                "category": "Fact Sheet",
                "topic": "Dengue",
                "jurisdiction": "Global",
                "country": "International",
                "year": 2025
            },
            document_id="who_dengue",
            title="Dengue and Severe Dengue - WHO"
        ),
    ]

    return sources


def run_full_pipeline():
    """Run complete pipeline: preprocessing → extraction → analysis → export."""

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
    analysis_dir = case_study_dir / "frontend_data"  # For static JSON export

    logger.info(f"\nConfiguration:")
    logger.info(f"  - Dimension config: {config_path}")
    logger.info(f"  - Output directory: {output_dir}")
    logger.info(f"  - Analysis data: {analysis_dir}")
    logger.info(f"  - Preprocessing: Loaded from cfg/preprocess/*.yml")

    pipeline = STIndexPipeline(
        dimension_config=str(config_path),
        output_dir=str(output_dir),
        save_intermediate=True
    )

    # Run full pipeline
    logger.info("\n" + "=" * 80)
    logger.info("Running Full Pipeline")
    logger.info("=" * 80)

    pipeline_output = pipeline.run_pipeline(
        input_docs=sources,
        save_results=True,
        analyze=True  # Enable analysis instead of visualization
    )

    # Extract results and analysis
    results = pipeline_output['results']
    analysis_data = pipeline_output['analysis']

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("Case Study Complete!")
    logger.info("=" * 80)

    logger.info(f"\n📊 Extraction Summary:")
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

    # Analysis summary
    if analysis_data:
        logger.info(f"\n  📈 Analysis Summary:")
        clusters = analysis_data.get('clusters', {})
        story_arcs = analysis_data.get('story_arcs', [])
        dim_analysis = analysis_data.get('dimension_analysis', {})
        exported_files = analysis_data.get('exported_files', {})

        logger.info(f"    - Clusters detected: {len(clusters.get('clusters', []))}")
        logger.info(f"    - Story arcs found: {len(story_arcs)}")
        logger.info(f"    - Dimensions analyzed: {len([k for k in dim_analysis.keys() if k not in ['global', 'cross_dimensional']])}")
        logger.info(f"    - Data files exported: {len(exported_files)}")

        logger.info(f"\n  📁 Exported Data Files:")
        for file_type, filepath in exported_files.items():
            logger.info(f"    - {file_type}: {Path(filepath).name}")

    logger.info(f"\n📁 Output Files:")
    logger.info(f"  - Chunks: {output_dir / 'chunks' / 'preprocessed_chunks.json'}")
    logger.info(f"  - Results: {output_dir / 'results' / 'extraction_results.json'}")
    if analysis_data:
        analysis_output_dir = Path(list(analysis_data.get('exported_files', {}).values())[0]).parent if analysis_data.get('exported_files') else analysis_dir
        logger.info(f"  - Analysis data: {analysis_output_dir}/*.json")

    return results, analysis_data


if __name__ == "__main__":
    try:
        results, analysis_data = run_full_pipeline()
        logger.success("\n✓ Pipeline completed successfully!")
        sys.exit(0)
    except Exception as e:
        logger.error(f"\n✗ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
