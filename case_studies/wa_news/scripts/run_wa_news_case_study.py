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
    """Create input documents from Western Australia and Australian news sources.

    Total: 10 valid documents (all >5000 chars parsed)
    - ABC News (3 documents)
    - WA News Sites (3 documents)
    - Universities (2 documents)
    - Australian Geographic (1 document)
    - Government (1 document)
    """

    sources = [
        # 1. ABC News - Western Australia
        InputDocument.from_url(
            url="https://www.abc.net.au/news/wa",
            metadata={
                "source": "ABC News",
                "category": "Regional News",
                "topic": "WA News",
                "state": "Western Australia",
                "year": 2025
            },
            document_id="abc_wa_news",
            title="Western Australia News - ABC"
        ),

        # 2. Curtin University News
        InputDocument.from_url(
            url="https://www.curtin.edu.au/news/",
            metadata={
                "source": "Curtin University",
                "category": "News",
                "topic": "University News",
                "institution": "Curtin University",
                "state": "Western Australia",
                "year": 2025
            },
            document_id="curtin_news",
            title="Curtin University News"
        ),

        # 3. Edith Cowan University News
        InputDocument.from_url(
            url="https://www.ecu.edu.au/news",
            metadata={
                "source": "ECU",
                "category": "News",
                "topic": "University News",
                "institution": "Edith Cowan University",
                "state": "Western Australia",
                "year": 2025
            },
            document_id="ecu_news",
            title="Edith Cowan University News"
        ),

        # 4. Perth Now - WA News
        InputDocument.from_url(
            url="https://www.perthnow.com.au/news/wa",
            metadata={
                "source": "PerthNow",
                "category": "News",
                "topic": "WA News",
                "state": "Western Australia",
                "year": 2025
            },
            document_id="perthnow_wa",
            title="Perth Now - WA News"
        ),

        # 5. The West - WA News
        InputDocument.from_url(
            url="https://www.thewest.com.au/news/wa",
            metadata={
                "source": "The West Australian",
                "category": "News",
                "topic": "WA News",
                "state": "Western Australia",
                "year": 2025
            },
            document_id="thewest_wa",
            title="The West - WA News"
        ),

        # 6. Australian Geographic
        InputDocument.from_url(
            url="https://www.australiangeographic.com.au/",
            metadata={
                "source": "Australian Geographic",
                "category": "Geography",
                "topic": "Australian Geography & Nature",
                "year": 2025
            },
            document_id="ausgeo_home",
            title="Australian Geographic"
        ),

        # 7. ABC Science News
        InputDocument.from_url(
            url="https://www.abc.net.au/news/science/",
            metadata={
                "source": "ABC News",
                "category": "Science",
                "topic": "Science News",
                "year": 2025
            },
            document_id="abc_science",
            title="ABC Science News"
        ),

        # 8. ABC Rural News
        InputDocument.from_url(
            url="https://www.abc.net.au/news/rural/",
            metadata={
                "source": "ABC News",
                "category": "Rural",
                "topic": "Rural & Agriculture News",
                "year": 2025
            },
            document_id="abc_rural",
            title="ABC Rural News"
        ),

        # 9. WAtoday Homepage
        InputDocument.from_url(
            url="https://www.watoday.com.au/",
            metadata={
                "source": "WAtoday",
                "category": "News",
                "topic": "WA News & Current Affairs",
                "state": "Western Australia",
                "year": 2025
            },
            document_id="watoday_home",
            title="WAtoday Homepage"
        ),

        # 10. PerthNow Homepage
        InputDocument.from_url(
            url="https://www.perthnow.com.au/",
            metadata={
                "source": "PerthNow",
                "category": "News",
                "topic": "Perth News & Current Affairs",
                "state": "Western Australia",
                "year": 2025
            },
            document_id="perthnow_home",
            title="PerthNow Homepage"
        ),
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
