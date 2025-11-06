"""
Test script for multi-dimensional health surveillance extraction.

Demonstrates STIndex's new dimensional extraction framework with
health-specific dimensions (event_type, venue_type, disease, etc.).
"""

import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from loguru import logger

from stindex import DimensionalExtractor


def test_health_extraction():
    """Test dimensional extraction with health surveillance config."""

    logger.info("=" * 80)
    logger.info("Testing Multi-Dimensional Health Surveillance Extraction")
    logger.info("=" * 80)

    # Test cases with document metadata
    test_cases = [
        {
            "name": "WA Health Measles Alert (Australia)",
            "text": """
            Measles exposure sites in Western Australia:

            1. Margaret River Emergency Department
               Monday 27 October 2025 from 11:00 am to 7:00 pm

            2. Broome Regional Hospital
               Tuesday 28 October 2025 from 2:00 pm to 6:00 pm

            Anyone who attended these locations during these times may have been exposed to measles.
            If you develop symptoms, please contact your GP or hospital immediately.
            """,
            "metadata": {
                "publication_date": "2025-10-25",
                "source_location": "Western Australia",
                "source_url": "https://www.health.wa.gov.au/news/2025/measles-alert"
            }
        },
        {
            "name": "WA DOH Measles Cases (USA)",
            "text": """
            Washington State Department of Health has confirmed 3 measles cases in King County
            and 2 cases in Spokane County as of October 2025. All cases were unvaccinated.

            Exposure locations include:
            - Seattle Children's Hospital Emergency Department
            - Spokane Regional Health District Clinic

            Public health officials are monitoring the situation closely.
            """,
            "metadata": {
                "publication_date": "2025-10-20",
                "source_location": "Washington State, USA",
                "source_url": "https://doh.wa.gov/measles-cases-2025"
            }
        },
        {
            "name": "Influenza Surveillance Update",
            "text": """
            Influenza notifications in Western Australia:

            As of 27 October 2025, there have been 31,518 laboratory-confirmed influenza cases
            reported in WA this year. The majority of cases are Influenza A.

            Vaccination clinics are available at:
            - Perth CBD Medical Centre (weekdays 9am-5pm)
            - Fremantle Community Health Centre (Mon-Fri 8:30am-4pm)
            """,
            "metadata": {
                "publication_date": "2025-10-27",
                "source_location": "Western Australia",
                "source_url": "https://www.health.wa.gov.au/influenza-stats"
            }
        }
    ]

    # Initialize extractor with health dimensions config
    logger.info("\nInitializing DimensionalExtractor with health surveillance config...")
    try:
        extractor = DimensionalExtractor(
            config_path="extract",  # Main LLM config
            dimension_config_path="case_studies/public_health/extraction/config/health_dimensions"
        )
    except Exception as e:
        logger.error(f"Failed to initialize extractor: {e}")
        logger.info("Make sure vLLM server is running: ./scripts/start_server.sh")
        return

    # Run extraction on each test case
    for i, test_case in enumerate(test_cases, 1):
        logger.info("\n" + "=" * 80)
        logger.info(f"Test Case {i}/{len(test_cases)}: {test_case['name']}")
        logger.info("=" * 80)

        # Extract
        result = extractor.extract(
            text=test_case["text"],
            document_metadata=test_case["metadata"]
        )

        # Display results
        if result.success:
            logger.info(f"✓ Extraction successful ({result.processing_time:.2f}s)")

            # Show extracted dimensions
            for dim_name, entities in result.entities.items():
                if entities:
                    logger.info(f"\n{dim_name.upper()} ({len(entities)} entities):")
                    for j, entity in enumerate(entities, 1):
                        logger.info(f"  {j}. {json.dumps(entity, indent=6)}")

            # Save detailed results
            output_dir = Path("case_studies/public_health/data/results")
            output_dir.mkdir(parents=True, exist_ok=True)

            output_file = output_dir / f"test_{i}_{test_case['name'].lower().replace(' ', '_')}.json"
            with open(output_file, 'w') as f:
                json.dump(result.model_dump(), f, indent=2)

            logger.info(f"\n✓ Detailed results saved to: {output_file}")

        else:
            logger.error(f"✗ Extraction failed: {result.error}")

            # Show raw LLM output for debugging
            if result.extraction_config and result.extraction_config.get("raw_llm_output"):
                logger.debug(f"Raw LLM output: {result.extraction_config['raw_llm_output'][:500]}...")

    logger.info("\n" + "=" * 80)
    logger.info("Testing complete!")
    logger.info("=" * 80)


def test_dimension_disambiguation():
    """Test WA disambiguation (Australia vs USA)."""

    logger.info("\n" + "=" * 80)
    logger.info("Testing WA Disambiguation (Australia vs USA)")
    logger.info("=" * 80)

    extractor = DimensionalExtractor(
        config_path="extract",
        dimension_config_path="case_studies/public_health/extraction/config/health_dimensions"
    )

    # Extract from both documents
    wa_au_text = "Measles exposure site in Broome, Western Australia near the Pilbara region."
    wa_us_text = "Measles cases confirmed in Spokane County, Washington State near King County."

    result_au = extractor.extract(
        wa_au_text,
        document_metadata={"source_location": "Western Australia"}
    )

    result_us = extractor.extract(
        wa_us_text,
        document_metadata={"source_location": "Washington State, USA"}
    )

    # Compare spatial extractions
    logger.info("\nAustralia document (Broome):")
    for entity in result_au.entities.get("spatial", []):
        logger.info(f"  - {entity['text']}: ({entity.get('latitude')}, {entity.get('longitude')})")

    logger.info("\nUSA document (Spokane):")
    for entity in result_us.entities.get("spatial", []):
        logger.info(f"  - {entity['text']}: ({entity.get('latitude')}, {entity.get('longitude')})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test multi-dimensional health surveillance extraction"
    )
    parser.add_argument(
        "--test",
        type=str,
        default="all",
        choices=["all", "extraction", "disambiguation"],
        help="Which test to run"
    )

    args = parser.parse_args()

    try:
        if args.test in ["all", "extraction"]:
            test_health_extraction()

        if args.test in ["all", "disambiguation"]:
            test_dimension_disambiguation()

    except KeyboardInterrupt:
        logger.info("\nTest interrupted by user")
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
