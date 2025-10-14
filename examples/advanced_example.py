"""
Advanced example demonstrating batch processing and configuration.

NOTE: This example demonstrates using OpenAI API for comparison with local models.
To run this example:
1. Set environment variable: export OPENAI_API_KEY="your-key-here"
2. Or modify the config to use local model: llm_provider="local", model_name="Qwen/Qwen3-8B"

For local model examples without API requirements, see basic_example.py
"""

import json
from pathlib import Path

from stindex import STIndexExtractor
from stindex.models.schemas import ExtractionConfig


def example_custom_config():
    """Example with custom configuration using OpenAI API."""
    print("=" * 80)
    print("Example 1: Custom Configuration (OpenAI API)")
    print("=" * 80)

    # Create custom config for OpenAI API
    # NOTE: Requires OPENAI_API_KEY environment variable
    config = ExtractionConfig(
        llm_provider="openai",  # Explicitly use OpenAI
        model_name="gpt-4",
        temperature=0.0,
        geocoder="nominatim",
        user_agent="stindex_advanced_example",
        min_confidence=0.7,
    )

    extractor = STIndexExtractor(config=config)

    text = "In January 2024, researchers in Tokyo, Japan discovered evidence of ancient civilizations."

    result = extractor.extract(text)

    print(f"\nExtracted {result.temporal_count} temporal and {result.spatial_count} spatial entities")
    print(f"Processing time: {result.processing_time:.2f}s\n")


def example_temporal_only():
    """Example extracting only temporal entities."""
    print("=" * 80)
    print("Example 2: Temporal-Only Extraction")
    print("=" * 80)

    config = ExtractionConfig(enable_temporal=True, enable_spatial=False)
    extractor = STIndexExtractor(config=config)

    text = """The project started on June 1st, 2023 and is scheduled to complete
    by December 31st, 2024. Weekly meetings are held every Monday at 2 PM."""

    result = extractor.extract(text)

    print("\nTemporal Entities:")
    for entity in result.temporal_entities:
        print(f"  - {entity.text} → {entity.normalized} ({entity.temporal_type.value})")


def example_spatial_only():
    """Example extracting only spatial entities."""
    print("\n" + "=" * 80)
    print("Example 3: Spatial-Only Extraction")
    print("=" * 80)

    config = ExtractionConfig(enable_temporal=False, enable_spatial=True)
    extractor = STIndexExtractor(config=config)

    text = """The conference will be held in Paris, France, with satellite events
    in London, Berlin, and New York."""

    result = extractor.extract(text)

    print("\nSpatial Entities:")
    for entity in result.spatial_entities:
        print(f"  - {entity.text}: ({entity.latitude:.4f}°, {entity.longitude:.4f}°)")


def example_batch_processing():
    """Example of batch processing multiple texts."""
    print("\n" + "=" * 80)
    print("Example 4: Batch Processing")
    print("=" * 80)

    extractor = STIndexExtractor()

    texts = [
        "On April 15, 2023, the conference was held in San Francisco.",
        "The expedition reached Mount Everest base camp on May 20, 2024.",
        "Scientists in Geneva announced the discovery on December 1, 2023.",
    ]

    print("\nProcessing batch of 3 texts...")
    results = extractor.extract_batch(texts)

    for i, result in enumerate(results, 1):
        print(f"\nText {i}:")
        print(f"  Temporal: {result.temporal_count}, Spatial: {result.spatial_count}")


def example_reference_date():
    """Example using reference date for relative time resolution."""
    print("\n" + "=" * 80)
    print("Example 5: Reference Date for Relative Times")
    print("=" * 80)

    # Set reference date
    config = ExtractionConfig(reference_date="2024-06-15")
    extractor = STIndexExtractor(config=config)

    text = "The meeting was scheduled for last week, and the follow-up is tomorrow."

    result = extractor.extract(text)

    print("\nRelative times resolved against 2024-06-15:")
    for entity in result.temporal_entities:
        print(f"  - '{entity.text}' → {entity.normalized}")


def example_export_formats():
    """Example showing different export formats."""
    print("\n" + "=" * 80)
    print("Example 6: Export Formats")
    print("=" * 80)

    extractor = STIndexExtractor()
    text = "On July 4, 1776, the Declaration was signed in Philadelphia."

    result = extractor.extract(text)

    # Export as dictionary
    result_dict = result.to_dict()
    print("\nExported as dictionary:")
    print(f"  Keys: {list(result_dict.keys())}")

    # Export as JSON
    json_str = json.dumps(result.model_dump(), indent=2, ensure_ascii=False)
    print(f"\nJSON output length: {len(json_str)} characters")


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("STIndex - Advanced Examples")
    print("=" * 80 + "\n")

    examples = [
        example_custom_config,
        example_temporal_only,
        example_spatial_only,
        example_batch_processing,
        example_reference_date,
        example_export_formats,
    ]

    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"\nError in {example.__name__}: {e}")

    print("\n" + "=" * 80)
    print("All examples completed!")
    print("=" * 80)


if __name__ == "__main__":
    main()
