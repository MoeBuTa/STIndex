"""
Test script for local Qwen3-8B model integration.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from stindex import STIndexExtractor
from stindex.models.schemas import ExtractionConfig


def test_local_model():
    """Test STIndex with local Qwen3-8B model."""
    print("=" * 80)
    print("STIndex - Local Qwen3-8B Model Test")
    print("=" * 80)

    # Sample text from PDF
    text = """On March 15, 2022, a strong cyclone hit the coastal areas near
    Broome, Western Australia and later moved inland towards Fitzroy Crossing
    by March 17."""

    print(f"\nInput text:\n{text}\n")

    # Configure for local model
    config = ExtractionConfig(
        llm_provider="local",
        model_name="Qwen/Qwen3-8B",
        device="auto",  # Will use CUDA if available
        temperature=0.0,
        enable_temporal=True,
        enable_spatial=True,
    )

    print("Initializing STIndex with local Qwen3-8B model...")
    print(f"Device: {config.device}")
    print(f"Model: {config.model_name}\n")

    try:
        # Initialize extractor
        extractor = STIndexExtractor(config=config)
        print("✓ Model loaded successfully\n")

        # Extract spatiotemporal indices
        print("Extracting spatiotemporal indices...")
        result = extractor.extract(text)

        # Display results
        print("\n" + "=" * 80)
        print("EXTRACTION RESULTS")
        print("=" * 80)

        print(f"\n✓ Processing time: {result.processing_time:.2f}s")
        print(f"✓ Temporal entities found: {result.temporal_count}")
        print(f"✓ Spatial entities found: {result.spatial_count}\n")

        if result.temporal_entities:
            print("-" * 80)
            print("TEMPORAL ENTITIES")
            print("-" * 80)
            for i, entity in enumerate(result.temporal_entities, 1):
                print(f"\n{i}. Text: '{entity.text}'")
                print(f"   Normalized: {entity.normalized}")
                print(f"   Type: {entity.temporal_type.value}")
                print(f"   Confidence: {entity.confidence:.2f}")
                if entity.start_char is not None:
                    print(f"   Position: {entity.start_char}-{entity.end_char}")

        if result.spatial_entities:
            print("\n" + "-" * 80)
            print("SPATIAL ENTITIES")
            print("-" * 80)
            for i, entity in enumerate(result.spatial_entities, 1):
                print(f"\n{i}. Text: '{entity.text}'")
                print(f"   Coordinates: ({entity.latitude:.4f}°, {entity.longitude:.4f}°)")
                print(f"   Type: {entity.location_type}")
                print(f"   Confidence: {entity.confidence:.2f}")
                if entity.address:
                    print(f"   Address: {entity.address}")
                if entity.country:
                    print(f"   Country: {entity.country}")

        print("\n" + "=" * 80)
        print("TEST COMPLETED SUCCESSFULLY")
        print("=" * 80)

        return True

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_temporal_only():
    """Test only temporal extraction."""
    print("\n\n" + "=" * 80)
    print("TEMPORAL-ONLY EXTRACTION TEST")
    print("=" * 80)

    text = """The project started on June 1st, 2023 and is scheduled to complete
    by December 31st, 2024. Weekly meetings are held every Monday at 2 PM."""

    print(f"\nInput: {text}\n")

    config = ExtractionConfig(
        llm_provider="local",
        model_name="Qwen/Qwen3-8B",
        enable_temporal=True,
        enable_spatial=False,  # Disable spatial extraction
    )

    try:
        extractor = STIndexExtractor(config=config)
        result = extractor.extract(text)

        print(f"Processing time: {result.processing_time:.2f}s\n")
        print("Temporal entities found:")
        for entity in result.temporal_entities:
            print(f"  - '{entity.text}' → {entity.normalized} ({entity.temporal_type.value})")

        return True

    except Exception as e:
        print(f"Error: {e}")
        return False


def test_batch_processing():
    """Test batch processing with local model."""
    print("\n\n" + "=" * 80)
    print("BATCH PROCESSING TEST")
    print("=" * 80)

    texts = [
        "On April 15, 2023, the conference was held in San Francisco.",
        "The expedition reached Mount Everest base camp on May 20, 2024.",
        "Scientists in Geneva announced the discovery on December 1, 2023.",
    ]

    print(f"\nProcessing {len(texts)} texts...\n")

    config = ExtractionConfig(
        llm_provider="local",
        model_name="Qwen/Qwen3-8B",
    )

    try:
        extractor = STIndexExtractor(config=config)
        results = extractor.extract_batch(texts)

        for i, (text, result) in enumerate(zip(texts, results), 1):
            print(f"\nText {i}: {text[:50]}...")
            print(f"  Temporal: {result.temporal_count}, Spatial: {result.spatial_count}")
            print(f"  Time: {result.processing_time:.2f}s")

        total_time = sum(r.processing_time for r in results)
        print(f"\nTotal processing time: {total_time:.2f}s")
        print(f"Average per text: {total_time/len(texts):.2f}s")

        return True

    except Exception as e:
        print(f"Error: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "STIndex Local Model Test Suite" + " " * 28 + "║")
    print("╚" + "═" * 78 + "╝\n")

    results = []

    # Test 1: Basic extraction
    print("\n[Test 1/3] Basic spatiotemporal extraction")
    results.append(("Basic extraction", test_local_model()))

    # Test 2: Temporal only
    print("\n[Test 2/3] Temporal-only extraction")
    results.append(("Temporal only", test_temporal_only()))

    # Test 3: Batch processing
    print("\n[Test 3/3] Batch processing")
    results.append(("Batch processing", test_batch_processing()))

    # Summary
    print("\n\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")

    print(f"\nPassed: {passed}/{total}")

    if passed == total:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")

    print("=" * 80)


if __name__ == "__main__":
    main()
