#!/usr/bin/env python3
"""
Simple integration test for new agentic architecture.

Tests the full pipeline: STIndexExtractor -> ExtractionPipeline -> SpatioTemporalExtractorAgent
"""

from stindex.extractors.extractor import STIndexExtractor


def test_basic_extraction():
    """Test basic extraction with new architecture."""
    print("=" * 80)
    print("Testing New Agentic Architecture")
    print("=" * 80)

    # Initialize extractor
    print("\n1. Initializing STIndexExtractor...")
    extractor = STIndexExtractor()
    print("✓ Extractor initialized")

    # Test text
    text = "On March 15, 2022, a strong cyclone hit the coastal areas near Broome, Western Australia."

    print(f"\n2. Extracting from text:")
    print(f"   '{text}'")

    # Extract
    result = extractor.extract(text)

    print(f"\n3. Results:")
    print(f"   Temporal entities: {len(result.temporal_entities)}")
    for entity in result.temporal_entities:
        print(f"      - {entity.text} → {entity.normalized} ({entity.temporal_type})")

    print(f"\n   Spatial entities: {len(result.spatial_entities)}")
    for entity in result.spatial_entities:
        print(f"      - {entity.text} → ({entity.latitude}, {entity.longitude})")

    print(f"\n   Processing time: {result.processing_time:.2f}s")

    # Validation
    assert len(result.temporal_entities) > 0, "Should extract temporal entities"
    assert len(result.spatial_entities) > 0, "Should extract spatial entities"

    # Check specific extractions
    temporal_texts = [e.text for e in result.temporal_entities]
    spatial_texts = [e.text for e in result.spatial_entities]

    print("\n4. Validation:")
    if "March 15, 2022" in temporal_texts or "March 15" in temporal_texts:
        print("   ✓ Temporal extraction correct")
    else:
        print(f"   ✗ Temporal extraction failed: {temporal_texts}")

    if any("Broome" in text for text in spatial_texts):
        print("   ✓ Spatial extraction correct")
    else:
        print(f"   ✗ Spatial extraction failed: {spatial_texts}")

    print("\n" + "=" * 80)
    print("Test completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    test_basic_extraction()
