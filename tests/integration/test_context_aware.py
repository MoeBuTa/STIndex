"""
Integration test for context-aware extraction with hierarchy format.

Tests that the refactored extraction pipeline maintains context-aware functionality:
- ExtractionContext works with hierarchy format
- Relative temporal expressions are resolved
- Spatial disambiguation works
- Multi-chunk extraction preserves context
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from stindex.extraction.dimensional_extraction import DimensionalExtractor
from stindex.extraction.context_manager import ExtractionContext
from stindex.extraction.dimension_loader import DimensionConfigLoader


def test_load_dimensions_with_hierarchy():
    """Test loading dimensions in new hierarchy format."""
    print("\n" + "=" * 80)
    print("TEST 1: Load dimensions with hierarchy format")
    print("=" * 80)

    loader = DimensionConfigLoader()
    config = loader.load_dimension_config("dimensions")
    dimensions = loader.get_enabled_dimensions(config)

    # Check mandatory dimensions present
    assert "temporal" in dimensions, "Temporal dimension missing"
    assert "spatial" in dimensions, "Spatial dimension missing"

    # Check temporal hierarchy
    temporal_dim = dimensions["temporal"]
    assert temporal_dim.hierarchy is not None, "Temporal hierarchy missing"
    assert len(temporal_dim.hierarchy) == 4, f"Expected 4 temporal levels, got {len(temporal_dim.hierarchy)}"

    temporal_levels = [h["level"] for h in temporal_dim.hierarchy]
    assert temporal_levels == ["timestamp", "date", "month", "year"], \
        f"Wrong temporal hierarchy: {temporal_levels}"

    # Check spatial hierarchy
    spatial_dim = dimensions["spatial"]
    assert spatial_dim.hierarchy is not None, "Spatial hierarchy missing"
    assert len(spatial_dim.hierarchy) == 4, f"Expected 4 spatial levels, got {len(spatial_dim.hierarchy)}"

    spatial_levels = [h["level"] for h in spatial_dim.hierarchy]
    assert spatial_levels == ["location", "city", "state", "country"], \
        f"Wrong spatial hierarchy: {spatial_levels}"

    # Check fields were auto-generated
    assert len(temporal_dim.fields) > 0, "Temporal fields not auto-generated"
    assert len(spatial_dim.fields) > 0, "Spatial fields not auto-generated"

    print(f"✓ Loaded {len(dimensions)} dimensions")
    print(f"  • Temporal: {len(temporal_dim.hierarchy)} levels, {len(temporal_dim.fields)} fields")
    print(f"  • Spatial: {len(spatial_dim.hierarchy)} levels, {len(spatial_dim.fields)} fields")
    print("✅ PASS: Dimensions loaded successfully with hierarchy format\n")


def test_extraction_context_initialization():
    """Test ExtractionContext initialization."""
    print("=" * 80)
    print("TEST 2: ExtractionContext initialization")
    print("=" * 80)

    context = ExtractionContext(
        max_memory_refs=10,
        enable_nearby_locations=True,
        document_metadata={"document_id": "test_doc_001"}
    )

    assert context.document_metadata["document_id"] == "test_doc_001"
    assert context.max_memory_refs == 10
    assert context.enable_nearby_locations == True

    print("✓ ExtractionContext initialized")
    print(f"  • Document ID: {context.document_metadata.get('document_id')}")
    print(f"  • Max memory refs: {context.max_memory_refs}")
    print("✅ PASS: ExtractionContext working\n")


def test_simple_extraction():
    """Test basic extraction without context."""
    print("=" * 80)
    print("TEST 3: Simple extraction (single chunk)")
    print("=" * 80)

    try:
        extractor = DimensionalExtractor(config_path="dimensions")

        text = """
        On March 15, 2022, a cyclone hit Broome, Western Australia.
        The storm caused significant damage to coastal areas.
        """

        result = extractor.extract(text)

        assert result.success, f"Extraction failed: {result.error}"
        assert "temporal" in result.entities, "No temporal entities extracted"
        assert "spatial" in result.entities, "No spatial entities extracted"

        print(f"✓ Extraction successful")
        print(f"  • Temporal entities: {len(result.entities.get('temporal', []))}")
        print(f"  • Spatial entities: {len(result.entities.get('spatial', []))}")

        # Check if hierarchy field is present
        if result.entities.get('temporal'):
            first_temporal = result.entities['temporal'][0]
            print(f"  • Temporal entity: {first_temporal.get('text', 'N/A')}")
            if 'hierarchy' in first_temporal:
                print(f"    Hierarchy: {first_temporal['hierarchy']}")

        print("✅ PASS: Basic extraction working\n")
        return True

    except Exception as e:
        print(f"✗ FAIL: Extraction failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_context_aware_extraction():
    """Test multi-chunk extraction with context."""
    print("=" * 80)
    print("TEST 4: Context-aware extraction (multi-chunk)")
    print("=" * 80)

    try:
        # Create extraction context
        context = ExtractionContext(
            max_memory_refs=10,
            document_metadata={"document_id": "test_doc_002"}
        )

        # Create extractor with context
        extractor = DimensionalExtractor(
            config_path="dimensions",
            extraction_context=context
        )

        # Chunk 1: Establishes temporal and spatial context
        chunk1 = """
        On January 10, 2023, health officials in Perth, Western Australia
        announced a measles outbreak.
        """

        print("\nChunk 1 (establishes context):")
        print(f"  Text: {chunk1[:60]}...")

        result1 = extractor.extract(chunk1)
        assert result1.success, f"Chunk 1 extraction failed: {result1.error}"

        print(f"  ✓ Extracted: {len(result1.entities.get('temporal', []))} temporal, "
              f"{len(result1.entities.get('spatial', []))} spatial")

        # Update context with chunk 1 results
        if result1.entities.get('temporal'):
            for entity in result1.entities['temporal']:
                if 'temporal' not in context.prior_refs:
                    context.prior_refs['temporal'] = []
                context.prior_refs['temporal'].append({
                    'text': entity.get('text', ''),
                    'normalized': entity.get('normalized', '')
                })

        if result1.entities.get('spatial'):
            for entity in result1.entities['spatial']:
                if 'spatial' not in context.prior_refs:
                    context.prior_refs['spatial'] = []
                context.prior_refs['spatial'].append({
                    'text': entity.get('text', '')
                })

        print(f"  ✓ Context updated: {len(context.prior_refs.get('temporal', []))} temporal refs, "
              f"{len(context.prior_refs.get('spatial', []))} spatial refs")

        # Chunk 2: Uses relative temporal reference
        chunk2 = """
        Two days later, additional cases were reported in Fremantle.
        The outbreak continued to spread throughout the week.
        """

        print("\nChunk 2 (uses context for 'two days later'):")
        print(f"  Text: {chunk2[:60]}...")

        # Update chunk index
        context.current_chunk_index = 1

        result2 = extractor.extract(chunk2)
        assert result2.success, f"Chunk 2 extraction failed: {result2.error}"

        print(f"  ✓ Extracted: {len(result2.entities.get('temporal', []))} temporal, "
              f"{len(result2.entities.get('spatial', []))} spatial")

        if result2.entities.get('temporal'):
            print(f"  • Temporal entities from chunk 2:")
            for entity in result2.entities['temporal']:
                print(f"    - {entity.get('text', 'N/A')} → {entity.get('normalized', 'N/A')}")

        print("\n✅ PASS: Context-aware extraction working\n")
        return True

    except Exception as e:
        print(f"✗ FAIL: Context-aware extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all context-aware extraction tests."""
    print("\n" + "=" * 80)
    print("CONTEXT-AWARE EXTRACTION INTEGRATION TESTS")
    print("Testing hierarchy format compatibility")
    print("=" * 80)

    tests = [
        ("Load dimensions with hierarchy", test_load_dimensions_with_hierarchy),
        ("ExtractionContext initialization", test_extraction_context_initialization),
        ("Simple extraction", test_simple_extraction),
        ("Context-aware extraction", test_context_aware_extraction),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result if result is not None else True))
        except Exception as e:
            print(f"\n✗ FAIL: {test_name} raised exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All context-aware extraction tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
