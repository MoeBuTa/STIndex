"""
Smoke test for case studies with refactored hierarchy format.

Validates that case study configs load correctly after migration.
Does not run full extraction (which requires web scraping).
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from stindex.extraction.dimension_loader import DimensionConfigLoader
from stindex.extraction.dimensional_extraction import DimensionalExtractor


def test_public_health_config():
    """Test public health case study config loads correctly."""
    print("\n" + "=" * 80)
    print("TEST 1: Public Health Case Study Config")
    print("=" * 80)

    try:
        config_path = "case_studies/public_health/config/health_dimensions.yml"

        # Load dimensions
        loader = DimensionConfigLoader()
        config = loader.load_dimension_config(config_path)
        dimensions = loader.get_enabled_dimensions(config)

        assert len(dimensions) > 0, "No dimensions loaded"

        print(f"✓ Loaded {len(dimensions)} dimensions:")
        for dim_name, dim_config in dimensions.items():
            hierarchy_levels = len(dim_config.hierarchy) if dim_config.hierarchy else 0
            fields_count = len(dim_config.fields) if dim_config.fields else 0
            print(f"  • {dim_name}: {hierarchy_levels} hierarchy levels, {fields_count} fields")

        # Check mandatory dimensions
        assert "temporal" in dimensions, "Missing temporal dimension"
        assert "spatial" in dimensions, "Missing spatial dimension"

        # Check temporal hierarchy
        temporal = dimensions["temporal"]
        assert temporal.hierarchy is not None, "Temporal missing hierarchy"
        temporal_levels = [h["level"] for h in temporal.hierarchy]
        assert len(temporal_levels) == 4, f"Expected 4 temporal levels, got {len(temporal_levels)}"

        # Check spatial hierarchy
        spatial = dimensions["spatial"]
        assert spatial.hierarchy is not None, "Spatial missing hierarchy"
        spatial_levels = [h["level"] for h in spatial.hierarchy]
        assert len(spatial_levels) == 4, f"Expected 4 spatial levels, got {len(spatial_levels)}"

        print("✓ Mandatory dimensions validated (temporal, spatial)")
        print("✅ PASS: Public health config working\n")
        return True

    except Exception as e:
        print(f"✗ FAIL: Config loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_wa_news_config():
    """Test WA News case study config loads correctly."""
    print("=" * 80)
    print("TEST 2: WA News Case Study Config")
    print("=" * 80)

    try:
        config_path = "case_studies/wa_news/config/wa_dimensions.yml"

        # Load dimensions
        loader = DimensionConfigLoader()
        config = loader.load_dimension_config(config_path)
        dimensions = loader.get_enabled_dimensions(config)

        assert len(dimensions) > 0, "No dimensions loaded"

        print(f"✓ Loaded {len(dimensions)} dimensions:")
        for dim_name, dim_config in dimensions.items():
            hierarchy_levels = len(dim_config.hierarchy) if dim_config.hierarchy else 0
            fields_count = len(dim_config.fields) if dim_config.fields else 0
            print(f"  • {dim_name}: {hierarchy_levels} hierarchy levels, {fields_count} fields")

        # Check mandatory dimensions
        assert "temporal" in dimensions, "Missing temporal dimension"
        assert "spatial" in dimensions, "Missing spatial dimension"

        print("✓ Mandatory dimensions present")
        print("✅ PASS: WA News config working\n")
        return True

    except Exception as e:
        print(f"✗ FAIL: Config loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_extractor_initialization():
    """Test DimensionalExtractor initializes with case study configs."""
    print("=" * 80)
    print("TEST 3: DimensionalExtractor Initialization")
    print("=" * 80)

    try:
        # Test public health
        print("\n  Testing public health extractor...")
        extractor_health = DimensionalExtractor(
            config_path="extract",
            dimension_config_path="case_studies/public_health/config/health_dimensions"
        )
        assert extractor_health is not None
        print("  ✓ Public health extractor initialized")

        # Test WA news
        print("  Testing WA news extractor...")
        extractor_news = DimensionalExtractor(
            config_path="extract",
            dimension_config_path="case_studies/wa_news/config/wa_dimensions"
        )
        assert extractor_news is not None
        print("  ✓ WA news extractor initialized")

        print("\n✅ PASS: Extractors initialize with case study configs\n")
        return True

    except Exception as e:
        print(f"\n✗ FAIL: Extractor initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all case study smoke tests."""
    print("\n" + "=" * 80)
    print("CASE STUDY SMOKE TESTS")
    print("Validating migrated configs load correctly")
    print("=" * 80)

    tests = [
        ("Public health config", test_public_health_config),
        ("WA news config", test_wa_news_config),
        ("Extractor initialization", test_extractor_initialization),
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
        print("\n🎉 All case study smoke tests passed!")
        print("\n📝 Note: Full case study pipelines not run (require web scraping).")
        print("   Configs validated successfully - case studies should work with refactored format.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
