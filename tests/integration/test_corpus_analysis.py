"""
Integration test for corpus analysis with hierarchy format.

Tests that analysis pipeline works with refactored extraction:
- DimensionAnalyzer processes extraction results
- Event clustering works
- Dimension statistics generation works
"""

import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from stindex.analysis.dimension_analyzer import DimensionAnalyzer


def test_dimension_analyzer_initialization():
    """Test DimensionAnalyzer initialization."""
    print("\n" + "=" * 80)
    print("TEST 1: DimensionAnalyzer initialization")
    print("=" * 80)

    analyzer = DimensionAnalyzer()

    assert analyzer is not None, "Analyzer initialization failed"
    print("✓ DimensionAnalyzer initialized")
    print("✅ PASS: DimensionAnalyzer working\n")
    return True


def test_analyze_sample_results():
    """Test analyzing sample extraction results."""
    print("=" * 80)
    print("TEST 2: Analyze sample extraction results")
    print("=" * 80)

    try:
        # Create sample extraction results with hierarchy format
        # Format matches STIndexPipeline output
        sample_results = [
            {
                "document_id": "doc_001",
                "text": "On March 15, 2022, a cyclone hit Broome, Western Australia.",
                "extraction": {
                    "success": True,
                    "entities": {
                        "temporal": [
                            {
                                "text": "March 15, 2022",
                                "normalized": "2022-03-15",
                                "timestamp": "2022-03-15T00:00:00",
                                "date": "2022-03-15",
                                "month": "2022-03",
                                "year": "2022"
                            }
                        ],
                        "spatial": [
                            {
                                "text": "Broome",
                                "latitude": -17.9614,
                                "longitude": 122.2359,
                                "location": "Broome",
                                "city": "Broome",
                                "state": "Western Australia",
                                "country": "Australia"
                            }
                        ]
                    }
                }
            },
            {
                "document_id": "doc_002",
                "text": "On March 20, 2022, recovery efforts began in Perth.",
                "extraction": {
                    "success": True,
                    "entities": {
                        "temporal": [
                            {
                                "text": "March 20, 2022",
                                "normalized": "2022-03-20",
                                "timestamp": "2022-03-20T00:00:00",
                                "date": "2022-03-20",
                                "month": "2022-03",
                                "year": "2022"
                            }
                        ],
                        "spatial": [
                            {
                                "text": "Perth",
                                "latitude": -31.9505,
                                "longitude": 115.8605,
                                "location": "Perth",
                                "city": "Perth",
                                "state": "Western Australia",
                                "country": "Australia"
                            }
                        ]
                    }
                }
            }
        ]

        analyzer = DimensionAnalyzer()

        # Analyze results
        analysis = analyzer.analyze(sample_results)

        assert analysis is not None, "Analysis returned None"
        assert len(analysis) > 0, "Analysis is empty"

        print("✓ Analysis completed successfully")
        print(f"  • Analysis dimensions: {[k for k in analysis.keys() if k not in ['cross_dimensional', 'global']]}")

        # Check for temporal analysis
        if "temporal" in analysis:
            print(f"  • Temporal entities analyzed: {analysis['temporal'].get('total_entities', 0)}")

        # Check for spatial analysis
        if "spatial" in analysis:
            print(f"  • Spatial entities analyzed: {analysis['spatial'].get('total_entities', 0)}")

        print("✅ PASS: Analysis working with hierarchy format\n")
        return True

    except Exception as e:
        print(f"✗ FAIL: Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dimension_stats():
    """Test dimension statistics generation."""
    print("=" * 80)
    print("TEST 3: Dimension statistics generation")
    print("=" * 80)

    try:
        # Sample results with multiple dimensions
        sample_results = [
            {
                "document_id": "doc_001",
                "extraction": {
                    "success": True,
                    "entities": {
                        "temporal": [
                            {
                                "text": "January 15, 2023",
                                "normalized": "2023-01-15",
                                "date": "2023-01-15",
                                "month": "2023-01",
                                "year": "2023"
                            }
                        ],
                        "spatial": [
                            {
                                "text": "Sydney",
                                "latitude": -33.8688,
                                "longitude": 151.2093,
                                "city": "Sydney",
                                "state": "New South Wales",
                                "country": "Australia"
                            }
                        ]
                    }
                }
            },
            {
                "document_id": "doc_002",
                "extraction": {
                    "success": True,
                    "entities": {
                        "temporal": [
                            {
                                "text": "January 16, 2023",
                                "normalized": "2023-01-16",
                                "date": "2023-01-16",
                                "month": "2023-01",
                                "year": "2023"
                            }
                        ],
                        "spatial": [
                            {
                                "text": "Melbourne",
                                "latitude": -37.8136,
                                "longitude": 144.9631,
                                "city": "Melbourne",
                                "state": "Victoria",
                                "country": "Australia"
                            }
                        ]
                    }
                }
            }
        ]

        analyzer = DimensionAnalyzer()
        analysis = analyzer.analyze(sample_results)

        assert analysis is not None, "Analysis returned None"
        assert "temporal" in analysis, "Missing temporal analysis"
        assert "spatial" in analysis, "Missing spatial analysis"

        print(f"✓ Generated statistics for dimensions:")
        print(f"  • Temporal: {analysis['temporal'].get('total_entities', 0)} entities")
        print(f"  • Spatial: {analysis['spatial'].get('total_entities', 0)} entities")

        if "global" in analysis:
            print(f"  • Global stats: {list(analysis['global'].keys())}")

        print("✅ PASS: Dimension statistics working\n")
        return True

    except Exception as e:
        print(f"✗ FAIL: Dimension statistics failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all corpus analysis tests."""
    print("\n" + "=" * 80)
    print("CORPUS ANALYSIS INTEGRATION TESTS")
    print("Testing hierarchy format compatibility")
    print("=" * 80)

    tests = [
        ("DimensionAnalyzer initialization", test_dimension_analyzer_initialization),
        ("Analyze sample results", test_analyze_sample_results),
        ("Dimension statistics", test_dimension_stats),
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
        print("\n🎉 All corpus analysis tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
