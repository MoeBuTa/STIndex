#!/usr/bin/env python3
"""
Test Qwen3-8B's ability to generate geographic coordinates.

This script evaluates whether Qwen3-8B can accurately predict latitude/longitude
for various locations compared to ground truth from geocoding APIs.
"""

import json
from stindex.llm.local_llm import LocalQwenLLM
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import time


# Test cases with ground truth coordinates from Nominatim
TEST_LOCATIONS = [
    # Well-known cities (should be easy)
    {"name": "Paris, France", "expected_lat": 48.8566, "expected_lon": 2.3522},
    {"name": "New York, USA", "expected_lat": 40.7128, "expected_lon": -74.0060},
    {"name": "Tokyo, Japan", "expected_lat": 35.6762, "expected_lon": 139.6503},

    # Ambiguous locations (disambiguation challenge)
    {"name": "Springfield, USA", "expected_lat": None, "expected_lon": None},  # Multiple Springfields
    {"name": "Paris, Texas", "expected_lat": 33.6609, "expected_lon": -95.5555},

    # Australian locations (STIndex use case)
    {"name": "Broome, Western Australia", "expected_lat": -17.9614, "expected_lon": 122.2359},
    {"name": "Fitzroy Crossing, Western Australia", "expected_lat": -18.1981, "expected_lon": 125.5664},

    # Less common locations (knowledge boundary test)
    {"name": "Noordwijk, Netherlands", "expected_lat": 52.2397, "expected_lon": 4.4297},
    {"name": "Lhasa, Tibet", "expected_lat": 29.6500, "expected_lon": 91.1000},

    # Geographic features
    {"name": "Mount Everest", "expected_lat": 27.9881, "expected_lon": 86.9250},
]


def get_ground_truth_coordinates(location_name):
    """Get authoritative coordinates from Nominatim."""
    try:
        geolocator = Nominatim(user_agent="stindex_test")
        location = geolocator.geocode(location_name)
        if location:
            return location.latitude, location.longitude
        return None, None
    except Exception as e:
        print(f"Geocoding error for {location_name}: {e}")
        return None, None


def test_llm_coordinate_generation():
    """Test if Qwen3-8B can generate accurate coordinates."""

    print("=" * 80)
    print("Testing Qwen3-8B Geographic Coordinate Generation")
    print("=" * 80)

    # Initialize Qwen3-8B
    print("\nLoading Qwen3-8B model...")
    llm = LocalQwenLLM(
        model_name="Qwen/Qwen3-8B",
        device="auto",
        temperature=0.0  # Deterministic output
    )
    print("Model loaded successfully!\n")

    results = []

    for i, test_case in enumerate(TEST_LOCATIONS):
        location_name = test_case["name"]
        print(f"\n[{i+1}/{len(TEST_LOCATIONS)}] Testing: {location_name}")
        print("-" * 80)

        # Prompt LLM for coordinates
        prompt = f"""What are the precise geographic coordinates (latitude and longitude) for the following location?

Location: {location_name}

Provide your answer in JSON format with exactly this structure:
{{
  "latitude": <decimal number>,
  "longitude": <decimal number>,
  "confidence": <0.0 to 1.0>
}}

Output (JSON only):"""

        # Get LLM prediction
        llm_response = llm.generate_structured(prompt)

        if "error" in llm_response:
            print(f"❌ LLM Error: {llm_response.get('error')}")
            results.append({
                "location": location_name,
                "status": "llm_error",
                "error": llm_response.get("error")
            })
            continue

        llm_lat = llm_response.get("latitude")
        llm_lon = llm_response.get("longitude")
        llm_confidence = llm_response.get("confidence", "N/A")

        # Get ground truth
        gt_lat, gt_lon = get_ground_truth_coordinates(location_name)
        time.sleep(1)  # Respect Nominatim rate limit

        if gt_lat is None or llm_lat is None:
            print(f"⚠️  Skipped: Ground truth or LLM prediction unavailable")
            results.append({
                "location": location_name,
                "status": "incomplete",
                "llm_prediction": {"lat": llm_lat, "lon": llm_lon},
                "ground_truth": {"lat": gt_lat, "lon": gt_lon}
            })
            continue

        # Calculate error distance
        llm_coords = (llm_lat, llm_lon)
        gt_coords = (gt_lat, gt_lon)
        error_km = geodesic(llm_coords, gt_coords).kilometers

        # Determine accuracy
        if error_km < 1:
            accuracy = "✅ EXCELLENT"
        elif error_km < 25:
            accuracy = "✓ GOOD"
        elif error_km < 200:
            accuracy = "⚠ FAIR"
        else:
            accuracy = "❌ POOR"

        print(f"LLM Prediction:  ({llm_lat:.4f}, {llm_lon:.4f}) [confidence: {llm_confidence}]")
        print(f"Ground Truth:    ({gt_lat:.4f}, {gt_lon:.4f})")
        print(f"Error Distance:  {error_km:.2f} km")
        print(f"Assessment:      {accuracy}")

        results.append({
            "location": location_name,
            "status": "success",
            "llm_prediction": {"lat": llm_lat, "lon": llm_lon, "confidence": llm_confidence},
            "ground_truth": {"lat": gt_lat, "lon": gt_lon},
            "error_km": round(error_km, 2),
            "accuracy": accuracy
        })

    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    successful_tests = [r for r in results if r["status"] == "success"]
    if successful_tests:
        total_tests = len(successful_tests)
        excellent = len([r for r in successful_tests if "EXCELLENT" in r["accuracy"]])
        good = len([r for r in successful_tests if "GOOD" in r["accuracy"]])
        fair = len([r for r in successful_tests if "FAIR" in r["accuracy"]])
        poor = len([r for r in successful_tests if "POOR" in r["accuracy"]])

        avg_error = sum(r["error_km"] for r in successful_tests) / total_tests

        print(f"\nTotal Tests: {total_tests}")
        print(f"  ✅ Excellent (<1 km):      {excellent} ({excellent/total_tests*100:.1f}%)")
        print(f"  ✓  Good (<25 km):          {good} ({good/total_tests*100:.1f}%)")
        print(f"  ⚠  Fair (<200 km):         {fair} ({fair/total_tests*100:.1f}%)")
        print(f"  ❌ Poor (>200 km):         {poor} ({poor/total_tests*100:.1f}%)")
        print(f"\nAverage Error Distance: {avg_error:.2f} km")

        # Determine overall recommendation
        accuracy_rate = (excellent + good) / total_tests * 100
        print(f"\nAccuracy Rate (<25km): {accuracy_rate:.1f}%")

        if accuracy_rate >= 90:
            recommendation = "✅ LLM CAN be used for coordinate generation"
        elif accuracy_rate >= 70:
            recommendation = "⚠️ LLM MAY be used with caution (verify critical coords)"
        else:
            recommendation = "❌ LLM SHOULD NOT be used for coordinates (use geocoding API)"

        print(f"\n{recommendation}")
    else:
        print("No successful tests completed.")

    # Save results
    output_file = "data/output/qwen3_geocoding_test_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to: {output_file}")

    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print("""
Based on GDELT research (2024), LLMs typically show:
- GPT-4.0: 32-55% accuracy with 33% hallucination rate
- High instability across runs (different results for same prompt)
- Geographic bias (better at Western locations)

This test evaluates if Qwen3-8B matches, exceeds, or falls below these baselines.

If error rates are high, STIndex's current architecture is correct:
  LLM: Entity detection + context extraction ✓
  Nominatim: Authoritative coordinate lookup ✓
""")


if __name__ == "__main__":
    test_llm_coordinate_generation()
