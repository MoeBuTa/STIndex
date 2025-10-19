"""
Comprehensive Test Suite for STIndex
Directly evaluates system capabilities without predefined answers
"""

import sys
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from stindex import ExtractionPipeline

# Setup output directory
output_dir = project_root / "data" / "output"
output_dir.mkdir(parents=True, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = output_dir / f"test_comprehensive_{timestamp}.json"
output_txt = output_dir / f"test_comprehensive_{timestamp}.txt"

# Test configuration
config = {
    "llm_provider": "local",
    "model_name": "Qwen/Qwen3-4B-Thinking-2507",
    "enable_temporal": True,
    "enable_spatial": True,
}

print("=" * 100)
print("STIndex Comprehensive Capability Test")
print("=" * 100)
print("\nLoading model...")

pipeline = ExtractionPipeline(config=config)

print("✓ Model loaded\n")

# Test results tracking
test_results = []

def run_test(category: str, name: str, text: str):
    """Run a test and display results."""
    print(f"\n{'─' * 100}")
    print(f"[{category}] {name}")
    print(f"{'─' * 100}")
    print(f"Input: {text}")
    print()

    try:
        result = pipeline.extract(text)

        # Display temporal results
        if result.temporal_entities:
            print(f"Temporal Entities ({len(result.temporal_entities)}):")
            for entity in result.temporal_entities:
                print(f"  • '{entity.get('text', '')}' → {entity.get('normalized', '')} [{entity.get('temporal_type', '')}]")
        else:
            print("Temporal Entities: None")

        print()

        # Display spatial results
        if result.spatial_entities:
            print(f"Spatial Entities ({len(result.spatial_entities)}):")
            for entity in result.spatial_entities:
                lat_str = f"{abs(entity.get('latitude', 0.0)):.4f}° {'S' if entity.get('latitude', 0.0) < 0 else 'N'}"
                lon_str = f"{abs(entity.get('longitude', 0.0)):.4f}° {'E' if entity.get('longitude', 0.0) > 0 else 'W'}"
                print(f"  • '{entity.get('text', '')}' → ({lat_str}, {lon_str})")
        else:
            print("Spatial Entities: None")

        test_results.append({
            "category": category,
            "name": name,
            "temporal_count": len(result.temporal_entities),
            "spatial_count": len(result.spatial_entities),
            "status": "success"
        })

    except Exception as e:
        print(f"✗ Error: {str(e)}")
        test_results.append({
            "category": category,
            "name": name,
            "status": "error",
            "error": str(e)
        })

# =============================================================================
# TEMPORAL TESTS
# =============================================================================

print("\n" + "=" * 100)
print("PART 1: Temporal Extraction Capabilities")
print("=" * 100)

run_test("Temporal", "1.1 Absolute Dates with Explicit Years",
    "The project started on January 15, 2020, was paused on March 20, 2021, and resumed on September 5, 2022.")

run_test("Temporal", "1.2 Dates Without Years - Year Inference",
    "In 2023, the conference began on March 10. The workshop was on March 11, and the closing ceremony happened on March 12.")

run_test("Temporal", "1.3 Date Intervals",
    "The exhibition will run from May 1, 2024 to May 31, 2024.")

run_test("Temporal", "1.4 Mixed Date Formats",
    "The event on 2024-06-15 follows the announcement from June 1, 2024, and precedes the deadline of July 15, 2024.")

run_test("Temporal", "1.5 Relative Time Expressions",
    "The meeting was yesterday, the report is due tomorrow, and the review happens next week.")

run_test("Temporal", "1.6 Dates with Specific Times",
    "The webinar starts at 2:00 PM on March 15, 2024.")

run_test("Temporal", "1.7 Duration Expressions",
    "The training program lasts 3 weeks.")

run_test("Temporal", "1.8 Complex Temporal Context",
    "The study began in January 2020, was interrupted in March 2020 due to COVID-19, and resumed in September 2021.")

run_test("Temporal", "1.9 Cross-Year Intervals",
    "The study ran from December 2022 to February 2023.")

run_test("Temporal", "1.10 Historical Dates",
    "World War II ended on September 2, 1945. The Berlin Wall fell on November 9, 1989.")

# =============================================================================
# SPATIAL TESTS
# =============================================================================

print("\n" + "=" * 100)
print("PART 2: Spatial Extraction Capabilities")
print("=" * 100)

run_test("Spatial", "2.1 Major World Cities",
    "The tour includes stops in Paris, Tokyo, New York, and Sydney.")

run_test("Spatial", "2.2 Cities with Country Context",
    "The conference has venues in Berlin, Germany; Toronto, Canada; and Melbourne, Australia.")

run_test("Spatial", "2.3 Ambiguous Place Names - Springfield",
    "Springfield, Illinois is the state capital. Springfield, Massachusetts has a different history.")

run_test("Spatial", "2.4 States and Regions",
    "California, Texas, and Florida are the most populous US states.")

run_test("Spatial", "2.5 Landmarks",
    "The Eiffel Tower in Paris and the Statue of Liberty in New York are iconic landmarks.")

run_test("Spatial", "2.6 Multiple Locations in Same Country",
    "The Australian tour covers Sydney, Melbourne, Brisbane, Perth, and Adelaide.")

run_test("Spatial", "2.7 Small Towns with State Context",
    "The study was conducted in Boulder, Colorado and Ann Arbor, Michigan.")

run_test("Spatial", "2.8 African Cities",
    "The research team visited Lagos, Nigeria; Nairobi, Kenya; and Cairo, Egypt.")

run_test("Spatial", "2.9 Asian Cities",
    "The company has offices in Singapore, Seoul, Bangkok, and Mumbai.")

run_test("Spatial", "2.10 European Capitals",
    "The summit rotates between Brussels, Geneva, Vienna, and Copenhagen.")

# =============================================================================
# COMBINED TESTS
# =============================================================================

print("\n" + "=" * 100)
print("PART 3: Combined Spatiotemporal Extraction")
print("=" * 100)

run_test("Combined", "3.1 News Report - Hurricane",
    "On August 29, 2005, Hurricane Katrina made landfall near New Orleans, Louisiana. By August 31, the storm had moved through Mississippi.")

run_test("Combined", "3.2 Travel Itinerary",
    "We'll arrive in Rome on June 5, 2024, stay three days, then travel to Florence on June 8.")

run_test("Combined", "3.3 Conference Announcement",
    "The International AI Conference will be held in Singapore from September 15-20, 2024.")

run_test("Combined", "3.4 Historical Event - Moon Landing",
    "On July 20, 1969, Apollo 11 landed on the Moon.")

run_test("Combined", "3.5 Business Expansion Timeline",
    "The company opened its Tokyo office in March 2020, followed by Shanghai in July 2020.")

run_test("Combined", "3.6 Research Field Study",
    "The expedition began in Nairobi, Kenya on February 1, 2023. Researchers spent two weeks in the Serengeti.")

run_test("Combined", "3.7 Sports Event",
    "The 2026 FIFA World Cup will be jointly hosted by the United States, Canada, and Mexico from June 11 to July 19, 2026.")

run_test("Combined", "3.8 Climate Event (PDF Example)",
    "On March 15, 2022, a strong cyclone hit the coastal areas near Broome, Western Australia and later moved inland by March 17.")

run_test("Combined", "3.9 Political Event - Summit",
    "The G20 Summit took place in Bali, Indonesia on November 15-16, 2022.")

run_test("Combined", "3.10 Natural Disaster Timeline",
    "The earthquake struck off the coast of Sumatra on December 26, 2004. The tsunami affected Thailand and Sri Lanka.")

# =============================================================================
# EDGE CASES
# =============================================================================

print("\n" + "=" * 100)
print("PART 4: Edge Cases")
print("=" * 100)

run_test("Edge Case", "4.1 No Spatiotemporal Information",
    "The algorithm uses machine learning to optimize performance.")

run_test("Edge Case", "4.2 Dense Information",
    "Between January 5 and January 10, 2024, the team visited Paris, London, Berlin, and Amsterdam.")

run_test("Edge Case", "4.3 Nested Locations",
    "The office is located in Austin, Texas, United States, near the University of Texas campus.")

run_test("Edge Case", "4.4 Non-English Place Names",
    "The meeting will be held in Beijing, China on December 1, 2024.")

run_test("Edge Case", "4.5 Multiple Ambiguous References",
    "Cambridge researchers met with Cambridge colleagues to discuss the Cambridge study.")

# =============================================================================
# RESULTS SUMMARY
# =============================================================================

summary_text = []

summary_text.append("=" * 100)
summary_text.append("TEST RESULTS SUMMARY")
summary_text.append("=" * 100)

# Count by category
categories = {}
for result in test_results:
    cat = result["category"]
    if cat not in categories:
        categories[cat] = {"total": 0, "success": 0, "error": 0}
    categories[cat]["total"] += 1
    if result["status"] == "success":
        categories[cat]["success"] += 1
    else:
        categories[cat]["error"] += 1

# Overall stats
total = len(test_results)
success = sum(1 for r in test_results if r["status"] == "success")
errors = total - success

summary_text.append(f"\nOverall Statistics:")
summary_text.append(f"  Total Tests: {total}")
summary_text.append(f"  Success: {success} ({100*success/total:.1f}%)")
summary_text.append(f"  Errors: {errors} ({100*errors/total:.1f}%)")

summary_text.append(f"\nCategory Breakdown:")
for cat, stats in categories.items():
    success_rate = 100 * stats["success"] / stats["total"]
    summary_text.append(f"  [{cat}] {stats['success']}/{stats['total']} ({success_rate:.1f}%)")

# Temporal/Spatial extraction stats
temporal_counts = [r.get("temporal_count", 0) for r in test_results if r["status"] == "success"]
spatial_counts = [r.get("spatial_count", 0) for r in test_results if r["status"] == "success"]

if temporal_counts:
    summary_text.append(f"\nTemporal Entity Extraction:")
    summary_text.append(f"  Total: {sum(temporal_counts)} entities")
    summary_text.append(f"  Average: {sum(temporal_counts)/len(temporal_counts):.1f} per test")

if spatial_counts:
    summary_text.append(f"\nSpatial Entity Extraction:")
    summary_text.append(f"  Total: {sum(spatial_counts)} entities")
    summary_text.append(f"  Average: {sum(spatial_counts)/len(spatial_counts):.1f} per test")

summary_text.append("\n" + "=" * 100)
summary_text.append("TEST COMPLETE")
summary_text.append("=" * 100)

# Print summary
for line in summary_text:
    print(line)

# Save results to JSON
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump({
        "timestamp": timestamp,
        "test_results": test_results,
        "summary": {
            "total": total,
            "success": success,
            "errors": errors,
            "categories": categories,
            "temporal_total": sum(temporal_counts) if temporal_counts else 0,
            "spatial_total": sum(spatial_counts) if spatial_counts else 0,
        }
    }, f, indent=2, ensure_ascii=False)

# Save summary to text file
with open(output_txt, 'w', encoding='utf-8') as f:
    f.write('\n'.join(summary_text))

print(f"\n✓ Results saved to:")
print(f"  JSON: {output_file}")
print(f"  TXT:  {output_txt}")
