"""
Test STIndex with Pure English Test Suite
"""

import sys
sys.path.insert(0, '/media/liuyu/DataDrive/WWW2026_demo/stindex')

from stindex import STIndexExtractor
from stindex.models.schemas import ExtractionConfig

# Test configuration
config = ExtractionConfig(
    llm_provider="local",
    model_name="Qwen/Qwen3-8B",
    enable_temporal=True,
    enable_spatial=True,
)

print("=" * 100)
print("STIndex - English Test Suite Evaluation")
print("=" * 100)
print("\nLoading model...")

extractor = STIndexExtractor(config=config)

print("✓ Model loaded\n")

# Test cases organized by category
test_cases = {
    "Temporal": [
        ("T1.1", "Absolute Dates with Explicit Years",
         "The project started on January 15, 2020, was paused on March 20, 2021, and resumed on September 5, 2022."),

        ("T1.2", "Dates Without Years - Year Inference",
         "In 2023, the conference began on March 10. The workshop was on March 11, and the closing ceremony happened on March 12."),

        ("T1.3", "Date Intervals",
         "The exhibition will run from May 1, 2024 to May 31, 2024."),

        ("T1.4", "Mixed Date Formats",
         "The event on 2024-06-15 follows the announcement from June 1, 2024, and precedes the deadline of July 15, 2024."),

        ("T1.5", "Relative Time Expressions",
         "The meeting was yesterday, the report is due tomorrow, and the review happens next week."),

        ("T1.6", "Dates with Specific Times",
         "The webinar starts at 2:00 PM on March 15, 2024."),

        ("T1.7", "Duration Expressions",
         "The training program lasts 3 weeks."),

        ("T1.8", "Complex Temporal Context",
         "The study began in January 2020, was interrupted in March 2020 due to COVID-19, and resumed in September 2021."),

        ("T1.9", "Cross-Year Intervals",
         "The study ran from December 2022 to February 2023."),

        ("T1.10", "Historical Dates",
         "World War II ended on September 2, 1945. The Berlin Wall fell on November 9, 1989."),
    ],

    "Spatial": [
        ("S2.1", "Major World Cities",
         "The tour includes stops in Paris, Tokyo, New York, and Sydney."),

        ("S2.2", "Cities with Country Context",
         "The conference has venues in Berlin, Germany; Toronto, Canada; and Melbourne, Australia."),

        ("S2.3", "Ambiguous Place Names",
         "Springfield, Illinois is the state capital. Springfield, Massachusetts has a different history."),

        ("S2.4", "States and Regions",
         "California, Texas, and Florida are the most populous US states."),

        ("S2.5", "Landmarks",
         "The Eiffel Tower in Paris and the Statue of Liberty in New York are iconic landmarks."),

        ("S2.6", "Multiple Locations in Same Country",
         "The Australian tour covers Sydney, Melbourne, Brisbane, Perth, and Adelaide."),

        ("S2.7", "Small Towns with State Context",
         "The study was conducted in Boulder, Colorado and Ann Arbor, Michigan."),

        ("S2.8", "African Cities",
         "The research team visited Lagos, Nigeria; Nairobi, Kenya; and Cairo, Egypt."),

        ("S2.9", "Asian Cities",
         "The company has offices in Singapore, Seoul, Bangkok, and Mumbai."),

        ("S2.10", "European Capitals",
         "The summit rotates between Brussels, Geneva, Vienna, and Copenhagen."),
    ],

    "Combined": [
        ("C3.1", "News Report - Hurricane",
         "On August 29, 2005, Hurricane Katrina made landfall near New Orleans, Louisiana. By August 31, the storm had moved through Mississippi."),

        ("C3.2", "Travel Itinerary",
         "We'll arrive in Rome on June 5, 2024, stay three days, then travel to Florence on June 8."),

        ("C3.3", "Conference Announcement",
         "The International AI Conference will be held in Singapore from September 15-20, 2024."),

        ("C3.4", "Historical Event - Moon Landing",
         "On July 20, 1969, Apollo 11 landed on the Moon."),

        ("C3.5", "Business Expansion Timeline",
         "The company opened its Tokyo office in March 2020, followed by Shanghai in July 2020."),

        ("C3.6", "Research Field Study",
         "The expedition began in Nairobi, Kenya on February 1, 2023. Researchers spent two weeks in the Serengeti."),

        ("C3.7", "Sports Event",
         "The 2026 FIFA World Cup will be jointly hosted by the United States, Canada, and Mexico from June 11 to July 19, 2026."),

        ("C3.8", "Climate Event (PDF Example)",
         "On March 15, 2022, a strong cyclone hit the coastal areas near Broome, Western Australia and later moved inland by March 17."),

        ("C3.9", "Political Event - Summit",
         "The G20 Summit took place in Bali, Indonesia on November 15-16, 2022."),

        ("C3.10", "Natural Disaster Timeline",
         "The earthquake struck off the coast of Sumatra on December 26, 2004. The tsunami affected Thailand and Sri Lanka."),
    ],

    "Edge Cases": [
        ("E4.1", "No Spatiotemporal Information",
         "The algorithm uses machine learning to optimize performance."),

        ("E4.2", "Dense Information",
         "Between January 5 and January 10, 2024, the team visited Paris, London, Berlin, and Amsterdam."),

        ("E4.3", "Nested Locations",
         "The office is located in Austin, Texas, United States, near the University of Texas campus."),

        ("E4.4", "Non-English Place Names",
         "The meeting will be held in Beijing, China on December 1, 2024."),

        ("E4.5", "Multiple Ambiguous References",
         "Cambridge researchers met with Cambridge colleagues to discuss the Cambridge study."),
    ],

    "Challenge": [
        ("A1", "Scientific Abstract",
         "The study, conducted between March 2019 and August 2021 at Stanford University, California, analyzed climate data from Alaska, Greenland, and Antarctica."),

        ("A2", "Historical Narrative",
         "The expedition departed from London on May 19, 1845. The ships were last seen in Baffin Bay in July 1845, and the fate of the crew remained unknown until artifacts were discovered in the Canadian Arctic in 2014."),

        ("A3", "Flight Schedule",
         "Flight AA123 departs Los Angeles at 8:15 AM on Monday, arrives in Chicago at 2:30 PM, and continues to New York, landing at 6:45 PM."),

        ("A4", "Medical Record",
         "Patient admitted to Massachusetts General Hospital on January 5, 2024. Symptoms began three days earlier. Follow-up scheduled for February 12."),

        ("A5", "Corporate Timeline",
         "Founded in Seattle in 1994, the company expanded to San Francisco in 2000, opened European headquarters in Dublin in 2008, and established Asian operations in Singapore by 2015."),
    ],
}

# Statistics
stats = {
    "total": 0,
    "success": 0,
    "error": 0,
    "temporal_extracted": 0,
    "spatial_extracted": 0,
}

# Run tests
for category, tests in test_cases.items():
    print("\n" + "=" * 100)
    print(f"SECTION: {category.upper()}")
    print("=" * 100)

    for test_id, test_name, text in tests:
        print(f"\n{'─' * 100}")
        print(f"[{test_id}] {test_name}")
        print(f"{'─' * 100}")
        print(f"Input: {text[:80]}{'...' if len(text) > 80 else ''}")
        print()

        stats["total"] += 1

        try:
            result = extractor.extract(text)

            # Display temporal results
            if result.temporal_entities:
                print(f"Temporal ({len(result.temporal_entities)}):")
                for entity in result.temporal_entities:
                    print(f"  • '{entity.text}' → {entity.normalized} [{entity.temporal_type.value}]")
                stats["temporal_extracted"] += len(result.temporal_entities)
            else:
                print("Temporal: None")

            print()

            # Display spatial results
            if result.spatial_entities:
                print(f"Spatial ({len(result.spatial_entities)}):")
                for entity in result.spatial_entities:
                    lat_str = f"{abs(entity.latitude):.4f}° {'S' if entity.latitude < 0 else 'N'}"
                    lon_str = f"{abs(entity.longitude):.4f}° {'E' if entity.longitude > 0 else 'W'}"
                    print(f"  • '{entity.text}' → ({lat_str}, {lon_str})")
                stats["spatial_extracted"] += len(result.spatial_entities)
            else:
                print("Spatial: None")

            print(f"\n✓ Success")
            stats["success"] += 1

        except Exception as e:
            print(f"✗ Error: {str(e)[:100]}")
            stats["error"] += 1

# Summary
print("\n" + "=" * 100)
print("TEST SUMMARY")
print("=" * 100)

print(f"\nOverall:")
print(f"  Total Tests: {stats['total']}")
print(f"  Success: {stats['success']} ({100*stats['success']/stats['total']:.1f}%)")
print(f"  Error: {stats['error']} ({100*stats['error']/stats['total']:.1f}%)")

print(f"\nExtraction Statistics:")
print(f"  Temporal Entities: {stats['temporal_extracted']}")
print(f"  Spatial Entities: {stats['spatial_extracted']}")
print(f"  Total Entities: {stats['temporal_extracted'] + stats['spatial_extracted']}")

if stats['success'] > 0:
    print(f"\nAverage per Test:")
    print(f"  Temporal: {stats['temporal_extracted']/stats['success']:.1f}")
    print(f"  Spatial: {stats['spatial_extracted']/stats['success']:.1f}")

print("\n" + "=" * 100)
print("Evaluation Complete!")
print("=" * 100)
