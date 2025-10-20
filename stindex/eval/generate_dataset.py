"""
Generate evaluation dataset for STIndex.

Creates diverse test cases covering various spatiotemporal extraction scenarios.
"""

import json
from typing import List, Dict, Any
from datetime import datetime


def generate_evaluation_dataset(num_entries: int = 100) -> List[Dict[str, Any]]:
    """
    Generate evaluation dataset with ground truth annotations.

    Returns:
        List of dataset entries with text and ground truth
    """
    dataset = []

    # Diverse test cases with varying complexity
    test_cases = [
        # Simple cases - single date, single location
        {
            "text": "On March 15, 2022, a cyclone hit Broome, Western Australia.",
            "ground_truth": {
                "temporal": [
                    {"text": "March 15, 2022", "normalized": "2022-03-15", "temporal_type": "DATE"}
                ],
                "spatial": [
                    {"text": "Broome", "location_type": "CITY", "latitude": -17.9614, "longitude": 122.2359},
                    {"text": "Western Australia", "location_type": "REGION", "latitude": -25.0, "longitude": 122.0}
                ]
            }
        },
        {
            "text": "The conference will be held in Sydney on December 10, 2023.",
            "ground_truth": {
                "temporal": [
                    {"text": "December 10, 2023", "normalized": "2023-12-10", "temporal_type": "DATE"}
                ],
                "spatial": [
                    {"text": "Sydney", "location_type": "CITY", "latitude": -33.8688, "longitude": 151.2093}
                ]
            }
        },
        # Year inference
        {
            "text": "The project started on January 15, 2023 and ended on March 20.",
            "ground_truth": {
                "temporal": [
                    {"text": "January 15, 2023", "normalized": "2023-01-15", "temporal_type": "DATE"},
                    {"text": "March 20", "normalized": "2023-03-20", "temporal_type": "DATE"}
                ],
                "spatial": []
            }
        },
        # Multiple locations
        {
            "text": "The tour goes from Melbourne to Brisbane via Sydney.",
            "ground_truth": {
                "temporal": [],
                "spatial": [
                    {"text": "Melbourne", "location_type": "CITY", "latitude": -37.8136, "longitude": 144.9631},
                    {"text": "Brisbane", "location_type": "CITY", "latitude": -27.4698, "longitude": 153.0251},
                    {"text": "Sydney", "location_type": "CITY", "latitude": -33.8688, "longitude": 151.2093}
                ]
            }
        },
        # Complex temporal
        {
            "text": "Between June 1 and June 15, 2024, the team will conduct field research.",
            "ground_truth": {
                "temporal": [
                    {"text": "June 1", "normalized": "2024-06-01", "temporal_type": "DATE"},
                    {"text": "June 15, 2024", "normalized": "2024-06-15", "temporal_type": "DATE"}
                ],
                "spatial": []
            }
        },
        # Spatial disambiguation
        {
            "text": "Perth, Australia experienced heavy rainfall, unlike Perth, Scotland.",
            "ground_truth": {
                "temporal": [],
                "spatial": [
                    {"text": "Perth, Australia", "location_type": "CITY", "latitude": -31.9505, "longitude": 115.8605},
                    {"text": "Perth, Scotland", "location_type": "CITY", "latitude": 56.3959, "longitude": -3.4375}
                ]
            }
        },
        # Natural disaster scenario
        {
            "text": "On February 6, 2023, a devastating earthquake struck southern Turkey and northern Syria.",
            "ground_truth": {
                "temporal": [
                    {"text": "February 6, 2023", "normalized": "2023-02-06", "temporal_type": "DATE"}
                ],
                "spatial": [
                    {"text": "Turkey", "location_type": "COUNTRY", "latitude": 38.9637, "longitude": 35.2433},
                    {"text": "Syria", "location_type": "COUNTRY", "latitude": 34.8021, "longitude": 38.9968}
                ]
            }
        },
        # Time expressions
        {
            "text": "The meeting is scheduled for 3:00 PM on July 4, 2024.",
            "ground_truth": {
                "temporal": [
                    {"text": "3:00 PM", "normalized": "15:00:00", "temporal_type": "TIME"},
                    {"text": "July 4, 2024", "normalized": "2024-07-04", "temporal_type": "DATE"}
                ],
                "spatial": []
            }
        },
        # Historical event
        {
            "text": "The Berlin Wall fell on November 9, 1989 in Berlin, Germany.",
            "ground_truth": {
                "temporal": [
                    {"text": "November 9, 1989", "normalized": "1989-11-09", "temporal_type": "DATE"}
                ],
                "spatial": [
                    {"text": "Berlin", "location_type": "CITY", "latitude": 52.5200, "longitude": 13.4050},
                    {"text": "Germany", "location_type": "COUNTRY", "latitude": 51.1657, "longitude": 10.4515}
                ]
            }
        },
        # Climate event
        {
            "text": "Hurricane Katrina made landfall near New Orleans on August 29, 2005.",
            "ground_truth": {
                "temporal": [
                    {"text": "August 29, 2005", "normalized": "2005-08-29", "temporal_type": "DATE"}
                ],
                "spatial": [
                    {"text": "New Orleans", "location_type": "CITY", "latitude": 29.9511, "longitude": -90.0715}
                ]
            }
        },
    ]

    # Replicate and vary test cases to reach target number
    base_count = len(test_cases)
    repetitions = (num_entries + base_count - 1) // base_count

    for rep in range(repetitions):
        for idx, case in enumerate(test_cases):
            if len(dataset) >= num_entries:
                break

            entry_id = len(dataset) + 1

            # Add slight variations for repeated entries
            text = case["text"]
            if rep > 0:
                # Add context variations
                prefixes = [
                    "According to reports, ",
                    "News sources indicate that ",
                    "It was reported that ",
                    "Witnesses confirmed that ",
                    "Officials announced that "
                ]
                if idx < len(prefixes):
                    text = prefixes[idx % len(prefixes)] + text.lower()

            dataset.append({
                "id": f"entry_{entry_id:03d}",
                "text": text,
                "prompt": "Extract all temporal expressions and spatial locations from the following text. Normalize temporal expressions to ISO 8601 format and geocode spatial locations to coordinates.",
                "ground_truth": case["ground_truth"],
                "metadata": {
                    "category": "weather" if "cyclone" in text.lower() or "hurricane" in text.lower() else
                               "historical" if any(year in text for year in ["1989", "2005"]) else
                               "general",
                    "complexity": "simple" if len(case["ground_truth"]["temporal"]) + len(case["ground_truth"]["spatial"]) <= 2 else "complex"
                }
            })

    return dataset[:num_entries]


def save_dataset(dataset: List[Dict[str, Any]], output_path: str):
    """Save dataset to JSON file"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    print(f"Dataset with {len(dataset)} entries saved to {output_path}")


if __name__ == "__main__":
    # Generate 100-entry dataset
    dataset = generate_evaluation_dataset(100)

    # Save to file
    output_path = "data/input/eval_dataset_100.json"
    save_dataset(dataset, output_path)

    # Print summary
    print(f"\nDataset Summary:")
    print(f"Total entries: {len(dataset)}")

    temporal_counts = [len(e["ground_truth"]["temporal"]) for e in dataset]
    spatial_counts = [len(e["ground_truth"]["spatial"]) for e in dataset]

    print(f"Temporal entities: {sum(temporal_counts)} total, avg {sum(temporal_counts)/len(dataset):.1f} per entry")
    print(f"Spatial entities: {sum(spatial_counts)} total, avg {sum(spatial_counts)/len(dataset):.1f} per entry")

    # Category breakdown
    categories = {}
    for e in dataset:
        cat = e["metadata"]["category"]
        categories[cat] = categories.get(cat, 0) + 1

    print(f"\nCategories:")
    for cat, count in categories.items():
        print(f"  {cat}: {count}")
