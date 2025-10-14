"""
Example demonstrating integration with pandas for data analysis.
"""

import pandas as pd
from stindex import STIndexExtractor


def create_sample_dataset():
    """Create sample news dataset."""
    return [
        {
            "id": 1,
            "title": "Cyclone Hits Australia",
            "content": "On March 15, 2022, a strong cyclone hit Broome, Western Australia.",
        },
        {
            "id": 2,
            "title": "Summit in Paris",
            "content": "World leaders met in Paris, France on June 10, 2023 for climate talks.",
        },
        {
            "id": 3,
            "title": "Tech Conference",
            "content": "The annual tech conference will be held in San Francisco from September 20-22, 2024.",
        },
        {
            "id": 4,
            "title": "Archaeological Discovery",
            "content": "On January 15, 2024, archaeologists in Cairo, Egypt uncovered ancient artifacts.",
        },
    ]


def main():
    print("=" * 80)
    print("STIndex with Pandas Integration")
    print("=" * 80)

    # Create sample dataset
    data = create_sample_dataset()
    df = pd.DataFrame(data)

    print("\nOriginal DataFrame:")
    print(df)

    # Initialize extractor
    extractor = STIndexExtractor()

    # Extract spatiotemporal indices
    print("\nExtracting spatiotemporal indices...")

    results = []
    for _, row in df.iterrows():
        result = extractor.extract(row["content"])
        results.append(
            {
                "id": row["id"],
                "title": row["title"],
                "temporal_count": result.temporal_count,
                "spatial_count": result.spatial_count,
                "dates": [e.normalized for e in result.temporal_entities],
                "locations": [e.text for e in result.spatial_entities],
                "coordinates": [
                    (e.latitude, e.longitude) for e in result.spatial_entities
                ],
            }
        )

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    print("\nExtraction Results:")
    print(results_df)

    # Analyze temporal distribution
    print("\n" + "=" * 80)
    print("Temporal Analysis")
    print("=" * 80)

    all_dates = []
    for dates in results_df["dates"]:
        all_dates.extend(dates)

    if all_dates:
        dates_df = pd.DataFrame({"date": pd.to_datetime(all_dates)})
        print(f"\nDate range: {dates_df['date'].min()} to {dates_df['date'].max()}")
        print(f"Total events: {len(dates_df)}")

    # Analyze spatial distribution
    print("\n" + "=" * 80)
    print("Spatial Analysis")
    print("=" * 80)

    all_locations = []
    for locs in results_df["locations"]:
        all_locations.extend(locs)

    if all_locations:
        loc_counts = pd.Series(all_locations).value_counts()
        print("\nLocation frequency:")
        print(loc_counts)

    # Export results
    output_file = "stindex_results.csv"
    results_df.to_csv(output_file, index=False)
    print(f"\n✓ Results exported to: {output_file}")


if __name__ == "__main__":
    main()
