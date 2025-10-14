"""
Basic example of using STIndex for spatiotemporal extraction.
"""

from stindex import STIndexExtractor

def main():
    # Example text from the PDF
    text = """On March 15, 2022, a strong cyclone hit the coastal areas near
    Broome, Western Australia and later moved inland towards Fitzroy Crossing
    by March 17."""

    print("=" * 80)
    print("STIndex - Basic Example")
    print("=" * 80)
    print(f"\nInput text:\n{text}\n")

    # Initialize extractor
    print("Initializing STIndex extractor...")
    extractor = STIndexExtractor()

    # Extract spatiotemporal indices
    print("Extracting spatiotemporal indices...\n")
    result = extractor.extract(text)

    # Display results
    print("=" * 80)
    print("TEMPORAL ENTITIES")
    print("=" * 80)
    for entity in result.temporal_entities:
        print(f"Text: {entity.text}")
        print(f"Normalized: {entity.normalized}")
        print(f"Type: {entity.temporal_type.value}")
        print(f"Confidence: {entity.confidence:.2f}")
        print("-" * 40)

    print("\n" + "=" * 80)
    print("SPATIAL ENTITIES")
    print("=" * 80)
    for entity in result.spatial_entities:
        print(f"Text: {entity.text}")
        print(f"Coordinates: ({entity.latitude:.4f}°, {entity.longitude:.4f}°)")
        print(f"Address: {entity.address}")
        print(f"Country: {entity.country}")
        print(f"Confidence: {entity.confidence:.2f}")
        print("-" * 40)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total temporal entities: {result.temporal_count}")
    print(f"Total spatial entities: {result.spatial_count}")
    print(f"Processing time: {result.processing_time:.2f}s")

    # Export to JSON
    result_dict = result.to_dict()
    print("\nJSON Output (excerpt):")
    import json
    print(json.dumps(result_dict, indent=2, ensure_ascii=False)[:500] + "...")


if __name__ == "__main__":
    main()
