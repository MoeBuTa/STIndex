"""
Demo script for context-aware extraction features.

Demonstrates Priority 1 and Priority 2 features from context-aware-extraction.md:
- Element-based chunking with hierarchical metadata
- Document memory system (ExtractionContext)
- Temporal context window
- OpenStreetMap nearby locations
- Two-pass verification
- Context-aware prompting
"""

from stindex.extraction.context_manager import ExtractionContext
from stindex.extraction.dimensional_extraction import DimensionalExtractor
from stindex.preprocess.chunking import DocumentChunker
from stindex.preprocess.input_models import ParsedDocument
from stindex.postprocess.spatial.osm_context import OSMContextProvider
from stindex.postprocess.verification import ExtractionVerifier
from stindex.llm.manager import LLMManager


def demo_element_based_chunking():
    """Demonstrate element-based chunking with hierarchical metadata."""
    print("=" * 60)
    print("DEMO 1: Element-Based Chunking")
    print("=" * 60)

    # Create a sample parsed document with sections
    parsed_doc = ParsedDocument(
        document_id='cyclone_report',
        title='Cyclone Ellie Impact Report',
        content='Full document text...',
        sections=[
            {'type': 'title', 'text': 'Executive Summary', 'level': 1},
            {'type': 'text', 'text': 'On March 15, 2022, Tropical Cyclone Ellie made landfall near Broome, Western Australia.'},
            {'type': 'title', 'text': 'Timeline of Events', 'level': 1},
            {'type': 'text', 'text': 'The cyclone formed on March 12 and intensified rapidly over the next three days.'},
            {'type': 'title', 'text': 'Impact Assessment', 'level': 1},
            {'type': 'text', 'text': 'Significant damage was reported in Broome and surrounding areas including Port Hedland.'},
        ],
        metadata={'source': 'weather_bureau'}
    )

    # Create chunker with element-based strategy
    chunker = DocumentChunker(
        max_chunk_size=500,
        strategy='element_based'
    )

    # Chunk the document
    chunks = chunker.chunk_parsed_document(parsed_doc)

    print(f"\n✓ Created {len(chunks)} chunks with metadata:")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i}:")
        print(f"  - Section: {chunk.section_hierarchy}")
        print(f"  - Keywords: {chunk.keywords[:3] if chunk.keywords else []}")
        print(f"  - Preview: {chunk.preview[:60]}...")
        print(f"  - Element types: {chunk.element_types}")


def demo_context_aware_extraction():
    """Demonstrate context-aware extraction with temporal/spatial memory."""
    print("\n" + "=" * 60)
    print("DEMO 2: Context-Aware Extraction")
    print("=" * 60)

    # Create extraction context
    context = ExtractionContext(
        document_metadata={
            'publication_date': '2022-03-16',
            'source_location': 'Australia'
        },
        max_memory_refs=10
    )

    # Simulate processing multiple chunks
    chunks = [
        "On March 15, 2022, a tropical cyclone hit Broome, Western Australia.",
        "The next day, the cyclone moved towards Port Hedland.",
        "Yesterday's damage assessment revealed significant structural damage."
    ]

    print("\nProcessing document chunks with context memory:")

    for i, chunk_text in enumerate(chunks):
        context.set_chunk_position(i, len(chunks), "Report > Events")

        print(f"\n--- Chunk {i + 1} ---")
        print(f"Text: {chunk_text}")

        # Simulate extraction results
        if i == 0:
            extraction_result = {
                'temporal_entities': [
                    {'text': 'March 15, 2022', 'normalized': '2022-03-15'}
                ],
                'spatial_entities': [
                    {'text': 'Broome', 'parent_region': 'Western Australia'}
                ]
            }
        elif i == 1:
            extraction_result = {
                'temporal_entities': [
                    {'text': 'The next day', 'normalized': '2022-03-16'}
                ],
                'spatial_entities': [
                    {'text': 'Port Hedland', 'parent_region': 'Western Australia'}
                ]
            }
        else:
            extraction_result = {
                'temporal_entities': [
                    {'text': "Yesterday's", 'normalized': '2022-03-15'}
                ]
            }

        # Update context memory
        context.update_memory(extraction_result)

        # Show context state
        print(f"Context: {len(context.prior_temporal_refs)} temporal, {len(context.prior_spatial_refs)} spatial refs")
        print(f"Anchor date: {context.get_anchor_date()}")

    # Show final context prompt
    print("\n--- Final Context Prompt ---")
    print(context.to_prompt_context())


def demo_osm_nearby_locations():
    """Demonstrate OSM nearby locations for spatial disambiguation."""
    print("\n" + "=" * 60)
    print("DEMO 3: OSM Nearby Locations")
    print("=" * 60)

    # Create OSM context provider
    osm = OSMContextProvider()

    # Example: Get nearby locations for Broome, WA (-17.9614, 122.2359)
    broome_coords = (-17.9614, 122.2359)

    print(f"\nQuerying nearby locations for Broome coordinates: {broome_coords}")
    print("(This requires internet connection to Overpass API)")

    try:
        nearby = osm.get_nearby_locations(broome_coords, radius_km=100)

        if nearby:
            print(f"\n✓ Found {len(nearby)} nearby locations:")
            for poi in nearby[:5]:
                print(f"  - {poi['name']} ({poi['type']}): {poi['distance_km']}km {poi['direction']}")
        else:
            print("\nNote: No nearby locations found (may be offline or API limit)")

        # Show as formatted context string
        context_str = osm.get_location_context_str(broome_coords, radius_km=100, max_display=3)
        if context_str:
            print("\nFormatted for LLM prompt:")
            print(context_str)
    except Exception as e:
        print(f"\nNote: OSM query failed (expected if offline): {e}")


def demo_two_pass_verification():
    """Demonstrate two-pass verification concept."""
    print("\n" + "=" * 60)
    print("DEMO 4: Two-Pass Verification (Concept)")
    print("=" * 60)

    print("\nTwo-pass verification workflow:")
    print("1. Pass 1: Extract entities (DimensionalExtractor)")
    print("2. Pass 2: Verify and score extractions (ExtractionVerifier)")
    print("\nFilters entities based on:")
    print("  - Relevance: Is it in the text?")
    print("  - Accuracy: Does it match exactly?")
    print("  - Completeness: Is it complete?")

    # Show example filtering
    print("\nExample:")
    print("  Input: 5 extracted entities")
    print("  Verification scores:")
    print("    Entity 1: relevance=0.95, accuracy=0.90 → PASS")
    print("    Entity 2: relevance=0.50, accuracy=0.60 → FAIL (low relevance)")
    print("    Entity 3: relevance=0.85, accuracy=0.85 → PASS")
    print("    Entity 4: relevance=0.40, accuracy=0.80 → FAIL (low relevance)")
    print("    Entity 5: relevance=0.95, accuracy=0.75 → PASS")
    print("  Output: 3 verified entities (40% reduction in false positives)")


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("CONTEXT-AWARE EXTRACTION DEMO")
    print("Priority 1 & 2 Features Implementation")
    print("=" * 60)

    demo_element_based_chunking()
    demo_context_aware_extraction()
    demo_osm_nearby_locations()
    demo_two_pass_verification()

    print("\n" + "=" * 60)
    print("✓ All demos complete!")
    print("=" * 60)
    print("\nFor full integration examples, see:")
    print("  - stindex/extraction/dimensional_extraction.py")
    print("  - stindex/extraction/context_manager.py")
    print("  - stindex/preprocess/chunking.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
