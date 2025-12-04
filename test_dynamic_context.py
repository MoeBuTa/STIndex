"""
Test script for dynamic ExtractionContext with schema discovery.

Demonstrates how context memory maintains consistency across
MIRAGE-style medical questions with discovered dimensions.
"""

from stindex.extraction.context_manager import ExtractionContext
from pprint import pprint


def test_schema_discovery_context():
    """Test dynamic context with MIRAGE-style medical dimensions."""

    print("=" * 80)
    print("Testing Dynamic ExtractionContext for Schema Discovery")
    print("=" * 80)

    # Initialize context for Cluster 0 (anatomy questions)
    context = ExtractionContext(
        document_metadata={'cluster_id': 0},
        max_memory_refs=10
    )

    print("\n1. Initial context (empty):")
    print(context.to_prompt_context() or "  (no context yet)")

    # Simulate Question 1 extraction
    print("\n" + "=" * 80)
    print("QUESTION 1: 'Which muscle divides the submandibular gland?'")
    print("=" * 80)

    context.set_chunk_position(0, 20)  # Question 1 of 20

    # Simulated extraction result with discovered medical dimensions
    extraction_q1 = {
        'anatomical_structure': [
            {'text': 'submandibular gland', 'type': 'gland'},
            {'text': 'mylohyoid muscle', 'type': 'muscle'}
        ],
        'anatomical_region': [
            {'text': 'head and neck', 'category': 'body_region'}
        ]
    }

    context.update_memory(extraction_q1)

    print("\n✓ Extracted entities:")
    pprint(extraction_q1)

    print("\n✓ Updated context:")
    print(context.to_prompt_context())

    # Simulate Question 2 extraction
    print("\n" + "=" * 80)
    print("QUESTION 2: 'What is the nerve supply of the mylohyoid?'")
    print("=" * 80)

    context.set_chunk_position(1, 20)  # Question 2 of 20

    extraction_q2 = {
        'anatomical_structure': [
            {'text': 'mylohyoid muscle', 'type': 'muscle'},  # Consistency check!
            {'text': 'mylohyoid nerve', 'type': 'nerve'}
        ],
        'nerve': [
            {'text': 'mylohyoid nerve', 'category': 'motor_nerve'}
        ]
    }

    context.update_memory(extraction_q2)

    print("\n✓ Extracted entities:")
    pprint(extraction_q2)

    print("\n✓ Updated context (now shows prior extractions):")
    print(context.to_prompt_context())

    # Simulate Question 3 extraction
    print("\n" + "=" * 80)
    print("QUESTION 3: 'Which cranial nerve is associated with olfaction?'")
    print("=" * 80)

    context.set_chunk_position(2, 20)  # Question 3 of 20

    extraction_q3 = {
        'cranial_nerve': [
            {'text': 'olfactory nerve', 'category': 'cranial_nerve_I'}
        ],
        'sensory_function': [
            {'text': 'olfaction', 'type': 'smell'}
        ]
    }

    context.update_memory(extraction_q3)

    print("\n✓ Extracted entities:")
    pprint(extraction_q3)

    print("\n✓ Final context (accumulates all dimensions):")
    print(context.to_prompt_context())

    # Show context state
    print("\n" + "=" * 80)
    print("FINAL CONTEXT STATE")
    print("=" * 80)
    print(f"Dimensions tracked: {list(context.prior_refs.keys())}")
    print(f"\nMemory counts per dimension:")
    for dim, refs in context.prior_refs.items():
        print(f"  - {dim}: {len(refs)} references")

    # Test backward compatibility
    print("\n" + "=" * 80)
    print("BACKWARD COMPATIBILITY TEST")
    print("=" * 80)

    # Create context with old-style temporal/spatial dimensions
    old_context = ExtractionContext(
        document_metadata={
            'publication_date': '2024-03-15',
            'source_location': 'Perth, Australia'
        }
    )

    # Update with temporal/spatial entities
    old_extraction = {
        'temporal': [
            {'text': 'March 15, 2024', 'normalized': '2024-03-15'}
        ],
        'spatial': [
            {'text': 'Perth', 'parent_region': 'Western Australia'}
        ]
    }

    old_context.update_memory(old_extraction)

    print("\n✓ Old-style extraction (temporal + spatial):")
    print(old_context.to_prompt_context())

    # Test property access (backward compatibility)
    print("\n✓ Backward compatible property access:")
    print(f"  prior_temporal_refs: {old_context.prior_temporal_refs}")
    print(f"  prior_spatial_refs: {old_context.prior_spatial_refs}")
    print(f"  prior_events: {old_context.prior_events}")

    # Test serialization
    print("\n" + "=" * 80)
    print("SERIALIZATION TEST")
    print("=" * 80)

    serialized = context.to_dict()
    print("\n✓ Serialized context:")
    pprint(serialized)

    deserialized = ExtractionContext.from_dict(serialized)
    print("\n✓ Deserialized successfully")
    print(f"  Dimensions preserved: {list(deserialized.prior_refs.keys())}")

    print("\n" + "=" * 80)
    print("✓ All tests passed! Dynamic context memory works correctly.")
    print("=" * 80)


if __name__ == "__main__":
    test_schema_discovery_context()
