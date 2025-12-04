"""
Realistic MIRAGE example: How dynamic context works in schema discovery.

Shows how context memory maintains consistency when extracting from
actual MIRAGE medical questions across a cluster.
"""

from stindex.extraction.context_manager import ExtractionContext


def mirage_realistic_example():
    """Demonstrate context memory with actual MIRAGE-style questions."""

    print("=" * 80)
    print("MIRAGE Dataset: Context Memory Example")
    print("Cluster 0: Anatomy Questions (654 total)")
    print("=" * 80)

    # Initialize context for Cluster 0
    context = ExtractionContext(
        document_metadata={'cluster_id': 0, 'dataset': 'mirage'},
        max_memory_refs=10
    )

    # === Question 1 ===
    print("\n" + "─" * 80)
    print("QUESTION 1/654:")
    print("─" * 80)
    print("Which of the following muscle divides the sub-mandibular gland into")
    print("a superficial and deep part?")
    print()

    context.set_chunk_position(0, 654)

    # Simulated LLM extraction (would come from actual LLM call)
    extraction_1 = {
        'anatomical_structure': [
            {'text': 'submandibular gland', 'type': 'salivary_gland'},
            {'text': 'mylohyoid muscle', 'type': 'muscle'}
        ],
        'anatomical_location': [
            {'text': 'superficial part'},
            {'text': 'deep part'}
        ]
    }

    context.update_memory(extraction_1)
    print("Extracted dimensions: anatomical_structure, anatomical_location")

    # === Question 2 ===
    print("\n" + "─" * 80)
    print("QUESTION 2/654:")
    print("─" * 80)
    print("What is the nerve supply of the angle of the jaw?")
    print()

    context.set_chunk_position(1, 654)

    # LLM sees context from Q1:
    print("🔍 LLM SEES THIS CONTEXT:")
    print(context.to_prompt_context())

    # Simulated extraction - LLM uses context to maintain consistency
    extraction_2 = {
        'nerve': [
            {'text': 'great auricular nerve', 'type': 'sensory_nerve'}
        ],
        'anatomical_structure': [
            {'text': 'angle of mandible', 'type': 'bone'},  # Uses "mandible" (consistent)
        ]
    }

    context.update_memory(extraction_2)
    print("Extracted dimensions: nerve, anatomical_structure")
    print("✓ Used 'mandible' (consistent with prior 'submandibular')")

    # === Question 3 ===
    print("\n" + "─" * 80)
    print("QUESTION 3/654:")
    print("─" * 80)
    print("Which of the following cranial nerve not associated with olfaction?")
    print()

    context.set_chunk_position(2, 654)

    print("🔍 LLM SEES THIS CONTEXT:")
    print(context.to_prompt_context())

    extraction_3 = {
        'cranial_nerve': [
            {'text': 'olfactory nerve', 'number': 'CN I', 'category': 'sensory'},
            {'text': 'optic nerve', 'number': 'CN II', 'category': 'sensory'}
        ],
        'sensory_function': [
            {'text': 'olfaction', 'type': 'smell'}
        ]
    }

    context.update_memory(extraction_3)
    print("Extracted dimensions: cranial_nerve, sensory_function")

    # === Question 15 (later in cluster) ===
    print("\n" + "─" * 80)
    print("QUESTION 15/654 (Later in cluster):")
    print("─" * 80)
    print("During periradicular surgery, which nerve must be avoided?")
    print()

    context.set_chunk_position(14, 654)

    # Simulate 11 more extractions (sliding window keeps last 10)
    for i in range(3, 14):
        dummy_extraction = {
            'anatomical_structure': [{'text': f'structure_{i}', 'type': 'generic'}]
        }
        context.set_chunk_position(i, 654)
        context.update_memory(dummy_extraction)

    print("🔍 LLM SEES THIS CONTEXT:")
    print("(After 11 more questions, sliding window keeps last 10 refs per dimension)")
    print(context.to_prompt_context())

    extraction_15 = {
        'nerve': [
            {'text': 'inferior alveolar nerve', 'type': 'sensory_nerve'}
        ],
        'surgical_procedure': [
            {'text': 'periradicular surgery', 'category': 'endodontic_surgery'}
        ]
    }

    context.update_memory(extraction_15)
    print("Extracted dimensions: nerve, surgical_procedure")
    print("✓ 'great auricular nerve' from Q2 is still in context (within last 10)")

    # === Final State ===
    print("\n" + "=" * 80)
    print("FINAL CONTEXT STATE (Question 15/654)")
    print("=" * 80)
    print(f"\nDimensions discovered so far: {sorted(context.prior_refs.keys())}")
    print(f"\nMemory counts (last 10 refs per dimension):")
    for dim in sorted(context.prior_refs.keys()):
        refs = context.prior_refs[dim]
        print(f"  • {dim}: {len(refs)} references")
        if refs:
            # Show first 3 examples
            for ref in refs[:3]:
                text = ref.get('text', '')
                type_or_cat = ref.get('type') or ref.get('category') or ''
                if type_or_cat:
                    print(f"    - {text} ({type_or_cat})")
                else:
                    print(f"    - {text}")

    print("\n" + "=" * 80)
    print("KEY BENEFITS FOR MIRAGE SCHEMA DISCOVERY:")
    print("=" * 80)
    print("""
1. CONSISTENCY: "submandibular" in Q1 → "mandible" in Q2 (related terms)
2. TERMINOLOGY: LLM learns dataset's naming conventions from prior extractions
3. DISAMBIGUATION: "nerve" in Q2 vs "cranial_nerve" in Q3 (different dimensions)
4. HIERARCHY: Structures like "salivary_gland" → helps build hierarchy later
5. SLIDING WINDOW: Keeps last 10 refs per dimension (memory efficient)
6. DOMAIN-AGNOSTIC: Works for medical (MIRAGE) or any other domain (finance, legal)
    """)

    print("=" * 80)
    print("NEXT: Use this context in ClusterEntityPrompt")
    print("      to extract from all 654 questions in Cluster 0")
    print("=" * 80)


if __name__ == "__main__":
    mirage_realistic_example()
