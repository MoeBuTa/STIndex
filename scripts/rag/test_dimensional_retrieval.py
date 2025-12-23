#!/usr/bin/env python3
"""
Test dimensional retrieval: Compare Stage 3 (no filter) vs Stage 4 (with filter).

Creates 10 test samples from MIRAGE questions and evaluates:
1. Without dimensional filtering (baseline)
2. With dimensional filtering (using extracted dimensions)

Usage:
    python -m scripts.rag.test_dimensional_retrieval
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any

# Test samples with questions, expected answers, and dimension filters
# Each sample has:
# - question: The query
# - answer_keywords: Keywords that should appear in relevant context
# - dimension_filters: Filters to apply for Stage 4
# - description: What we're testing

TEST_SAMPLES = [
    {
        "id": 1,
        "question": "What is the mechanism of action of cisplatin in cancer treatment?",
        "answer_keywords": ["cross-link", "DNA", "platinum", "alkylating"],
        "dimension_filters": [{"dimension": "drug", "values": ["cisplatin"]}],
        "description": "Drug mechanism - cisplatin"
    },
    {
        "id": 2,
        "question": "What causes hearing loss as a side effect of chemotherapy?",
        "answer_keywords": ["ototoxicity", "cochlea", "cisplatin", "aminoglycoside"],
        "dimension_filters": [{"dimension": "side_effect", "values": ["hearing loss", "ototoxicity"]}],
        "description": "Side effect - ototoxicity"
    },
    {
        "id": 3,
        "question": "What is the treatment for diabetic ketoacidosis?",
        "answer_keywords": ["insulin", "fluid", "potassium", "bicarbonate"],
        "dimension_filters": [{"dimension": "drug", "values": ["insulin"]}],
        "description": "Drug treatment - insulin for DKA"
    },
    {
        "id": 4,
        "question": "What organism causes tuberculosis and how is it identified?",
        "answer_keywords": ["mycobacterium", "acid-fast", "tuberculosis", "AFB"],
        "dimension_filters": [{"dimension": "pathogen", "values": ["mycobacterium tuberculosis", "tuberculosis"]}],
        "description": "Pathogen - Mycobacterium tuberculosis"
    },
    {
        "id": 5,
        "question": "What are the symptoms of hyperthyroidism?",
        "answer_keywords": ["tachycardia", "weight loss", "tremor", "anxiety", "heat intolerance"],
        "dimension_filters": [{"dimension": "symptom", "values": ["tachycardia", "weight loss"]}],
        "description": "Symptoms - hyperthyroidism"
    },
    {
        "id": 6,
        "question": "What diagnostic test confirms meningitis?",
        "answer_keywords": ["lumbar puncture", "CSF", "cerebrospinal fluid", "culture"],
        "dimension_filters": [{"dimension": "diagnostic_standard", "values": ["lumbar puncture", "CSF analysis"]}],
        "description": "Diagnostic - lumbar puncture for meningitis"
    },
    {
        "id": 7,
        "question": "What is the mechanism of warfarin anticoagulation?",
        "answer_keywords": ["vitamin K", "clotting factors", "II", "VII", "IX", "X"],
        "dimension_filters": [{"dimension": "drug", "values": ["warfarin"]}],
        "description": "Drug mechanism - warfarin"
    },
    {
        "id": 8,
        "question": "What causes cholesterol embolization syndrome after cardiac catheterization?",
        "answer_keywords": ["cholesterol", "crystal", "atheroma", "embolization", "blue toe"],
        "dimension_filters": [{"dimension": "intervention", "values": ["catheterization", "cardiac catheterization"]}],
        "description": "Intervention complication - cardiac catheterization"
    },
    {
        "id": 9,
        "question": "What is the treatment for Staphylococcus aureus bacteremia?",
        "answer_keywords": ["vancomycin", "nafcillin", "oxacillin", "antibiotic"],
        "dimension_filters": [{"dimension": "pathogen", "values": ["staphylococcus aureus", "staphylococcus"]}],
        "description": "Pathogen treatment - S. aureus"
    },
    {
        "id": 10,
        "question": "What histological finding is characteristic of sarcoidosis?",
        "answer_keywords": ["granuloma", "non-caseating", "epithelioid", "giant cell"],
        "dimension_filters": [{"dimension": "histologic_finding", "values": ["granuloma", "non-caseating granuloma"]}],
        "description": "Histology - sarcoidosis"
    },
]


def count_keyword_matches(text: str, keywords: List[str]) -> int:
    """Count how many keywords appear in text."""
    text_lower = text.lower()
    return sum(1 for kw in keywords if kw.lower() in text_lower)


def evaluate_results(results: List[Any], answer_keywords: List[str]) -> Dict[str, Any]:
    """Evaluate retrieval results based on keyword matches."""
    if not results:
        return {"matches": 0, "total_keywords": len(answer_keywords), "top_match": 0, "mrr": 0}

    # Check each result for keyword matches
    matches_per_result = []
    for r in results:
        text = f"{r.title} {r.text}".lower()
        matches = count_keyword_matches(text, answer_keywords)
        matches_per_result.append(matches)

    # Find first result with any match (for MRR)
    first_match_rank = 0
    for i, m in enumerate(matches_per_result):
        if m > 0:
            first_match_rank = i + 1
            break

    return {
        "matches": sum(1 for m in matches_per_result if m > 0),
        "total_keywords": len(answer_keywords),
        "top_match": matches_per_result[0] if matches_per_result else 0,
        "mrr": 1.0 / first_match_rank if first_match_rank > 0 else 0,
        "keyword_hits": max(matches_per_result) if matches_per_result else 0,
    }


def main():
    from rag.retriever.three_stage_retriever import ThreeStageRetriever
    from loguru import logger

    # Reduce log verbosity
    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    print("=" * 80)
    print("Dimensional Retrieval Test: Stage 3 vs Stage 4")
    print("=" * 80)

    print("\nLoading ThreeStageRetriever...")
    retriever = ThreeStageRetriever(
        index_path="data/indices/textbook_bgem3",
        dimension_index_path="data/indices/medcorp/indexes",
        device="cuda",
        load_reranker=False,  # Skip reranker for speed
    )

    print(f"  Index: {retriever.get_stats()['index_size']:,} vectors")
    print(f"  Dimensions: {retriever.get_stats()['num_dimensions']}")

    # Results storage
    stage3_results = []
    stage4_results = []

    print("\n" + "=" * 80)
    print("Running tests...")
    print("=" * 80)

    for sample in TEST_SAMPLES:
        print(f"\n--- Test {sample['id']}: {sample['description']} ---")
        print(f"Q: {sample['question'][:70]}...")

        # Stage 3: Without dimensional filtering
        results_no_filter = retriever.retrieve(
            sample["question"],
            k=10,
            dimension_filters=None,
        )
        eval_no_filter = evaluate_results(results_no_filter, sample["answer_keywords"])

        # Stage 4: With dimensional filtering
        results_with_filter = retriever.retrieve(
            sample["question"],
            k=10,
            dimension_filters=sample["dimension_filters"],
        )
        eval_with_filter = evaluate_results(results_with_filter, sample["answer_keywords"])

        # Print comparison
        filter_desc = ", ".join([f"{f['dimension']}={f['values']}" for f in sample["dimension_filters"]])
        print(f"  Filter: {filter_desc}")
        print(f"  Stage 3 (no filter): {len(results_no_filter)} results, "
              f"{eval_no_filter['matches']}/10 relevant, MRR={eval_no_filter['mrr']:.2f}")
        print(f"  Stage 4 (filtered):  {len(results_with_filter)} results, "
              f"{eval_with_filter['matches']}/{len(results_with_filter)} relevant, MRR={eval_with_filter['mrr']:.2f}")

        # Show top result comparison
        if results_no_filter:
            print(f"  Top (no filter):  {results_no_filter[0].title[:40]}... (score: {results_no_filter[0].hybrid_score:.3f})")
        if results_with_filter:
            print(f"  Top (filtered):   {results_with_filter[0].title[:40]}... (score: {results_with_filter[0].hybrid_score:.3f})")

        stage3_results.append(eval_no_filter)
        stage4_results.append(eval_with_filter)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Calculate aggregates
    s3_mrr = sum(r["mrr"] for r in stage3_results) / len(stage3_results)
    s4_mrr = sum(r["mrr"] for r in stage4_results) / len(stage4_results)
    s3_relevant = sum(r["matches"] for r in stage3_results)
    s4_relevant = sum(r["matches"] for r in stage4_results)
    s3_top_hits = sum(r["top_match"] for r in stage3_results)
    s4_top_hits = sum(r["top_match"] for r in stage4_results)

    print(f"\n{'Metric':<30} {'Stage 3 (baseline)':<20} {'Stage 4 (filtered)':<20}")
    print("-" * 70)
    print(f"{'Mean Reciprocal Rank (MRR)':<30} {s3_mrr:<20.3f} {s4_mrr:<20.3f}")
    print(f"{'Total Relevant Results':<30} {s3_relevant:<20} {s4_relevant:<20}")
    print(f"{'Top-1 Keyword Hits':<30} {s3_top_hits:<20} {s4_top_hits:<20}")

    # Improvement analysis
    print("\n--- Per-Query Analysis ---")
    improved = 0
    degraded = 0
    same = 0

    for i, (s3, s4) in enumerate(zip(stage3_results, stage4_results)):
        sample = TEST_SAMPLES[i]
        if s4["mrr"] > s3["mrr"]:
            improved += 1
            status = "✓ IMPROVED"
        elif s4["mrr"] < s3["mrr"]:
            degraded += 1
            status = "✗ DEGRADED"
        else:
            same += 1
            status = "= SAME"
        print(f"  {i+1}. {sample['description'][:35]:<35} {status} (MRR: {s3['mrr']:.2f} → {s4['mrr']:.2f})")

    print(f"\nOverall: {improved} improved, {same} same, {degraded} degraded")

    return 0


if __name__ == "__main__":
    sys.exit(main())
