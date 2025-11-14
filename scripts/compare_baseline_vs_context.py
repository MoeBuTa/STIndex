#!/usr/bin/env python3
"""
Compare baseline (new) vs context-aware (existing case study results).

Baseline: data/output/ablation_study/openai_baseline/extraction_results_*.json
Context-Aware: case_studies/public_health/data/analysis/gpt-4o-mini/extraction_results.json
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any, Tuple
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
from geopy.distance import geodesic
from loguru import logger


# Paths
BASELINE_DIR = Path("data/output/ablation_study/openai_baseline")
CONTEXT_AWARE_FILE = Path("case_studies/public_health/data/analysis/gpt-4o-mini/extraction_results.json")


def normalize_results(results: List[Dict]) -> List[Dict]:
    """
    Normalize results to standard format (supports both old and new formats).

    Old format: extraction.temporal_entities, extraction.spatial_entities, extraction.custom_dimensions
    New format: extraction.entities.{dimension}
    """
    normalized = []
    for result in results:
        extraction = result.get("extraction", {})
        if not extraction.get("success"):
            continue

        # Check if using new format (extraction.entities.*)
        entities_dict = extraction.get("entities", {})
        if entities_dict:
            # New format (case study and updated baseline)
            normalized.append({
                "chunk_id": result.get("chunk_id"),
                "document_id": result.get("document_id"),
                "temporal": entities_dict.get("temporal", []),
                "spatial": entities_dict.get("spatial", []),
                "disease": entities_dict.get("disease", []),
                "event_type": entities_dict.get("event_type", []),
                "venue_type": entities_dict.get("venue_type", []),
            })
        else:
            # Old format (legacy baseline)
            custom_dims = extraction.get("custom_dimensions", {})
            normalized.append({
                "chunk_id": result.get("chunk_id"),
                "document_id": result.get("document_id"),
                "temporal": extraction.get("temporal_entities", []),
                "spatial": extraction.get("spatial_entities", []),
                "disease": custom_dims.get("disease", []),
                "event_type": custom_dims.get("event_type", []),
                "venue_type": custom_dims.get("venue_type", []),
            })
    return normalized


def fuzzy_text_match(text1: str, text2: str, threshold: float = 0.75) -> bool:
    """Check if two texts are similar using fuzzy matching."""
    if not text1 or not text2:
        return False
    ratio = SequenceMatcher(None, text1.lower().strip(), text2.lower().strip()).ratio()
    return ratio >= threshold


def compute_extraction_stats(results: List[Dict]) -> Dict[str, Any]:
    """Compute extraction volume and geocoding statistics."""
    stats = {
        "total_chunks": len(results),
        "total_temporal": 0,
        "total_spatial": 0,
        "total_disease": 0,
        "total_event_type": 0,
        "total_venue_type": 0,
        "geocoded_count": 0,
        "geocoding_failed": 0,
        "geocoding_success_rate": 0.0,
        "unique_locations": set(),
        "unique_dates": set(),
        "temporal_normalized": 0,
        "temporal_normalization_rate": 0.0,
    }

    for result in results:
        # Count temporal entities
        temporal = result.get("temporal", [])
        stats["total_temporal"] += len(temporal)
        for entity in temporal:
            normalized = entity.get("normalized") or entity.get("normalization_value")
            if normalized:
                stats["temporal_normalized"] += 1
                stats["unique_dates"].add(normalized)

        # Count spatial entities and geocoding
        spatial = result.get("spatial", [])
        stats["total_spatial"] += len(spatial)
        for entity in spatial:
            lat = entity.get("latitude")
            lon = entity.get("longitude")
            if lat and lon:
                stats["geocoded_count"] += 1
                location_name = entity.get("name") or entity.get("text", "")
                if location_name:
                    stats["unique_locations"].add(location_name)
            else:
                stats["geocoding_failed"] += 1

        # Count custom dimensions
        stats["total_disease"] += len(result.get("disease", []))
        stats["total_event_type"] += len(result.get("event_type", []))
        stats["total_venue_type"] += len(result.get("venue_type", []))

    # Calculate rates
    if stats["total_spatial"] > 0:
        stats["geocoding_success_rate"] = stats["geocoded_count"] / stats["total_spatial"]

    if stats["total_temporal"] > 0:
        stats["temporal_normalization_rate"] = stats["temporal_normalized"] / stats["total_temporal"]

    # Convert sets to counts
    stats["unique_locations_count"] = len(stats["unique_locations"])
    stats["unique_dates_count"] = len(stats["unique_dates"])
    del stats["unique_locations"]
    del stats["unique_dates"]

    return stats


def compute_cross_chunk_consistency(results: List[Dict]) -> Tuple[float, Dict]:
    """Compute cross-chunk entity consistency."""
    # Group results by document
    doc_groups = defaultdict(list)
    for result in results:
        doc_id = result.get("document_id", "unknown")
        doc_groups[doc_id].append(result)

    multi_chunk_docs = {k: v for k, v in doc_groups.items() if len(v) > 1}
    logger.info(f"  Multi-chunk documents: {len(multi_chunk_docs)}/{len(doc_groups)}")

    consistency_scores = []
    entity_groups_analyzed = 0
    total_multi_mention_entities = 0

    for doc_id, chunks in multi_chunk_docs.items():
        # Collect all spatial entities from all chunks
        all_spatial = []
        for chunk_result in chunks:
            spatial_entities = chunk_result.get("spatial", [])
            all_spatial.extend(spatial_entities)

        if not all_spatial:
            continue

        # Group by fuzzy text matching
        entity_groups = []
        for entity in all_spatial:
            entity_text = entity.get("text", "") or entity.get("name", "")
            if not entity_text:
                continue

            # Find existing group with similar text
            matched_group = None
            for group in entity_groups:
                group_text = group[0].get("text", "") or group[0].get("name", "")
                if fuzzy_text_match(entity_text, group_text):
                    matched_group = group
                    break

            if matched_group:
                matched_group.append(entity)
            else:
                entity_groups.append([entity])

        # Check consistency within each group
        for group in entity_groups:
            if len(group) > 1:  # Multi-mention entity
                total_multi_mention_entities += 1
                entity_groups_analyzed += 1

                # Get all geocoded coordinates
                coords = []
                for e in group:
                    lat = e.get("latitude")
                    lon = e.get("longitude")
                    if lat is not None and lon is not None:
                        coords.append((lat, lon))

                if len(coords) > 1:
                    # Compute max pairwise distance
                    max_dist = 0.0
                    for i, c1 in enumerate(coords):
                        for c2 in coords[i+1:]:
                            try:
                                dist = geodesic(c1, c2).km
                                max_dist = max(max_dist, dist)
                            except:
                                pass

                    # Consistent if all within 1km
                    is_consistent = (max_dist < 1.0)
                    consistency_scores.append(1.0 if is_consistent else 0.0)

    avg_consistency = float(np.mean(consistency_scores)) if consistency_scores else 0.0

    detailed_stats = {
        "multi_chunk_documents": len(multi_chunk_docs),
        "entity_groups_analyzed": entity_groups_analyzed,
        "multi_mention_entities": total_multi_mention_entities,
        "consistency_score": avg_consistency,
        "num_consistency_checks": len(consistency_scores),
    }

    return avg_consistency, detailed_stats


def main():
    """Compare baseline vs context-aware."""
    print("=" * 100)
    print("Baseline vs Context-Aware Comparison")
    print("=" * 100)

    # Load baseline results (latest file)
    baseline_files = list(BASELINE_DIR.glob("extraction_results_*.json"))
    if not baseline_files:
        logger.error(f"No baseline results found in {BASELINE_DIR}")
        sys.exit(1)

    baseline_file = max(baseline_files, key=lambda p: p.stat().st_mtime)
    logger.info(f"Loading baseline: {baseline_file.name}")

    with open(baseline_file, "r") as f:
        baseline_raw = json.load(f)
    baseline_results = normalize_results(baseline_raw)

    # Load context-aware results
    if not CONTEXT_AWARE_FILE.exists():
        logger.error(f"Context-aware results not found: {CONTEXT_AWARE_FILE}")
        sys.exit(1)

    logger.info(f"Loading context-aware: {CONTEXT_AWARE_FILE.name}")

    with open(CONTEXT_AWARE_FILE, "r") as f:
        context_aware_raw = json.load(f)
    context_aware_results = normalize_results(context_aware_raw)

    # Compute metrics for both
    conditions = [
        ("Baseline (No Context)", baseline_results),
        ("Context-Aware (Existing)", context_aware_results),
    ]

    comparison_data = []

    for condition_name, results in conditions:
        print(f"\n{condition_name}:")
        print("-" * 100)

        # Compute extraction stats
        logger.info("  Computing extraction statistics...")
        extraction_stats = compute_extraction_stats(results)

        # Compute consistency
        logger.info("  Computing cross-chunk consistency...")
        consistency_score, consistency_details = compute_cross_chunk_consistency(results)

        # Print summary
        print(f"\n  📊 Extraction Volume:")
        print(f"    - Chunks processed: {extraction_stats['total_chunks']}")
        print(f"    - Temporal entities: {extraction_stats['total_temporal']}")
        print(f"    - Spatial entities: {extraction_stats['total_spatial']}")
        print(f"    - Disease entities: {extraction_stats['total_disease']}")
        print(f"    - Event types: {extraction_stats['total_event_type']}")
        print(f"    - Venue types: {extraction_stats['total_venue_type']}")

        print(f"\n  📍 Geocoding Quality:")
        print(f"    - Geocoded: {extraction_stats['geocoded_count']}/{extraction_stats['total_spatial']}")
        print(f"    - Success rate: {extraction_stats['geocoding_success_rate']:.1%}")
        print(f"    - Failed: {extraction_stats['geocoding_failed']}")

        print(f"\n  🔗 Cross-Chunk Consistency:")
        print(f"    - Multi-chunk docs: {consistency_details['multi_chunk_documents']}")
        print(f"    - Multi-mention entities: {consistency_details['multi_mention_entities']}")
        print(f"    - Consistency score: {consistency_score:.1%}")

        print(f"\n  📅 Temporal Normalization:")
        print(f"    - Normalized: {extraction_stats['temporal_normalized']}/{extraction_stats['total_temporal']}")
        print(f"    - Normalization rate: {extraction_stats['temporal_normalization_rate']:.1%}")

        print(f"\n  🔢 Unique Entities:")
        print(f"    - Unique locations: {extraction_stats['unique_locations_count']}")
        print(f"    - Unique dates: {extraction_stats['unique_dates_count']}")

        # Store for comparison
        comparison_data.append({
            "Condition": condition_name,
            **extraction_stats,
            "cross_chunk_consistency": consistency_score,
        })

    # Calculate improvements
    if len(comparison_data) == 2:
        baseline = comparison_data[0]
        context = comparison_data[1]

        print("\n" + "=" * 100)
        print("Improvement Analysis (Context - Baseline)")
        print("=" * 100)

        metrics = [
            ("Temporal entities", "total_temporal"),
            ("Spatial entities", "total_spatial"),
            ("Disease entities", "total_disease"),
            ("Event types", "total_event_type"),
            ("Venue types", "total_venue_type"),
            ("Geocoding success", "geocoding_success_rate"),
            ("Consistency score", "cross_chunk_consistency"),
            ("Temporal normalization", "temporal_normalization_rate"),
            ("Unique locations", "unique_locations_count"),
            ("Unique dates", "unique_dates_count"),
        ]

        for metric_name, key in metrics:
            baseline_val = baseline[key]
            context_val = context[key]

            if isinstance(baseline_val, float):
                delta = context_val - baseline_val
                print(f"  {metric_name:.<30} {baseline_val:.1%} → {context_val:.1%} (Δ{delta:+.1%})")
            else:
                delta = context_val - baseline_val
                pct_change = ((context_val - baseline_val) / baseline_val * 100) if baseline_val > 0 else 0
                print(f"  {metric_name:.<30} {baseline_val} → {context_val} (Δ{delta:+d}, {pct_change:+.1f}%)")

    # Save comparison
    df = pd.DataFrame(comparison_data)
    output_csv = Path("data/output/ablation_study/baseline_vs_context_comparison.csv")
    df.to_csv(output_csv, index=False)

    print("\n" + "=" * 100)
    print(f"✓ Comparison saved to: {output_csv}")
    print("=" * 100)


if __name__ == "__main__":
    main()
