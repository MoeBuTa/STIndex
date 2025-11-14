#!/usr/bin/env python3
"""
Compare ablation study results.

Computes automatic metrics (no ground truth needed):
1. Extraction Volume (entity counts)
2. Geocoding Quality (success rate, failed count)
3. Cross-Chunk Consistency (entity coherence)
4. Cluster Formation (if analysis data available)
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


# Add STIndex to path
sys.path.insert(0, str(Path(__file__).parent.parent))


OUTPUT_BASE = Path("data/output/ablation_study")


def fuzzy_text_match(text1: str, text2: str, threshold: float = 0.75) -> bool:
    """Check if two texts are similar using fuzzy matching."""
    if not text1 or not text2:
        return False
    ratio = SequenceMatcher(None, text1.lower().strip(), text2.lower().strip()).ratio()
    return ratio >= threshold


def compute_extraction_stats(results: List[Dict]) -> Dict[str, Any]:
    """
    Compute extraction volume and geocoding statistics.

    Returns:
        Dict with extraction counts and geocoding metrics
    """
    stats = {
        # Extraction volume
        "total_chunks": len(results),
        "successful_chunks": 0,
        "failed_chunks": 0,
        # Entity counts by dimension
        "total_temporal": 0,
        "total_spatial": 0,
        "total_disease": 0,
        "total_event_type": 0,
        "total_venue_type": 0,
        # Geocoding quality
        "geocoded_count": 0,
        "geocoding_failed": 0,
        "geocoding_success_rate": 0.0,
        # Unique entities
        "unique_locations": set(),
        "unique_dates": set(),
        # Normalization
        "temporal_normalized": 0,
        "temporal_normalization_rate": 0.0,
    }

    for result in results:
        extraction = result.get("extraction", {})

        if extraction.get("success"):
            stats["successful_chunks"] += 1

            # Count temporal entities
            temporal = extraction.get("temporal_entities", [])
            stats["total_temporal"] += len(temporal)
            for entity in temporal:
                if entity.get("normalized"):
                    stats["temporal_normalized"] += 1
                    stats["unique_dates"].add(entity["normalized"])

            # Count spatial entities and geocoding
            spatial = extraction.get("spatial_entities", [])
            stats["total_spatial"] += len(spatial)
            for entity in spatial:
                if entity.get("latitude") and entity.get("longitude"):
                    stats["geocoded_count"] += 1
                    location_name = entity.get("name") or entity.get("text", "")
                    if location_name:
                        stats["unique_locations"].add(location_name)
                else:
                    stats["geocoding_failed"] += 1

            # Count custom dimensions
            custom_dims = extraction.get("custom_dimensions", {})
            stats["total_disease"] += len(custom_dims.get("disease", []))
            stats["total_event_type"] += len(custom_dims.get("event_type", []))
            stats["total_venue_type"] += len(custom_dims.get("venue_type", []))
        else:
            stats["failed_chunks"] += 1

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
    """
    Compute cross-chunk entity consistency.

    Groups similar entities mentioned across multiple chunks and checks
    if they're geocoded to consistent locations.

    Returns:
        (consistency_score, detailed_stats)
    """
    # Group results by document
    doc_groups = defaultdict(list)
    for result in results:
        if not result.get("extraction", {}).get("success"):
            continue
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
            spatial_entities = chunk_result.get("extraction", {}).get("spatial_entities", [])
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

                    # Consistent if all within 1km (very strict)
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


def load_latest_results(condition_dir: str) -> List[Dict]:
    """Load the latest extraction results for a condition."""
    results_dir = OUTPUT_BASE / condition_dir

    if not results_dir.exists():
        logger.warning(f"Results directory not found: {results_dir}")
        return []

    # Find latest results file
    result_files = list(results_dir.glob("extraction_results_*.json"))

    if not result_files:
        logger.warning(f"No results files found in: {results_dir}")
        return []

    # Get most recent file
    latest_file = max(result_files, key=lambda p: p.stat().st_mtime)
    logger.info(f"Loading: {latest_file.name}")

    with open(latest_file, "r") as f:
        return json.load(f)


def compare_conditions():
    """Compare all ablation conditions."""
    conditions = [
        {"name": "Baseline (No Context)", "dir": "openai_baseline", "short": "baseline"},
        {"name": "Context-Aware", "dir": "openai_context", "short": "context"},
    ]

    comparison_data = []

    print("=" * 100)
    print("Ablation Study Comparison")
    print("=" * 100)

    for condition in conditions:
        print(f"\n{condition['name']}:")
        print("-" * 100)

        # Load results
        results = load_latest_results(condition["dir"])

        if not results:
            print(f"  ⚠️  No results found for {condition['name']}")
            continue

        # Compute metrics
        logger.info("  Computing extraction statistics...")
        extraction_stats = compute_extraction_stats(results)

        logger.info("  Computing cross-chunk consistency...")
        consistency_score, consistency_details = compute_cross_chunk_consistency(results)

        # Print summary
        print(f"\n  📊 Extraction Volume:")
        print(f"    - Chunks processed: {extraction_stats['successful_chunks']}/{extraction_stats['total_chunks']}")
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

        # Store for comparison table
        comparison_data.append({
            "Condition": condition["name"],
            **extraction_stats,
            "cross_chunk_consistency": consistency_score,
            **{f"consistency_{k}": v for k, v in consistency_details.items()}
        })

    # Create comparison DataFrame
    df = pd.DataFrame(comparison_data)

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
                # Percentage metrics
                delta = context_val - baseline_val
                print(f"  {metric_name:.<30} {baseline_val:.1%} → {context_val:.1%} (Δ{delta:+.1%})")
            else:
                # Count metrics
                delta = context_val - baseline_val
                pct_change = ((context_val - baseline_val) / baseline_val * 100) if baseline_val > 0 else 0
                print(f"  {metric_name:.<30} {baseline_val} → {context_val} (Δ{delta:+d}, {pct_change:+.1f}%)")

    # Save comparison table
    output_csv = OUTPUT_BASE / "comparison_results.csv"
    df.to_csv(output_csv, index=False)

    print("\n" + "=" * 100)
    print(f"✓ Comparison saved to: {output_csv}")
    print("=" * 100)

    return df


def main():
    """Main comparison function."""
    try:
        df = compare_conditions()
        print(f"\n📊 Full comparison table:")
        print(df.to_string(index=False))
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
