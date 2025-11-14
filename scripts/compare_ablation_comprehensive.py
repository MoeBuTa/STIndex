#!/usr/bin/env python3
"""
Comprehensive Ablation Study Comparison

Compares 4 conditions in a 2x2 design:
- Model: OpenAI (GPT-4o-mini) vs Qwen3-8B
- Context: Baseline (no context) vs Context-Aware

Generates:
1. Per-condition metrics (extraction volume, geocoding quality, consistency, etc.)
2. Context-awareness impact (context - baseline) for each model
3. Model comparison (GPT-4o-mini vs Qwen3-8B) for each context setting
4. Summary CSV with all metrics
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
ABLATION_DIR = Path("data/output/ablation_study")

# All 4 conditions
CONDITIONS = {
    "openai_baseline": {
        "name": "GPT-4o-mini Baseline",
        "dir": "openai_baseline",
        "model": "GPT-4o-mini",
        "context": "No"
    },
    "openai_context": {
        "name": "GPT-4o-mini Context-Aware",
        "dir": "openai_context",
        "model": "GPT-4o-mini",
        "context": "Yes"
    },
    "qwen3_baseline": {
        "name": "Qwen3-8B Baseline",
        "dir": "qwen3_baseline",
        "model": "Qwen3-8B",
        "context": "No"
    },
    "qwen3_context": {
        "name": "Qwen3-8B Context-Aware",
        "dir": "qwen3_context",
        "model": "Qwen3-8B",
        "context": "Yes"
    }
}


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
            # New format (updated baseline and context-aware)
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
    logger.debug(f"  Multi-chunk documents: {len(multi_chunk_docs)}/{len(doc_groups)}")

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


def load_condition_results(condition_key: str) -> List[Dict]:
    """Load results for a specific condition."""
    condition_info = CONDITIONS[condition_key]
    condition_dir = ABLATION_DIR / condition_info["dir"]

    if not condition_dir.exists():
        logger.warning(f"Results not found for {condition_info['name']}: {condition_dir}")
        return []

    # Find latest results file
    result_files = list(condition_dir.glob("extraction_results_*.json"))
    if not result_files:
        logger.warning(f"No result files found in {condition_dir}")
        return []

    result_file = max(result_files, key=lambda p: p.stat().st_mtime)
    logger.info(f"Loading {condition_info['name']}: {result_file.name}")

    with open(result_file, "r") as f:
        raw_results = json.load(f)

    return normalize_results(raw_results)


def print_condition_summary(condition_name: str, stats: Dict, consistency_details: Dict):
    """Print summary for one condition."""
    print(f"\n{'='*100}")
    print(f"{condition_name}")
    print(f"{'='*100}")

    print(f"\n  📊 Extraction Volume:")
    print(f"    - Chunks processed: {stats['total_chunks']}")
    print(f"    - Temporal entities: {stats['total_temporal']}")
    print(f"    - Spatial entities: {stats['total_spatial']}")
    print(f"    - Disease entities: {stats['total_disease']}")
    print(f"    - Event types: {stats['total_event_type']}")
    print(f"    - Venue types: {stats['total_venue_type']}")

    print(f"\n  📍 Geocoding Quality:")
    print(f"    - Geocoded: {stats['geocoded_count']}/{stats['total_spatial']}")
    print(f"    - Success rate: {stats['geocoding_success_rate']:.1%}")
    print(f"    - Failed: {stats['geocoding_failed']}")

    print(f"\n  🔗 Cross-Chunk Consistency:")
    print(f"    - Multi-chunk docs: {consistency_details['multi_chunk_documents']}")
    print(f"    - Multi-mention entities: {consistency_details['multi_mention_entities']}")
    print(f"    - Consistency score: {consistency_details['consistency_score']:.1%}")

    print(f"\n  📅 Temporal Normalization:")
    print(f"    - Normalized: {stats['temporal_normalized']}/{stats['total_temporal']}")
    print(f"    - Normalization rate: {stats['temporal_normalization_rate']:.1%}")

    print(f"\n  🔢 Unique Entities:")
    print(f"    - Unique locations: {stats['unique_locations_count']}")
    print(f"    - Unique dates: {stats['unique_dates_count']}")


def print_comparison(title: str, baseline_data: Dict, treatment_data: Dict, baseline_name: str, treatment_name: str):
    """Print comparison between two conditions."""
    print(f"\n{'='*100}")
    print(f"{title}")
    print(f"{'='*100}")

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
        baseline_val = baseline_data[key]
        treatment_val = treatment_data[key]

        if isinstance(baseline_val, float):
            delta = treatment_val - baseline_val
            print(f"  {metric_name:.<30} {baseline_val:.1%} → {treatment_val:.1%} (Δ{delta:+.1%})")
        else:
            delta = treatment_val - baseline_val
            if baseline_val > 0:
                pct_change = (treatment_val - baseline_val) / baseline_val * 100
                print(f"  {metric_name:.<30} {baseline_val} → {treatment_val} (Δ{delta:+d}, {pct_change:+.1f}%)")
            else:
                print(f"  {metric_name:.<30} {baseline_val} → {treatment_val} (Δ{delta:+d})")


def main():
    """Run comprehensive comparison."""
    print("=" * 100)
    print("Ablation Study: Comprehensive Comparison")
    print("=" * 100)

    # Load all conditions
    all_results = {}
    all_stats = {}

    for condition_key in CONDITIONS.keys():
        results = load_condition_results(condition_key)
        if not results:
            continue

        all_results[condition_key] = results

        # Compute metrics
        logger.info(f"Computing metrics for {CONDITIONS[condition_key]['name']}...")
        extraction_stats = compute_extraction_stats(results)
        consistency_score, consistency_details = compute_cross_chunk_consistency(results)

        # Store combined stats
        all_stats[condition_key] = {
            **extraction_stats,
            "cross_chunk_consistency": consistency_score,
            "consistency_details": consistency_details
        }

        # Print summary
        print_condition_summary(CONDITIONS[condition_key]["name"], extraction_stats, consistency_details)

    # Context-awareness impact analysis
    if "openai_baseline" in all_stats and "openai_context" in all_stats:
        print_comparison(
            "Context-Awareness Impact: GPT-4o-mini (Context - Baseline)",
            all_stats["openai_baseline"],
            all_stats["openai_context"],
            "Baseline",
            "Context-Aware"
        )

    if "qwen3_baseline" in all_stats and "qwen3_context" in all_stats:
        print_comparison(
            "Context-Awareness Impact: Qwen3-8B (Context - Baseline)",
            all_stats["qwen3_baseline"],
            all_stats["qwen3_context"],
            "Baseline",
            "Context-Aware"
        )

    # Model comparison
    if "openai_baseline" in all_stats and "qwen3_baseline" in all_stats:
        print_comparison(
            "Model Comparison: Baseline (Qwen3-8B vs GPT-4o-mini)",
            all_stats["openai_baseline"],
            all_stats["qwen3_baseline"],
            "GPT-4o-mini",
            "Qwen3-8B"
        )

    if "openai_context" in all_stats and "qwen3_context" in all_stats:
        print_comparison(
            "Model Comparison: Context-Aware (Qwen3-8B vs GPT-4o-mini)",
            all_stats["openai_context"],
            all_stats["qwen3_context"],
            "GPT-4o-mini",
            "Qwen3-8B"
        )

    # Save to CSV
    comparison_data = []
    for condition_key, stats in all_stats.items():
        condition_info = CONDITIONS[condition_key]
        row = {
            "Condition": condition_info["name"],
            "Model": condition_info["model"],
            "Context-Aware": condition_info["context"],
            **{k: v for k, v in stats.items() if k != "consistency_details"}
        }
        comparison_data.append(row)

    if comparison_data:
        df = pd.DataFrame(comparison_data)
        output_csv = ABLATION_DIR / "comprehensive_comparison.csv"
        df.to_csv(output_csv, index=False)

        print(f"\n{'='*100}")
        print(f"✓ Comprehensive comparison saved to: {output_csv}")
        print(f"{'='*100}")


if __name__ == "__main__":
    main()
