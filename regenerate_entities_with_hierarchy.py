"""
Regenerate cluster entity files with hierarchical structure preserved.

This script re-runs entity extraction using the fixed ClusterEntityExtractor
that preserves hierarchy fields for each entity.
"""

import json
import pandas as pd
from pathlib import Path
from loguru import logger

from stindex.schema_discovery.cluster_entity_extractor import ClusterEntityExtractor

# Configuration
SCHEMA_DIR = Path("data/schema_discovery_full")
QUESTIONS_FILE = "data/original/mirage/train.jsonl"
BATCH_SIZE = 50
TEST_CLUSTERS = None  # Set to [0, 1] to test on just 2 clusters, or None for all

# LLM config
LLM_CONFIG = {
    "llm_provider": "openai",
    "model_name": "gpt-4o-mini"
}

def main():
    logger.info("=" * 80)
    logger.info("REGENERATING CLUSTER ENTITIES WITH HIERARCHY")
    logger.info("=" * 80)

    # Load global dimensions
    logger.info("\n1. Loading global dimensions...")
    global_dims_path = SCHEMA_DIR / "global_dimensions.json"
    with open(global_dims_path) as f:
        global_dimensions = json.load(f)

    logger.info(f"   ✓ Loaded {len(global_dimensions)} dimensions:")
    for dim_name, dim_info in sorted(global_dimensions.items()):
        hierarchy = ' → '.join(dim_info.get('hierarchy', []))
        logger.info(f"     • {dim_name}: {hierarchy}")

    # Load cluster assignments
    logger.info("\n2. Loading cluster assignments...")
    cluster_assignments_path = SCHEMA_DIR / "cluster_assignments.csv"
    cluster_assignments = pd.read_csv(cluster_assignments_path)
    logger.info(f"   ✓ Loaded {len(cluster_assignments)} question assignments")

    # Prepare cluster data
    logger.info("\n3. Preparing cluster data...")
    n_clusters = cluster_assignments['cluster_id'].max() + 1
    cluster_data = []

    for cluster_id in range(n_clusters):
        cluster_mask = cluster_assignments['cluster_id'] == cluster_id
        cluster_questions = cluster_assignments[cluster_mask]['question'].tolist()

        if not cluster_questions:
            logger.warning(f"   ⚠ No questions in Cluster {cluster_id}, skipping")
            continue

        cluster_data.append({
            'cluster_id': cluster_id,
            'questions': cluster_questions,
            'n_questions': len(cluster_questions)
        })

    # Filter if testing
    if TEST_CLUSTERS is not None:
        cluster_data = [c for c in cluster_data if c['cluster_id'] in TEST_CLUSTERS]
        logger.info(f"   Testing with {len(cluster_data)} clusters: {TEST_CLUSTERS}")
    else:
        logger.info(f"   Processing all {len(cluster_data)} clusters")

    # Re-extract entities for each cluster
    logger.info("\n4. Re-extracting entities with hierarchical structure...")
    logger.info(f"   Batch size: {BATCH_SIZE} questions per LLM call")
    logger.info(f"   Output directory: {SCHEMA_DIR}")

    results = []

    for i, cluster_info in enumerate(cluster_data, 1):
        cluster_id = cluster_info['cluster_id']
        n_questions = cluster_info['n_questions']

        logger.info(f"\n   [{i}/{len(cluster_data)}] Processing Cluster {cluster_id} ({n_questions} questions)...")

        try:
            # Create extractor with fixed code
            extractor = ClusterEntityExtractor(
                global_dimensions=global_dimensions,
                llm_config=LLM_CONFIG,
                batch_size=BATCH_SIZE,
                output_dir=str(SCHEMA_DIR)
            )

            # Extract entities
            result = extractor.extract_from_cluster(
                cluster_questions=cluster_info['questions'],
                cluster_id=cluster_id
            )

            # Save result
            output_path = SCHEMA_DIR / f"cluster_{cluster_id}_entities.json"
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2)

            logger.info(f"   ✓ Saved: {output_path}")

            # Log entity counts
            total_entities = sum(result['entity_counts'].values())
            logger.info(f"   ✓ Extracted {total_entities} unique entities:")
            for dim_name, count in sorted(result['entity_counts'].items()):
                logger.info(f"       - {dim_name}: {count}")

            results.append(result)

        except Exception as e:
            logger.error(f"   ✗ Failed to process Cluster {cluster_id}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("REGENERATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"\nSuccessfully regenerated {len(results)}/{len(cluster_data)} clusters")
    logger.info(f"Output directory: {SCHEMA_DIR}")

    # Show example entities with hierarchy
    if results:
        logger.info("\nExample entities with hierarchy fields:")
        first_result = results[0]
        for dim_name, entities in sorted(first_result['entities'].items()):
            if entities:  # Only show dimensions with entities
                logger.info(f"\n  {dim_name}:")
                # Show first entity as example
                example_entity = entities[0]
                logger.info(f"    Example: {example_entity.get('text', 'N/A')}")
                # Show hierarchy fields
                for key, value in example_entity.items():
                    if key != 'text':
                        logger.info(f"      • {key}: {value}")
                if len(entities) > 1:
                    logger.info(f"    ... and {len(entities) - 1} more entities")
                break  # Just show one dimension as example

    logger.info("\n✅ Done! Check the cluster_*_entities.json files for hierarchical structure.")

if __name__ == '__main__':
    main()
