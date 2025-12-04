"""
Test script: Regenerate just Cluster 0 to verify hierarchical structure fix.

This runs on a single cluster to quickly verify the fix before regenerating all clusters.
"""

import json
import pandas as pd
from pathlib import Path
from loguru import logger

from stindex.schema_discovery.cluster_entity_extractor import ClusterEntityExtractor

# Configuration
SCHEMA_DIR = Path("data/schema_discovery_full")
TEST_CLUSTER_ID = 0  # Test with cluster 0
BATCH_SIZE = 50

# LLM config
LLM_CONFIG = {
    "llm_provider": "openai",
    "model_name": "gpt-4o-mini"
}

def main():
    logger.info("=" * 80)
    logger.info(f"TEST: Regenerating Cluster {TEST_CLUSTER_ID} with Hierarchy")
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

    # Get questions for test cluster
    cluster_mask = cluster_assignments['cluster_id'] == TEST_CLUSTER_ID
    cluster_questions = cluster_assignments[cluster_mask]['question'].tolist()

    logger.info(f"   ✓ Cluster {TEST_CLUSTER_ID} has {len(cluster_questions)} questions")

    # Extract entities
    logger.info(f"\n3. Extracting entities from Cluster {TEST_CLUSTER_ID}...")
    logger.info(f"   Batch size: {BATCH_SIZE} questions per LLM call")

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
            cluster_questions=cluster_questions,
            cluster_id=TEST_CLUSTER_ID
        )

        # Save result
        output_path = SCHEMA_DIR / f"cluster_{TEST_CLUSTER_ID}_entities_NEW.json"
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)

        logger.info(f"\n   ✓ Saved to: {output_path}")

        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("EXTRACTION COMPLETE")
        logger.info("=" * 80)

        total_entities = sum(result['entity_counts'].values())
        logger.info(f"\nExtracted {total_entities} unique entities:")
        for dim_name, count in sorted(result['entity_counts'].items()):
            logger.info(f"  • {dim_name}: {count}")

        # Show example entities with hierarchy
        logger.info("\n" + "=" * 80)
        logger.info("EXAMPLE ENTITIES WITH HIERARCHY FIELDS")
        logger.info("=" * 80)

        for dim_name, entities in sorted(result['entities'].items()):
            if not entities:
                continue

            logger.info(f"\n{dim_name} ({len(entities)} entities):")

            # Show first 3 entities
            for i, entity in enumerate(entities[:3], 1):
                logger.info(f"\n  [{i}] {entity.get('text', 'N/A')}")

                # Show all hierarchy fields
                hierarchy_fields = [k for k in entity.keys() if k != 'text']
                if hierarchy_fields:
                    for field in hierarchy_fields:
                        value = entity.get(field, 'N/A')
                        logger.info(f"      • {field}: {value}")
                else:
                    logger.warning(f"      ⚠ No hierarchy fields! (only 'text')")

            if len(entities) > 3:
                logger.info(f"\n  ... and {len(entities) - 3} more entities")

        # Verification
        logger.info("\n" + "=" * 80)
        logger.info("VERIFICATION")
        logger.info("=" * 80)

        all_have_hierarchy = True
        missing_hierarchy = []

        for dim_name, entities in result['entities'].items():
            expected_hierarchy = global_dimensions[dim_name]['hierarchy']

            for entity in entities:
                # Check if entity has at least one hierarchy field (besides 'text')
                hierarchy_fields = [k for k in entity.keys() if k != 'text']

                if not hierarchy_fields:
                    all_have_hierarchy = False
                    missing_hierarchy.append({
                        'dimension': dim_name,
                        'entity': entity.get('text', 'N/A')
                    })

        if all_have_hierarchy and result['entities']:
            logger.info("✅ SUCCESS: All entities have hierarchical fields!")
            logger.info("\nYou can now run the full regeneration:")
            logger.info("  python regenerate_entities_with_hierarchy.py")
        elif not result['entities']:
            logger.warning("⚠️  No entities extracted (check LLM response)")
        else:
            logger.error(f"❌ FAILED: {len(missing_hierarchy)} entities missing hierarchy fields")
            for item in missing_hierarchy[:5]:
                logger.error(f"  • {item['dimension']}: '{item['entity']}'")

    except Exception as e:
        logger.error(f"\n✗ Extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0

if __name__ == '__main__':
    import sys
    sys.exit(main())
