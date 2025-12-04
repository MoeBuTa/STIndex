"""
Test script to verify schema merge works with hierarchical entities.
"""

import json
from pathlib import Path
from loguru import logger

from stindex.schema_discovery.schema_merger import SchemaMerger

# Paths
SCHEMA_DIR = Path("data/schema_discovery_full")
GLOBAL_DIMS_FILE = SCHEMA_DIR / "global_dimensions.json"
OUTPUT_FILE = SCHEMA_DIR / "final_schema_test.json"

logger.info("=" * 80)
logger.info("TESTING SCHEMA MERGE WITH HIERARCHICAL ENTITIES")
logger.info("=" * 80)

# Load global dimensions
logger.info("\n1. Loading global dimensions...")
with open(GLOBAL_DIMS_FILE) as f:
    global_dimensions = json.load(f)

logger.info(f"   ✓ Loaded {len(global_dimensions)} dimensions:")
for dim_name in sorted(global_dimensions.keys()):
    hierarchy = ' → '.join(global_dimensions[dim_name].get('hierarchy', []))
    logger.info(f"     • {dim_name}: {hierarchy}")

# Load cluster results
logger.info("\n2. Loading cluster entity files...")
cluster_results = []

for cluster_id in range(10):
    cluster_file = SCHEMA_DIR / f"cluster_{cluster_id}_entities.json"
    if not cluster_file.exists():
        logger.warning(f"   ⚠ Cluster {cluster_id} file not found")
        continue

    with open(cluster_file) as f:
        result = json.load(f)
        cluster_results.append(result)

    logger.debug(f"   ✓ Loaded cluster {cluster_id} ({result.get('n_questions', 0)} questions)")

logger.info(f"   ✓ Loaded {len(cluster_results)} cluster files")

# Test merge
logger.info("\n3. Testing schema merge...")
merger = SchemaMerger(similarity_threshold=0.85)

try:
    final_schema = merger.merge_clusters(cluster_results, global_dimensions)

    logger.info(f"\n✓ Merge successful!")
    logger.info(f"\nMerged {len(final_schema)} dimensions:")
    for dim_name, dim_info in sorted(final_schema.items()):
        count = dim_info.get('count', 0)
        logger.info(f"  • {dim_name}: {count} unique entities")

    # Save result
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(final_schema, f, indent=2)

    logger.info(f"\n✓ Saved to: {OUTPUT_FILE}")

    # Show example entities from merged schema
    logger.info("\n" + "=" * 80)
    logger.info("EXAMPLE MERGED ENTITIES (WITH HIERARCHY)")
    logger.info("=" * 80)

    for dim_name, dim_info in sorted(final_schema.items()):
        entities = dim_info.get('entities', [])
        if not entities:
            continue

        logger.info(f"\n{dim_name} ({len(entities)} total):")

        # Show first 3 entities
        for i, entity in enumerate(entities[:3], 1):
            if isinstance(entity, dict):
                text = entity.get('text', 'N/A')
                logger.info(f"  [{i}] {text}")
                # Show hierarchy fields
                for key, value in entity.items():
                    if key != 'text':
                        logger.info(f"      • {key}: {value}")
            else:
                # Old format (string)
                logger.info(f"  [{i}] {entity} (no hierarchy)")

        if len(entities) > 3:
            logger.info(f"  ... and {len(entities) - 3} more")

        break  # Just show one dimension as example

    logger.info("\n✅ Merge test complete!")

except Exception as e:
    logger.error(f"\n✗ Merge failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
