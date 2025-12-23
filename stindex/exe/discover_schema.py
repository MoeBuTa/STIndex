#!/usr/bin/env python3
"""
Schema discovery CLI - wraps SchemaDiscoveryPipeline for CLI usage.

This module provides a CLI interface to the schema discovery pipeline,
allowing discovery of dimensional schemas from Q&A datasets.

Usage:
    python -m stindex.exe.discover_schema \
        --config cfg/discovery/textbook_schema.yml

    # Test mode (3 clusters for quick validation)
    python -m stindex.exe.discover_schema \
        --config cfg/discovery/textbook_schema.yml \
        --test
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from stindex.pipeline.discovery_pipeline import SchemaDiscoveryPipeline
from stindex.utils.config import load_config_from_file


def main():
    parser = argparse.ArgumentParser(
        description="Discover dimensional schema from Q&A dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full discovery with 10 clusters
  python -m stindex.exe.discover_schema --config cfg/discovery/textbook_schema.yml

  # Test mode with 3 clusters (quick validation)
  python -m stindex.exe.discover_schema --config cfg/discovery/textbook_schema.yml --test

Config file format:
  input:
    questions_path: "data/questions/mirage_textbook_questions.jsonl"

  discovery:
    n_clusters: 10
    enable_parallel: true
    max_workers: 5
    reuse_clusters: true

  llm:
    llm_provider: "hf"
    model_name: null

  output:
    output_dir: "data/schema_discovery_mirage_textbook"
"""
    )
    parser.add_argument("--config", required=True, help="Path to config YAML file")
    parser.add_argument("--test", action="store_true",
                       help="Test mode: 3 clusters for quick validation")
    args = parser.parse_args()

    # Load config
    print(f"Loading config from: {args.config}")
    config = load_config_from_file(args.config)

    questions_file = config['input']['questions_path']
    output_dir = config['output']['output_dir']
    discovery_config = config.get('discovery', {})
    llm_config = config.get('llm', {})
    entity_extraction_config = config.get('entity_extraction', {})

    # Validate questions file exists
    if not Path(questions_file).exists():
        print(f"✗ Questions file not found: {questions_file}")
        print(f"  Please run question filtering first:")
        print(f"  python -m rag.preprocess.corpus.filter_questions --config cfg/corpus/textbook_question_filtering.yml")
        return 1

    # Test vs full mode
    if args.test:
        n_clusters = 3
        test_clusters = [0, 1, 2]
        print("🧪 Test mode: 3 clusters with adaptive batching")
    else:
        n_clusters = discovery_config.get('n_clusters', 10)
        test_clusters = None
        print(f"🚀 Full mode: {n_clusters} clusters with adaptive batching")

    print(f"\nConfiguration:")
    print(f"  Questions file: {questions_file}")
    print(f"  Output directory: {output_dir}")
    print(f"  LLM provider: {llm_config.get('llm_provider', 'openai')}")
    print(f"  Model: {llm_config.get('model_name', 'default')}")
    print(f"  Parallel workers: {discovery_config.get('max_workers', 5)}")
    print(f"  Reuse clusters: {discovery_config.get('reuse_clusters', True)}")

    # Initialize pipeline
    print(f"\n⚙️  Initializing schema discovery pipeline...")
    batch_size = entity_extraction_config.get('batch_size', 50)
    print(f"  Batch size: {batch_size}")
    pipeline = SchemaDiscoveryPipeline(
        llm_config=llm_config,
        n_clusters=n_clusters,
        batch_size=batch_size,
        enable_parallel=discovery_config.get('enable_parallel', True),
        max_workers=discovery_config.get('max_workers', 5),
        test_clusters=test_clusters
    )

    # Run discovery
    print(f"\n⚙️  Running schema discovery...")
    try:
        final_schema = pipeline.discover_schema(
            questions_file=questions_file,
            output_dir=output_dir,
            reuse_clusters=discovery_config.get('reuse_clusters', True)
        )

        print(f"\n✅ Schema discovery completed successfully!")
        print(f"   Output directory: {output_dir}")
        print(f"   Discovered {len(final_schema.dimensions)} dimensions:")

        for dim_name in sorted(final_schema.get_dimension_names()):
            dimension = final_schema.dimensions[dim_name]
            print(f"     • {dim_name}: {dimension.total_entity_count} entities " +
                  f"from {len(dimension.sources.cluster_ids)} clusters")

        print(f"\n📁 Schema saved to: {output_dir}/final_schema.yml")
        return 0

    except Exception as e:
        print(f"\n✗ Schema discovery failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
