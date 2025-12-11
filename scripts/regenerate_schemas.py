#!/usr/bin/env python3
"""
Schema Regeneration Script for v2.0

Regenerates discovered schemas using the new cluster-level architecture.
Replaces old global-seeded schemas with Pydantic-based cluster-level schemas.

Usage:
    python scripts/regenerate_schemas.py --dataset mirage --test
    python scripts/regenerate_schemas.py --dataset mirage --full
    python scripts/regenerate_schemas.py --all --full
"""

import argparse
import sys
from pathlib import Path
import shutil
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from stindex.pipeline.discovery_pipeline import SchemaDiscoveryPipeline


DATASETS = {
    'mirage': {
        'questions_file': 'data/original/mirage/train.jsonl',
        'output_dir': 'data/schema_discovery_mirage_v2',
        'description': 'MIRAGE medical QA benchmark (6,545 questions)'
    },
    # Add more datasets as needed
    # 'hotpotqa': {
    #     'questions_file': 'data/original/hotpotqa/train.jsonl',
    #     'output_dir': 'data/schema_discovery_hotpotqa_v2',
    #     'description': 'HotpotQA multi-hop QA (90,425 questions)'
    # },
}


def backup_old_schema(output_dir: Path):
    """Backup old schema if it exists."""
    if output_dir.exists():
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_dir = output_dir.parent / f"{output_dir.name}_backup_{timestamp}"

        print(f"  📦 Backing up old schema to: {backup_dir}")
        shutil.move(str(output_dir), str(backup_dir))
        print(f"  ✓ Backup complete")


def regenerate_schema(
    dataset_name: str,
    questions_file: str,
    output_dir: str,
    test_mode: bool = False,
    reuse_clusters: bool = True,
    llm_provider: str = 'openai',
    model: str = 'gpt-4o-mini',
    max_workers: int = 5
):
    """
    Regenerate schema for a dataset.

    Args:
        dataset_name: Dataset identifier
        questions_file: Path to questions JSONL file
        output_dir: Output directory for results
        test_mode: If True, run with fewer clusters for testing
        reuse_clusters: Reuse existing cluster assignments
        llm_provider: LLM provider (openai, anthropic, hf)
        model: Model name
        max_workers: Number of parallel workers
    """
    questions_path = Path(questions_file)
    output_path = Path(output_dir)

    # Validate input file
    if not questions_path.exists():
        print(f"  ✗ Questions file not found: {questions_file}")
        print(f"    Please ensure the dataset is downloaded")
        return False

    # Backup old schema
    backup_old_schema(output_path)

    # Configure pipeline
    if test_mode:
        n_clusters = 3
        n_samples = 10
        test_clusters = [0, 1, 2]
        print(f"  🧪 Test mode: 3 clusters, 10 samples per cluster")
    else:
        n_clusters = 10
        n_samples = 20
        test_clusters = None
        print(f"  🚀 Full mode: 10 clusters, 20 samples per cluster")

    # Initialize pipeline
    llm_config = {
        'llm_provider': llm_provider,
        'model_name': model
    }

    pipeline = SchemaDiscoveryPipeline(
        llm_config=llm_config,
        n_clusters=n_clusters,
        n_samples_for_discovery=n_samples,
        enable_parallel=True,
        max_workers=max_workers,
        test_clusters=test_clusters
    )

    # Run discovery
    print(f"  ⚙️  Running schema discovery...")
    print(f"     Provider: {llm_provider}, Model: {model}")
    print(f"     Parallel workers: {max_workers}")

    try:
        final_schema = pipeline.discover_schema(
            questions_file=str(questions_path),
            output_dir=str(output_path),
            reuse_clusters=reuse_clusters
        )

        print(f"\n  ✅ Schema regenerated successfully!")
        print(f"     Output directory: {output_path}")
        print(f"     Discovered {len(final_schema.dimensions)} dimensions:")

        for dim_name in sorted(final_schema.get_dimension_names()):
            dimension = final_schema.dimensions[dim_name]
            print(f"       • {dim_name}: {dimension.total_entity_count} entities from {len(dimension.sources.cluster_ids)} clusters")

        return True

    except Exception as e:
        print(f"\n  ✗ Schema regeneration failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Regenerate discovered schemas with v2.0 architecture',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test MIRAGE with 3 clusters (quick)
  python scripts/regenerate_schemas.py --dataset mirage --test

  # Full MIRAGE run with 10 clusters
  python scripts/regenerate_schemas.py --dataset mirage --full

  # Regenerate all datasets
  python scripts/regenerate_schemas.py --all --full

  # Use Anthropic instead of OpenAI
  python scripts/regenerate_schemas.py --dataset mirage --full --llm-provider anthropic --model claude-3-5-sonnet-20241022

Available datasets:
""" + "\n".join([f"  - {name}: {info['description']}" for name, info in DATASETS.items()])
    )

    parser.add_argument(
        '--dataset',
        choices=list(DATASETS.keys()),
        help='Dataset to regenerate schema for'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Regenerate schemas for all datasets'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test mode: 3 clusters, 10 samples (fast)'
    )
    parser.add_argument(
        '--full',
        action='store_true',
        help='Full mode: 10 clusters, 20 samples (complete)'
    )
    parser.add_argument(
        '--reuse-clusters',
        action='store_true',
        default=True,
        help='Reuse existing cluster assignments (default: True)'
    )
    parser.add_argument(
        '--llm-provider',
        default='openai',
        choices=['openai', 'anthropic', 'hf'],
        help='LLM provider (default: openai)'
    )
    parser.add_argument(
        '--model',
        default='gpt-4o-mini',
        help='Model name (default: gpt-4o-mini)'
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        default=5,
        help='Number of parallel workers (default: 5)'
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.dataset and not args.all:
        parser.error("Must specify either --dataset or --all")

    if not args.test and not args.full:
        parser.error("Must specify either --test or --full")

    # Determine datasets to process
    if args.all:
        datasets_to_process = list(DATASETS.keys())
    else:
        datasets_to_process = [args.dataset]

    # Process datasets
    print("\n" + "=" * 80)
    print("SCHEMA REGENERATION (v2.0 - Cluster-Level Architecture)")
    print("=" * 80)
    print(f"\nMode: {'TEST' if args.test else 'FULL'}")
    print(f"Datasets: {', '.join(datasets_to_process)}")
    print(f"LLM: {args.llm_provider} / {args.model}")
    print()

    results = {}
    for dataset_name in datasets_to_process:
        dataset_info = DATASETS[dataset_name]

        print(f"\n{'=' * 80}")
        print(f"Dataset: {dataset_name.upper()}")
        print(f"Description: {dataset_info['description']}")
        print(f"{'=' * 80}\n")

        success = regenerate_schema(
            dataset_name=dataset_name,
            questions_file=dataset_info['questions_file'],
            output_dir=dataset_info['output_dir'],
            test_mode=args.test,
            reuse_clusters=args.reuse_clusters,
            llm_provider=args.llm_provider,
            model=args.model,
            max_workers=args.max_workers
        )

        results[dataset_name] = success

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for dataset_name, success in results.items():
        status = "✅ SUCCESS" if success else "✗ FAILED"
        print(f"  {dataset_name}: {status}")

    all_success = all(results.values())
    if all_success:
        print("\n🎉 All schemas regenerated successfully!")
        return 0
    else:
        print("\n⚠️ Some schemas failed to regenerate. Check logs above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
