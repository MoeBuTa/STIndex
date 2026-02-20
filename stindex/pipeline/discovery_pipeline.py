"""
End-to-end schema discovery pipeline.

Pure cluster-level discovery: each cluster discovers dimensions independently,
then schemas are merged across clusters.
"""

from pathlib import Path
from typing import Dict, List, Optional
import json
import yaml
import pandas as pd
import gc
import torch
from loguru import logger
from concurrent.futures import ThreadPoolExecutor, as_completed

from stindex.llm.base import create_client
from stindex.discovery.question_clusterer import QuestionClusterer
from stindex.discovery.cluster_schema_discoverer import ClusterSchemaDiscoverer
from stindex.discovery.schema_merger import SchemaMerger
from stindex.discovery.cot_logger import CoTLogger
from stindex.discovery.models import FinalSchema
from stindex.utils.timing import TimingStats


class SchemaDiscoveryPipeline:
    """
    Pure cluster-level schema discovery pipeline.

    Steps:
    1. Cluster questions
    2. Per-cluster discovery + extraction (each cluster discovers dimensions independently)
    3. Cross-cluster schema merging (no global baseline)
    """

    def __init__(
        self,
        llm_config: Dict,
        config: Optional[Dict] = None,
        n_clusters: int = 10,
        similarity_threshold: float = 0.60,
        batch_size: int = 50,
        enable_parallel: bool = True,
        max_workers: int = 5,
        test_clusters: Optional[List[int]] = None
    ):
        """
        Initialize schema discovery pipeline.

        Args:
            llm_config: LLM configuration (provider, model, etc.)
            config: Optional pipeline config dict (overrides other params if provided)
            n_clusters: Number of question clusters
            similarity_threshold: Fuzzy matching threshold for deduplication
            batch_size: Number of questions per LLM call during extraction (default: 50)
            enable_parallel: Enable parallel cluster processing (default: True)
            max_workers: Max parallel workers for cluster processing (default: 5)
            test_clusters: Optional list of cluster IDs to test on (e.g., [0, 1])
        """
        self.llm_config = llm_config
        self.test_clusters = test_clusters

        # Create shared LLM client for all clusters (singleton engine pattern)
        self.llm_client = create_client(llm_config)
        logger.info(f"  ✓ Initialized shared LLM client: {llm_config.get('llm_provider')}")

        # Load config if provided
        if config:
            self.n_clusters = config.get('cluster_discovery', {}).get('num_clusters', n_clusters)

            # Adaptive batching configuration
            adaptive_config = config.get('cluster_discovery', {}).get('adaptive_batching', {})
            self.adaptive_first_batch = adaptive_config.get('enabled', True)
            self.first_batch_min = adaptive_config.get('min_size', 50)
            self.first_batch_max = adaptive_config.get('max_size', 150)
            self.first_batch_ratio = adaptive_config.get('ratio', 0.10)

            # Decay thresholds configuration
            decay_config = config.get('entity_extraction', {}).get('decay_thresholds', {})
            self.decay_config = {}
            for stage in ['early', 'medium', 'late']:
                stage_config = decay_config.get(stage, {})
                self.decay_config[stage] = (
                    stage_config.get('start_batch', 1),
                    stage_config.get('end_batch', 999999),
                    stage_config.get('threshold', 0.9)
                )

            self.batch_size = config.get('entity_extraction', {}).get('batch_size', batch_size)
            self.allow_new_dimensions = config.get('entity_extraction', {}).get('allow_new_dimensions', True)
            self.max_retries = config.get('entity_extraction', {}).get('retry', {}).get('max_retries', 3)
            self.enable_parallel = config.get('parallel', {}).get('enabled', enable_parallel)
            self.max_workers = config.get('parallel', {}).get('max_workers', max_workers)
            self.similarity_threshold = config.get('schema_merging', {}).get('similarity_threshold', similarity_threshold)
        else:
            self.n_clusters = n_clusters
            self.batch_size = batch_size
            self.allow_new_dimensions = True
            self.max_retries = 3
            self.enable_parallel = enable_parallel
            self.max_workers = max_workers
            self.similarity_threshold = similarity_threshold

            # Default adaptive batching
            self.adaptive_first_batch = True
            self.first_batch_min = 50
            self.first_batch_max = 150
            self.first_batch_ratio = 0.10

            # Default decay thresholds
            self.decay_config = {
                'early': (1, 2, 0.3),
                'medium': (3, 5, 0.6),
                'late': (6, 999999, 0.9)
            }

    def discover_schema(
        self,
        questions_file: str,
        output_dir: str,
        reuse_clusters: bool = True
    ) -> FinalSchema:
        """
        Run full schema discovery pipeline (pure cluster-level).

        Args:
            questions_file: Path to questions.jsonl (e.g., data/original/mirage/train.jsonl)
            output_dir: Output directory for results
            reuse_clusters: If True, reuse existing cluster results

        Returns:
            FinalSchema Pydantic model with all discovered dimensions and entities
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Initialize timing statistics
        timing_stats = TimingStats(name="schema_discovery")

        # Create shared CoT logger for all components
        cot_logger = CoTLogger(output_dir)

        logger.info("=" * 80)
        logger.info("SCHEMA DISCOVERY PIPELINE (Cluster-Level)")
        logger.info("=" * 80)

        # Step 1: Question Clustering
        logger.info("\nStep 1: Question Clustering")
        cluster_samples_path = output_path / "cluster_samples.json"

        with timing_stats.timer("clustering"):
            if reuse_clusters and cluster_samples_path.exists():
                logger.info("  ✓ Reusing existing cluster samples")
                with open(cluster_samples_path) as f:
                    cluster_samples = json.load(f)
                    # Convert string keys to int keys if needed
                    cluster_samples = {int(k): v for k, v in cluster_samples.items()}
            else:
                logger.info(f"  Clustering questions from {questions_file}...")
                clusterer = QuestionClusterer()
                cluster_result = clusterer.cluster_questions_from_file(
                    questions_file=questions_file,
                    output_dir=str(output_path),
                    n_clusters=self.n_clusters
                )
                with open(cluster_samples_path) as f:
                    cluster_samples = json.load(f)
                    cluster_samples = {int(k): v for k, v in cluster_samples.items()}
                logger.info(f"  ✓ Clustered into {self.n_clusters} clusters")

            # Add counters for clustering
            timing_stats.add_counter("num_clusters", self.n_clusters)

        # Step 2: Unified Discovery + Extraction (no separate discovery phase)
        logger.info("\nStep 2: Unified Discovery + Extraction")
        logger.info(f"  Parallel processing: {'enabled' if self.enable_parallel else 'disabled'} (max_workers={self.max_workers})")
        logger.info(f"  Standard batch size: {self.batch_size} questions per LLM call")
        logger.info(f"  Adaptive first batch: {'enabled' if self.adaptive_first_batch else 'disabled'}")
        if self.adaptive_first_batch:
            logger.info(f"    - First batch size: {self.first_batch_ratio*100:.0f}% of cluster (min={self.first_batch_min}, max={self.first_batch_max})")
        logger.info(f"  Decay thresholds: early={self.decay_config['early'][2]:.1f}, medium={self.decay_config['medium'][2]:.1f}, late={self.decay_config['late'][2]:.1f}")
        logger.info(f"  Dimensions per cluster: Data-driven (no constraint)")

        # Load all questions
        logger.info(f"  Loading questions from {questions_file}...")
        with open(questions_file) as f:
            all_questions_data = [json.loads(line) for line in f]
            all_questions = [q['question'] for q in all_questions_data]

        logger.info(f"  Loaded {len(all_questions)} questions")
        timing_stats.add_counter("num_questions", len(all_questions))

        # Load cluster assignments
        cluster_assignments_path = output_path / "cluster_assignments.csv"
        if not cluster_assignments_path.exists():
            logger.error(f"  ✗ Cluster assignments file not found: {cluster_assignments_path}")
            raise FileNotFoundError(f"Run clustering first or check output directory")

        cluster_assignments = pd.read_csv(cluster_assignments_path)
        logger.info(f"  Loaded cluster assignments: {len(cluster_assignments)} questions")

        # Prepare cluster data
        cluster_data = []
        for cluster_id in range(self.n_clusters):
            cluster_mask = cluster_assignments['cluster_id'] == cluster_id
            cluster_questions = cluster_assignments[cluster_mask]['question'].tolist()

            if not cluster_questions:
                logger.warning(f"    ⚠ No questions found for Cluster {cluster_id}, skipping...")
                continue

            cluster_data.append({
                'cluster_id': cluster_id,
                'questions': cluster_questions
            })

        # Filter clusters if testing on subset
        if self.test_clusters:
            cluster_data = [c for c in cluster_data if c['cluster_id'] in self.test_clusters]
            logger.info(f"  Testing with {len(cluster_data)} clusters: {self.test_clusters}")

        # Process clusters (parallel or sequential)
        cluster_results = []

        with timing_stats.timer("extraction"):
            if self.enable_parallel and len(cluster_data) > 1:
                # Parallel processing
                logger.info(f"\n  Processing {len(cluster_data)} clusters in parallel...")

                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    # Submit all cluster processing tasks
                    future_to_cluster = {
                        executor.submit(
                            self._process_cluster,
                            cluster_info['cluster_id'],
                            cluster_info['questions'],
                            output_path,
                            cot_logger
                        ): cluster_info['cluster_id']
                        for cluster_info in cluster_data
                    }

                    # Collect results as they complete
                    for future in as_completed(future_to_cluster):
                        cluster_id = future_to_cluster[future]
                        try:
                            result = future.result()
                            cluster_results.append(result)

                            # Log stats
                            total_entities = result.get_total_entities()
                            logger.info(f"  ✓ Cluster {cluster_id} complete: {total_entities} unique entities")
                        except Exception as e:
                            logger.error(f"  ✗ Cluster {cluster_id} failed: {e}")

            else:
                # Sequential processing
                logger.info(f"\n  Processing {len(cluster_data)} clusters sequentially...")

                for cluster_info in cluster_data:
                    cluster_id = cluster_info['cluster_id']
                    logger.info(f"\n  Processing Cluster {cluster_id}...")
                    logger.info(f"    Questions in cluster: {len(cluster_info['questions'])}")

                    try:
                        result = self._process_cluster(
                            cluster_id,
                            cluster_info['questions'],
                            output_path,
                            cot_logger
                        )
                        cluster_results.append(result)

                        # Log stats
                        total_entities = result.get_total_entities()
                        logger.info(f"    ✓ Extracted {total_entities} unique entities")
                        for dim_name, count in sorted(result.get_dimension_stats().items()):
                            logger.info(f"      - {dim_name}: {count}")
                    except Exception as e:
                        logger.error(f"    ✗ Failed: {e}")

            logger.info(f"\n  ✓ Completed discovery + extraction for all {len(cluster_results)} clusters")

        # Save CoT reasoning summary
        if cot_logger:
            cot_logger.save_final_summary()

        # Step 3: Cross-Cluster Merging (no global baseline)
        logger.info("\nStep 3: Cross-Cluster Schema Merging")

        with timing_stats.timer("merging"):
            merger = SchemaMerger(fuzzy_threshold=self.similarity_threshold)
            final_schema = merger.merge_clusters(cluster_results)

            # Add counter for final dimensions
            timing_stats.add_counter("final_dimensions", len(final_schema.dimensions))

            # Save final schema (YAML format)
            final_schema_path = output_path / "final_schema.yml"
            final_schema_dict = final_schema.to_yaml_dict()
            with open(final_schema_path, 'w') as f:
                yaml.dump(final_schema_dict, f, sort_keys=False, indent=2, allow_unicode=True)

            logger.info("  ✓ Final schema saved to: final_schema.yml")

            # Also save as JSON for easier programmatic access
            final_schema_json_path = output_path / "final_schema.json"
            with open(final_schema_json_path, 'w') as f:
                json.dump(final_schema.model_dump(mode='json'), f, indent=2)

            logger.info("  ✓ Final schema saved to: final_schema.json")

            # Save slim extraction schema (for corpus extraction)
            extraction_schema_path = output_path / "extraction_schema.yml"
            extraction_schema_dict = final_schema.to_extraction_schema()
            with open(extraction_schema_path, 'w') as f:
                yaml.dump(extraction_schema_dict, f, sort_keys=False, indent=2, allow_unicode=True)

            logger.info("  ✓ Extraction schema saved to: extraction_schema.yml")

        # Save timing statistics
        timing_output = output_path / "discovery_timing.json"
        timing_stats.save_json(timing_output)

        logger.info(f"\n⏱️  Timing Summary:")
        logger.info(f"  Total: {timing_stats.get_summary()['total_duration_seconds']:.1f}s")
        for phase, stats in timing_stats.get_summary()["timings"].items():
            logger.info(f"  {phase}: {stats['total_seconds']:.1f}s")

        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("SCHEMA DISCOVERY COMPLETE")
        logger.info("=" * 80)
        logger.info(f"\nOutput directory: {output_dir}")
        logger.info(f"Final schema: {final_schema_path}")
        logger.info(f"\nDiscovered {len(final_schema.dimensions)} dimensions:")

        for dim_name in sorted(final_schema.get_dimension_names()):
            dim = final_schema.dimensions[dim_name]
            hierarchy = ' → '.join(dim.hierarchy)
            count = dim.total_entity_count
            logger.info(f"  • {dim_name}: {hierarchy} ({count} entities)")

        return final_schema

    def _process_cluster(
        self,
        cluster_id: int,
        cluster_questions: List[str],
        output_path: Path,
        cot_logger: CoTLogger
    ):
        """
        Process a single cluster: discover dimensions + extract entities.

        Args:
            cluster_id: Cluster identifier
            cluster_questions: List of questions in cluster
            output_path: Output directory path
            cot_logger: Shared CoT logger instance

        Returns:
            ClusterSchemaDiscoveryResult with discovered dimensions and entities
        """
        # Use ClusterSchemaDiscoverer for discovery + extraction
        # Pass shared LLM manager (singleton engine pattern) instead of config
        discoverer = ClusterSchemaDiscoverer(
            llm_client=self.llm_client,
            batch_size=self.batch_size,
            cot_logger=cot_logger
        )

        result = discoverer.discover_and_extract(
            cluster_id=cluster_id,
            cluster_questions=cluster_questions,
            allow_new_dimensions=self.allow_new_dimensions,
            adaptive_first_batch=self.adaptive_first_batch,
            first_batch_min=self.first_batch_min,
            first_batch_max=self.first_batch_max,
            first_batch_ratio=self.first_batch_ratio,
            decay_config=self.decay_config
        )

        # Save intermediate result
        cluster_result_path = output_path / f"cluster_{cluster_id}_result.json"
        with open(cluster_result_path, 'w') as f:
            json.dump(result.model_dump(mode='json'), f, indent=2)

        return result


# CLI usage
def main():
    """Command-line interface for schema discovery pipeline."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Discover dimensional schema from questions using clustering + LLM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline with MIRAGE data
  python -m stindex.pipeline.discovery_pipeline \\
      --questions data/original/mirage/train.jsonl \\
      --output-dir data/schema_discovery \\
      --n-clusters 10

  # Reuse existing cluster results (faster)
  python -m stindex.pipeline.discovery_pipeline \\
      --questions data/original/mirage/train.jsonl \\
      --output-dir data/schema_discovery \\
      --reuse-clusters

  # Smaller test run (3 clusters, faster)
  python -m stindex.pipeline.discovery_pipeline \\
      --questions data/original/mirage/train.jsonl \\
      --output-dir data/schema_discovery_test \\
      --n-clusters 3
        """
    )

    parser.add_argument(
        '--questions',
        required=True,
        help='Path to questions.jsonl (e.g., data/original/mirage/train.jsonl)'
    )
    parser.add_argument(
        '--output-dir',
        default='data/schema_discovery',
        help='Output directory for results (default: data/schema_discovery)'
    )
    parser.add_argument(
        '--n-clusters',
        type=int,
        default=10,
        help='Number of question clusters (default: 10)'
    )
    # Removed --n-samples argument - using adaptive batching instead
    parser.add_argument(
        '--similarity-threshold',
        type=float,
        default=0.60,
        help='Fuzzy matching threshold for deduplication (default: 0.60)'
    )
    parser.add_argument(
        '--reuse-clusters',
        action='store_true',
        help='Reuse existing cluster results if available'
    )
    parser.add_argument(
        '--llm-provider',
        default='openai',
        choices=['openai', 'anthropic', 'hf'],
        help='LLM provider to use (default: openai)'
    )
    parser.add_argument(
        '--model',
        default='gpt-4o-mini',
        help='LLM model name (default: gpt-4o-mini)'
    )
    parser.add_argument(
        '--config',
        default='cfg/schema_discovery.yml',
        help='Path to pipeline config file (default: cfg/schema_discovery.yml)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Number of questions per LLM call (overrides config, default: 50)'
    )
    parser.add_argument(
        '--no-parallel',
        action='store_true',
        help='Disable parallel cluster processing'
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        default=None,
        help='Max parallel workers (overrides config, default: 5)'
    )
    parser.add_argument(
        '--test-clusters',
        type=str,
        default=None,
        help='Test with specific cluster IDs (comma-separated, e.g., "0,1")'
    )

    args = parser.parse_args()

    # Load pipeline config
    config = None
    config_path = Path(args.config)
    if config_path.exists():
        logger.info(f"Loading config from: {config_path}")
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        logger.warning(f"Config file not found: {config_path}, using defaults")

    # Override config with CLI args if provided
    if config and args.batch_size is not None:
        config.setdefault('entity_extraction', {})['batch_size'] = args.batch_size

    if config and args.no_parallel:
        config.setdefault('parallel', {})['enabled'] = False

    if config and args.max_workers is not None:
        config.setdefault('parallel', {})['max_workers'] = args.max_workers

    # LLM config
    llm_config = {
        'llm_provider': args.llm_provider,
        'model_name': args.model
    }

    # Parse test_clusters if provided
    test_clusters = None
    if args.test_clusters:
        test_clusters = [int(x.strip()) for x in args.test_clusters.split(',')]

    # Run pipeline
    pipeline = SchemaDiscoveryPipeline(
        llm_config=llm_config,
        config=config,
        n_clusters=args.n_clusters,
        similarity_threshold=args.similarity_threshold,
        test_clusters=test_clusters
    )

    try:
        result = pipeline.discover_schema(
            questions_file=args.questions,
            output_dir=args.output_dir,
            reuse_clusters=args.reuse_clusters
        )

        # Print summary to console
        print("\n" + "=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print(f"\nOutput directory: {args.output_dir}")
        print(f"\nDiscovered {len(result.dimensions)} dimensions:")
        for dim_name in sorted(result.get_dimension_names()):
            dim = result.dimensions[dim_name]
            hierarchy = ' → '.join(dim.hierarchy)
            count = dim.total_entity_count
            print(f"  • {dim_name}: {hierarchy} ({count} entities)")

    except Exception as e:
        logger.error(f"\n✗ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
