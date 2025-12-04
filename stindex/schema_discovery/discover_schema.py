"""
End-to-end schema discovery pipeline.

Simplified pipeline using global dimensions + entity tracking.
NO FAISS, NO retrieval - just simple entity list tracking!
"""

from pathlib import Path
from typing import Dict, List, Optional
import json
import yaml
import pandas as pd
from loguru import logger
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

from stindex.schema_discovery.question_clusterer import QuestionClusterer
from stindex.schema_discovery.global_schema_discoverer import GlobalSchemaDiscoverer
from stindex.schema_discovery.cluster_entity_extractor import ClusterEntityExtractor
from stindex.schema_discovery.schema_merger import SchemaMerger


class SchemaDiscoveryPipeline:
    """
    Simplified end-to-end schema discovery pipeline.

    Steps:
    1. Cluster questions (already complete - reuse results)
    2. Discover global dimensions from 200 representative questions
    3. Extract entities per cluster using global dimensions (parallel)
    4. Merge and deduplicate entity lists
    """

    def __init__(
        self,
        llm_config: Dict,
        config: Optional[Dict] = None,
        n_clusters: int = 10,
        n_samples_per_cluster: int = 20,
        similarity_threshold: float = 0.85,
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
            n_samples_per_cluster: Samples per cluster for global discovery
            similarity_threshold: Fuzzy matching threshold for deduplication
            batch_size: Number of questions per LLM call (default: 50)
            enable_parallel: Enable parallel cluster processing (default: True)
            max_workers: Max parallel workers for cluster processing (default: 5)
            test_clusters: Optional list of cluster IDs to test on (e.g., [0, 1])
        """
        self.llm_config = llm_config
        self.test_clusters = test_clusters

        # Load config if provided
        if config:
            self.n_clusters = config.get('global_discovery', {}).get('num_clusters', n_clusters)
            self.n_samples_per_cluster = config.get('global_discovery', {}).get('samples_per_cluster', n_samples_per_cluster)
            self.batch_size = config.get('entity_extraction', {}).get('batch_size', batch_size)
            self.enable_parallel = config.get('parallel', {}).get('enabled', enable_parallel)
            self.max_workers = config.get('parallel', {}).get('max_workers', max_workers)
            self.similarity_threshold = config.get('schema_merging', {}).get('min_entity_frequency', similarity_threshold)
        else:
            self.n_clusters = n_clusters
            self.n_samples_per_cluster = n_samples_per_cluster
            self.batch_size = batch_size
            self.enable_parallel = enable_parallel
            self.max_workers = max_workers
            self.similarity_threshold = similarity_threshold

    def discover_schema(
        self,
        questions_file: str,
        output_dir: str,
        reuse_clusters: bool = True
    ) -> Dict:
        """
        Run full schema discovery pipeline.

        Args:
            questions_file: Path to questions.jsonl (e.g., data/original/mirage/train.jsonl)
            output_dir: Output directory for results
            reuse_clusters: If True, reuse existing cluster results

        Returns:
            {
                'global_dimensions': {...},
                'cluster_results': [...],
                'final_schema': {...}
            }
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 80)
        logger.info("SCHEMA DISCOVERY PIPELINE")
        logger.info("=" * 80)

        # Step 1: Question Clustering
        logger.info("\nStep 1: Question Clustering")
        cluster_samples_path = output_path / "cluster_samples.json"

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
                n_clusters=self.n_clusters,
                n_samples_per_cluster=self.n_samples_per_cluster
            )
            with open(cluster_samples_path) as f:
                cluster_samples = json.load(f)
                cluster_samples = {int(k): v for k, v in cluster_samples.items()}
            logger.info(f"  ✓ Clustered into {self.n_clusters} clusters")

        logger.info("\nStep 2: Global Dimensional Discovery")
        total_samples = sum(len(samples) for samples in cluster_samples.values())
        logger.info(f"  Discovering dimensions from {total_samples} questions...")

        discoverer = GlobalSchemaDiscoverer(
            llm_config=self.llm_config,
            output_dir=output_dir
        )
        global_dimensions = discoverer.discover_dimensions(cluster_samples)

        # Save global dimensions
        global_dims_path = output_path / "global_dimensions.json"
        with open(global_dims_path, 'w') as f:
            json.dump(global_dimensions, f, indent=2)

        logger.info(f"  ✓ Discovered {len(global_dimensions)} dimensions:")
        for dim_name in sorted(global_dimensions.keys()):
            hierarchy = ' → '.join(global_dimensions[dim_name].get('hierarchy', []))
            logger.info(f"    • {dim_name}: {hierarchy}")

        # Step 3: Per-Cluster Entity Extraction
        logger.info("\nStep 3: Per-Cluster Entity Extraction")
        logger.info(f"  Parallel processing: {'enabled' if self.enable_parallel else 'disabled'} (max_workers={self.max_workers})")
        logger.info(f"  Batch size: {self.batch_size} questions per LLM call")

        # Load all questions
        logger.info(f"  Loading questions from {questions_file}...")
        with open(questions_file) as f:
            all_questions_data = [json.loads(line) for line in f]
            all_questions = [q['question'] for q in all_questions_data]

        logger.info(f"  Loaded {len(all_questions)} questions")

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

        # Extract entities per cluster (parallel or sequential)
        cluster_results = []

        if self.enable_parallel and len(cluster_data) > 1:
            # Parallel processing
            logger.info(f"\n  Processing {len(cluster_data)} clusters in parallel...")

            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit all cluster extraction tasks
                future_to_cluster = {
                    executor.submit(
                        self._process_cluster,
                        cluster_info['cluster_id'],
                        cluster_info['questions'],
                        global_dimensions,
                        output_path
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
                        total_entities = sum(count for count in result['entity_counts'].values())
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
                        global_dimensions,
                        output_path
                    )
                    cluster_results.append(result)

                    # Log stats
                    total_entities = sum(count for count in result['entity_counts'].values())
                    logger.info(f"    ✓ Extracted {total_entities} unique entities")
                    for dim_name, count in sorted(result['entity_counts'].items()):
                        logger.info(f"      - {dim_name}: {count}")
                except Exception as e:
                    logger.error(f"    ✗ Failed: {e}")

        logger.info(f"\n  ✓ Completed extraction for all {len(cluster_results)} clusters")

        # Step 4: Merge & Deduplicate
        logger.info("\nStep 4: Merge & Deduplicate")

        merger = SchemaMerger(similarity_threshold=self.similarity_threshold)
        final_schema = merger.merge_clusters(cluster_results, global_dimensions)

        # Save final schema (YAML format)
        final_schema_path = output_path / "final_schema.yml"
        with open(final_schema_path, 'w') as f:
            yaml.dump(final_schema, f, sort_keys=False, indent=2, allow_unicode=True)

        logger.info("  ✓ Final schema saved to: final_schema.yml")

        # Also save as JSON for easier programmatic access
        final_schema_json_path = output_path / "final_schema.json"
        with open(final_schema_json_path, 'w') as f:
            json.dump(final_schema, f, indent=2)

        logger.info("  ✓ Final schema saved to: final_schema.json")

        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("SCHEMA DISCOVERY COMPLETE")
        logger.info("=" * 80)
        logger.info(f"\nOutput directory: {output_dir}")
        logger.info(f"Final schema: {final_schema_path}")
        logger.info(f"\nDiscovered {len(final_schema)} dimensions:")

        for dim_name in sorted(final_schema.keys()):
            dim_info = final_schema[dim_name]
            hierarchy = ' → '.join(dim_info.get('hierarchy', []))
            count = dim_info.get('count', 0)
            logger.info(f"  • {dim_name}: {hierarchy} ({count} entities)")

        return {
            'global_dimensions': global_dimensions,
            'cluster_results': cluster_results,
            'final_schema': final_schema
        }

    def _process_cluster(
        self,
        cluster_id: int,
        cluster_questions: List[str],
        global_dimensions: Dict,
        output_path: Path
    ) -> Dict:
        """
        Process a single cluster (for parallel execution).

        Args:
            cluster_id: Cluster identifier
            cluster_questions: List of questions in cluster
            global_dimensions: Global dimensional schema
            output_path: Output directory path

        Returns:
            Cluster extraction result dict
        """
        # Extract entities
        extractor = ClusterEntityExtractor(
            global_dimensions,
            self.llm_config,
            batch_size=self.batch_size,
            output_dir=str(output_path)
        )
        result = extractor.extract_from_cluster(cluster_questions, cluster_id)

        # Save intermediate result
        cluster_result_path = output_path / f"cluster_{cluster_id}_entities.json"
        with open(cluster_result_path, 'w') as f:
            json.dump(result, f, indent=2)

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
  python -m stindex.schema_discovery.discover_schema \\
      --questions data/original/mirage/train.jsonl \\
      --output-dir data/schema_discovery \\
      --n-clusters 10 \\
      --n-samples 20

  # Reuse existing cluster results (faster)
  python -m stindex.schema_discovery.discover_schema \\
      --questions data/original/mirage/train.jsonl \\
      --output-dir data/schema_discovery \\
      --reuse-clusters

  # Smaller test run (3 clusters, faster)
  python -m stindex.schema_discovery.discover_schema \\
      --questions data/original/mirage/train.jsonl \\
      --output-dir data/schema_discovery_test \\
      --n-clusters 3 \\
      --n-samples 10
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
    parser.add_argument(
        '--n-samples',
        type=int,
        default=20,
        help='Number of samples per cluster for global discovery (default: 20)'
    )
    parser.add_argument(
        '--similarity-threshold',
        type=float,
        default=0.85,
        help='Fuzzy matching threshold for deduplication (default: 0.85)'
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
        default='stindex/schema_discovery/config/schema_discovery.yml',
        help='Path to pipeline config file (default: stindex/schema_discovery/config/schema_discovery.yml)'
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
        n_samples_per_cluster=args.n_samples,
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
        print(f"\nDiscovered {len(result['final_schema'])} dimensions:")
        for dim_name, dim_info in result['final_schema'].items():
            hierarchy = dim_info.get('hierarchy', [])
            count = dim_info.get('count', 0)
            print(f"  • {dim_name}: {' → '.join(hierarchy)} ({count} entities)")

    except Exception as e:
        logger.error(f"\n✗ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
