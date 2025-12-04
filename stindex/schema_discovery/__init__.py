"""
Schema Discovery Module for STIndex.

This module implements simplified cluster-based dimensional schema discovery from question-answering datasets.

Main Components:
- QuestionClusterer: Semantic clustering of questions using sentence embeddings
- GlobalSchemaDiscoverer: Discovers global dimensions from representative samples across all clusters
- ClusterEntityExtractor: Extracts entities per cluster using global dimensions with simple entity tracking
- SchemaMerger: Merges entity lists from all clusters with fuzzy deduplication
- SchemaDiscoveryPipeline: End-to-end pipeline orchestration

Example Usage:
    from stindex.schema_discovery import SchemaDiscoveryPipeline

    # Run full pipeline
    pipeline = SchemaDiscoveryPipeline(
        llm_config={'llm_provider': 'openai', 'model_name': 'gpt-4o-mini'},
        n_clusters=10,
        n_samples_per_cluster=20
    )

    result = pipeline.discover_schema(
        questions_file='data/original/mirage/train.jsonl',
        output_dir='data/schema_discovery'
    )
"""

from stindex.schema_discovery.question_clusterer import QuestionClusterer
from stindex.schema_discovery.global_schema_discoverer import GlobalSchemaDiscoverer
from stindex.schema_discovery.cluster_entity_extractor import ClusterEntityExtractor
from stindex.schema_discovery.schema_merger import SchemaMerger
from stindex.schema_discovery.discover_schema import SchemaDiscoveryPipeline

__all__ = [
    'QuestionClusterer',
    'GlobalSchemaDiscoverer',
    'ClusterEntityExtractor',
    'SchemaMerger',
    'SchemaDiscoveryPipeline'
]
