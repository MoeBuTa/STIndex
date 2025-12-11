"""
Schema Discovery Module for STIndex.

This module implements cluster-level dimensional schema discovery from question-answering datasets.

Main Components:
- QuestionClusterer: Semantic clustering of questions using sentence embeddings
- ClusterSchemaDiscoverer: Discovers dimensions from questions within a cluster
- ClusterEntityExtractor: Extracts entities per cluster using discovered dimensions
- SchemaMerger: Merges entity lists from all clusters with fuzzy deduplication
- SchemaDiscoveryPipeline: End-to-end pipeline orchestration (now in stindex.pipeline)

Example Usage:
    from stindex.pipeline.discovery_pipeline import SchemaDiscoveryPipeline

    # Run full pipeline
    pipeline = SchemaDiscoveryPipeline(
        llm_config={'llm_provider': 'openai', 'model_name': 'gpt-4o-mini'},
        n_clusters=10,
        n_samples_for_discovery=20
    )

    result = pipeline.discover_schema(
        questions_file='data/original/mirage/train.jsonl',
        output_dir='data/schema_discovery'
    )
"""

from stindex.discovery.question_clusterer import QuestionClusterer
from stindex.discovery.cluster_schema_discoverer import ClusterSchemaDiscoverer
from stindex.discovery.cluster_entity_extractor import ClusterEntityExtractor
from stindex.discovery.schema_merger import SchemaMerger

__all__ = [
    'QuestionClusterer',
    'ClusterSchemaDiscoverer',
    'ClusterEntityExtractor',
    'SchemaMerger'
]
