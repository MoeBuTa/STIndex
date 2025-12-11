"""
Pipeline orchestration for STIndex.

Provides multiple execution modes:
1. STIndexPipeline: Full pipeline (preprocessing → extraction → visualization)
2. SchemaDiscoveryPipeline: Schema discovery from Q&A datasets (clustering → discovery → merging)

Execution modes for STIndexPipeline:
- preprocessing: Preprocessing only (scraping → parsing → chunking)
- extraction: Extraction only (requires preprocessed chunks)
- visualization: Visualization only (requires extraction results)
"""

from stindex.pipeline.pipeline import STIndexPipeline
from stindex.pipeline.discovery_pipeline import SchemaDiscoveryPipeline

__all__ = ["STIndexPipeline", "SchemaDiscoveryPipeline"]
