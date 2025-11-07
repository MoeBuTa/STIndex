"""
Pipeline orchestration for STIndex.

Provides multiple execution modes:
1. pipeline: Full pipeline (preprocessing → extraction → visualization)
2. preprocessing: Preprocessing only (scraping → parsing → chunking)
3. extraction: Extraction only (requires preprocessed chunks)
4. visualization: Visualization only (requires extraction results)
"""

from stindex.pipeline.pipeline import STIndexPipeline

__all__ = ["STIndexPipeline"]
