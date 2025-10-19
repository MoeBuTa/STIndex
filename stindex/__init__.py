"""
STIndex: Spatiotemporal Index Extraction from Unstructured Text

A Python library for extracting and normalizing spatiotemporal indices from text.
Uses agentic architecture with observe-reason-act pattern.
"""

__version__ = "0.2.0"
__author__ = "STIndex Team"
__license__ = "MIT"

from stindex.pipeline.extraction_pipeline import ExtractionPipeline
from stindex.pipeline.models import ExtractionResult, BatchExtractionResult

__all__ = [
    "ExtractionPipeline",
    "ExtractionResult",
    "BatchExtractionResult",
]
