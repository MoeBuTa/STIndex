"""
STIndex: Spatiotemporal Index Extraction from Unstructured Text

A Python library for extracting and normalizing spatiotemporal indices from text.
"""

__version__ = "0.1.0"
__author__ = "STIndex Team"
__license__ = "MIT"

from stindex.core.extractor import STIndexExtractor
from stindex.models.schemas import (
    SpatialEntity,
    SpatioTemporalResult,
    TemporalEntity,
)

__all__ = [
    "STIndexExtractor",
    "TemporalEntity",
    "SpatialEntity",
    "SpatioTemporalResult",
]
