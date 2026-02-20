"""
STIndex: Spatiotemporal Index Extraction from Unstructured Text

A simple Python library for extracting and normalizing spatiotemporal information from text.
Uses native LLM providers for clean, type-safe interactions.

v0.5.0: Unified on DimensionalExtractor for multi-dimensional extraction.
"""

__version__ = "1.0.2"
__author__ = "STIndex Team"
__license__ = "MIT"

# Multi-dimensional extraction API
from stindex.extraction.dimensional_extraction import DimensionalExtractor

# Preprocessing API
from stindex.preprocess import DocumentChunk, InputDocument, Preprocessor

# Pipeline API
from stindex.pipeline import STIndexPipeline

# Dimensional models
from stindex.extraction.dimension_models import (
    CategoricalDimensionEntity,
    CategoricalDimensionMention,
    DimensionType,
    GeocodedDimensionEntity,
    GeocodedDimensionMention,
    MultiDimensionalResult,
    NormalizedDimensionEntity,
    NormalizedDimensionMention,
)

__all__ = [
    # Main API
    "DimensionalExtractor",
    # Preprocessing API
    "Preprocessor",
    "InputDocument",
    "DocumentChunk",
    # Pipeline API
    "STIndexPipeline",
    # Dimensional Models
    "MultiDimensionalResult",
    "DimensionType",
    "NormalizedDimensionMention",
    "NormalizedDimensionEntity",
    "GeocodedDimensionMention",
    "GeocodedDimensionEntity",
    "CategoricalDimensionMention",
    "CategoricalDimensionEntity",
]


