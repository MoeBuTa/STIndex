"""
STIndex: Spatiotemporal Index Extraction from Unstructured Text

A simple Python library for extracting and normalizing spatiotemporal information from text.
Uses native LLM providers for clean, type-safe interactions.

v0.4.0: Added generic preprocessing module and end-to-end pipeline.
"""

__version__ = "0.4.0"
__author__ = "STIndex Team"
__license__ = "MIT"

# Legacy API (backward compatible)
from stindex.core.extraction import STIndexExtractor

# New multi-dimensional API
from stindex.core.dimensional_extraction import DimensionalExtractor

# Preprocessing API
from stindex.preprocessing import DocumentChunk, InputDocument, Preprocessor

# Pipeline API
from stindex.pipeline import STIndexPipeline

# Visualization API
from stindex.visualization import STIndexVisualizer

# Response models (legacy)
from stindex.llm.response.models import (
    ExtractionResult,
    LocationType,
    SpatialEntity,
    SpatialMention,
    SpatioTemporalResult,
    TemporalEntity,
    TemporalMention,
    TemporalType,
)

# New dimensional models
from stindex.llm.response.dimension_models import (
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
    "STIndexExtractor",  # Legacy extractor (temporal + spatial only)
    "DimensionalExtractor",  # New multi-dimensional extractor
    # Preprocessing API
    "Preprocessor",  # Generic preprocessing orchestrator
    "InputDocument",  # Input document model
    "DocumentChunk",  # Document chunk model
    # Pipeline API
    "STIndexPipeline",  # End-to-end pipeline orchestrator
    # Visualization API
    "STIndexVisualizer",  # Visualization orchestrator
    # Legacy Models
    "ExtractionResult",
    "SpatioTemporalResult",
    "TemporalEntity",
    "SpatialEntity",
    "TemporalMention",
    "SpatialMention",
    "TemporalType",
    "LocationType",
    # New Dimensional Models
    "MultiDimensionalResult",
    "DimensionType",
    "NormalizedDimensionMention",
    "NormalizedDimensionEntity",
    "GeocodedDimensionMention",
    "GeocodedDimensionEntity",
    "CategoricalDimensionMention",
    "CategoricalDimensionEntity",
]


