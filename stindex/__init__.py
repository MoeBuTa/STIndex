"""
STIndex: Spatiotemporal Index Extraction from Unstructured Text

A simple Python library for extracting and normalizing spatiotemporal information from text.
Uses native LLM providers for clean, type-safe interactions.

v0.3.0: Added multi-dimensional extraction framework for domain-specific extraction.
"""

__version__ = "0.3.0"
__author__ = "STIndex Team"
__license__ = "MIT"

# Legacy API (backward compatible)
from stindex.core.extraction import STIndexExtractor

# New multi-dimensional API
from stindex.core.dimensional_extraction import DimensionalExtractor

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


