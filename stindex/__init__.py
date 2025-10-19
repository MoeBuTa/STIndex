"""
STIndex: Spatiotemporal Index Extraction from Unstructured Text

A simple Python library for extracting and normalizing spatiotemporal information from text.
Uses Instructor framework for clean, type-safe LLM interactions.
"""

__version__ = "0.3.0"
__author__ = "STIndex Team"
__license__ = "MIT"

from stindex.agents.extractor import STIndexExtractor
from stindex.agents.llm.client import UnifiedLLMClient, create_llm_client
from stindex.agents.response.models import (
    ExtractionResult,
    LocationType,
    SpatialEntity,
    SpatialMention,
    SpatioTemporalResult,
    TemporalEntity,
    TemporalMention,
    TemporalType,
)

__all__ = [
    # Main API
    "STIndexExtractor",
    # LLM Client
    "UnifiedLLMClient",
    "create_llm_client",
    # Models
    "ExtractionResult",
    "SpatioTemporalResult",
    "TemporalEntity",
    "SpatialEntity",
    "TemporalMention",
    "SpatialMention",
    "TemporalType",
    "LocationType",
]

