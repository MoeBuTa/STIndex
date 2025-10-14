"""
Prompt templates for entity extraction.
"""

from stindex.prompts.templates import (
    get_temporal_prompt,
    get_spatial_prompt,
    get_combined_prompt,
    get_disambiguation_prompt,
    TEMPORAL_EXTRACTION_SYSTEM,
    SPATIAL_EXTRACTION_SYSTEM,
)

__all__ = [
    "get_temporal_prompt",
    "get_spatial_prompt",
    "get_combined_prompt",
    "get_disambiguation_prompt",
    "TEMPORAL_EXTRACTION_SYSTEM",
    "SPATIAL_EXTRACTION_SYSTEM",
]
