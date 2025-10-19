"""Prompts module for agents."""

from stindex.agents.prompts.base import BasePrompt
from stindex.agents.prompts.extraction import (
    get_combined_prompt,
    get_spatial_prompt,
    get_temporal_prompt,
)

__all__ = [
    "BasePrompt",
    "get_temporal_prompt",
    "get_spatial_prompt",
    "get_combined_prompt",
]
