"""Prompts module for extraction tasks."""

from stindex.llm.prompts.dimensional_extraction import DimensionalExtractionPrompt
from stindex.llm.prompts.reflection import ReflectionPrompt

__all__ = [
    "DimensionalExtractionPrompt",
    "ReflectionPrompt",
]
