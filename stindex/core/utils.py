"""Utility functions for core extraction logic."""

import json
import re
from typing import Type, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


def extract_json_from_text(text: str, model: Type[T]) -> T:
    """
    Extract and validate JSON from LLM output text.

    Simple approach:
    1. Try to find JSON object in text (between { and })
    2. Parse it
    3. Validate with Pydantic model

    Args:
        text: Raw LLM output text (may contain markdown, extra text, etc.)
        model: Pydantic model class for validation

    Returns:
        Validated Pydantic model instance

    Raises:
        ValueError: If no valid JSON found or validation fails
    """
    # Remove markdown code blocks if present
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)

    # Find JSON object (simple approach: find outermost { ... })
    start = text.find('{')
    end = text.rfind('}')

    if start == -1 or end == -1 or start >= end:
        raise ValueError(f"No JSON object found in text: {text[:200]}")

    json_str = text[start:end+1]

    # Parse JSON
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}")

    # Validate with Pydantic
    try:
        return model(**data)
    except ValidationError as e:
        raise ValueError(f"JSON validation failed: {e}")
