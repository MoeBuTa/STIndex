"""Utility functions for core extraction logic."""

import json
import re
from typing import Dict, Optional, Type, TypeVar, Union

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


def extract_json_from_text(
    text: str,
    model: Optional[Type[T]] = None,
    return_dict: bool = False
) -> Union[T, Dict]:
    """
    Extract and validate JSON from LLM output text.

    Approach:
    1. Find the LAST complete JSON object in text (model may generate multiple attempts)
    2. Parse it
    3. Validate with Pydantic model (if provided)

    Args:
        text: Raw LLM output text (may contain markdown, extra text, etc.)
        model: Pydantic model class for validation (optional)
        return_dict: If True, return dict instead of validated model

    Returns:
        Validated Pydantic model instance or dict

    Raises:
        ValueError: If no valid JSON found or validation fails
    """
    # Remove markdown code blocks if present
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)

    # Find all complete JSON objects with balanced braces
    json_candidates = []

    i = 0
    while i < len(text):
        # Find next opening brace
        start = text.find('{', i)
        if start == -1:
            break

        # Track brace depth to find matching closing brace
        brace_count = 0
        in_string = False
        escape = False
        end = -1

        for j in range(start, len(text)):
            char = text[j]

            # Handle string escaping
            if escape:
                escape = False
                continue
            if char == '\\':
                escape = True
                continue

            # Track if we're inside a string
            if char == '"':
                in_string = not in_string
                continue

            # Only count braces outside of strings
            if not in_string:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end = j
                        break

        if end != -1:
            json_candidates.append(text[start:end+1])
            i = end + 1
        else:
            i = start + 1

    if not json_candidates:
        raise ValueError(f"No complete JSON object found in text: {text[:200]}")

    # Try parsing JSON candidates from last to first (prefer most recent)
    last_error = None
    for json_str in reversed(json_candidates):
        try:
            data = json.loads(json_str)

            # Return dict if requested or model not provided
            if return_dict or model is None:
                return data

            # Validate with Pydantic
            return model(**data)
        except (json.JSONDecodeError, ValidationError) as e:
            last_error = e
            continue

    # If we get here, none of the candidates were valid
    raise ValueError(f"No valid JSON found. Last error: {last_error}")
