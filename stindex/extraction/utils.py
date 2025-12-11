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
    1. Find the LAST complete JSON object or array in text (model may generate multiple attempts)
    2. Parse it
    3. Validate with Pydantic model (if provided)

    Args:
        text: Raw LLM output text (may contain markdown, extra text, etc.)
        model: Pydantic model class for validation (optional)
        return_dict: If True, return dict/list instead of validated model

    Returns:
        Validated Pydantic model instance, dict, or list

    Raises:
        ValueError: If no valid JSON found or validation fails
    """
    # Remove markdown code blocks if present
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)

    # Find all complete JSON objects and arrays with balanced braces/brackets
    json_candidates = []

    i = 0
    while i < len(text):
        # Find next opening brace or bracket
        obj_start = text.find('{', i)
        arr_start = text.find('[', i)

        # Determine which comes first
        if obj_start == -1 and arr_start == -1:
            break
        elif obj_start == -1:
            start = arr_start
            is_array = True
        elif arr_start == -1:
            start = obj_start
            is_array = False
        else:
            start = min(obj_start, arr_start)
            is_array = (start == arr_start)

        # Track brace/bracket depth to find matching closing brace/bracket
        depth = 0
        in_string = False
        escape = False
        end = -1

        open_char = '[' if is_array else '{'
        close_char = ']' if is_array else '}'

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

            # Only count braces/brackets outside of strings
            if not in_string:
                if char == open_char:
                    depth += 1
                elif char == close_char:
                    depth -= 1
                    if depth == 0:
                        end = j
                        break

        if end != -1:
            json_candidates.append(text[start:end+1])
            i = end + 1
        else:
            i = start + 1

    if not json_candidates:
        raise ValueError(f"No complete JSON object or array found in text: {text[:200]}")

    # Try parsing JSON candidates from last to first (prefer most recent)
    last_error = None
    for json_str in reversed(json_candidates):
        try:
            data = json.loads(json_str)

            # Return dict/list if requested or model not provided
            if return_dict or model is None:
                return data

            # Validate with Pydantic
            return model(**data)
        except (json.JSONDecodeError, ValidationError) as e:
            last_error = e
            continue

    # If we get here, none of the candidates were valid
    raise ValueError(f"No valid JSON found. Last error: {last_error}")


def extract_cot_and_json(llm_response: str) -> Dict:
    """
    Extract Chain-of-Thought reasoning and JSON from LLM response.

    Handles different CoT formats:
    - <think>...</think> tags (Qwen3, DeepSeek)
    - Text before JSON (OpenAI, Anthropic)
    - No reasoning (direct JSON output)

    Args:
        llm_response: Raw LLM response

    Returns:
        Dict with:
            - 'reasoning': Extracted reasoning text (or empty string)
            - 'data': Parsed JSON data (dict or list)
            - 'raw_response': Original LLM response

    Raises:
        ValueError: If no valid JSON found
    """
    # Extract <think> tags (for Qwen3, DeepSeek, etc.)
    think_pattern = r'<think>(.*?)</think>'
    think_matches = re.findall(think_pattern, llm_response, re.DOTALL)

    if think_matches:
        # Join all <think> blocks
        reasoning = '\n\n'.join(match.strip() for match in think_matches)

        # Remove <think> tags from response for JSON extraction
        json_text = re.sub(think_pattern, '', llm_response, flags=re.DOTALL)
    else:
        # No <think> tags - extract text before JSON
        # Find first JSON opening brace/bracket
        obj_start = llm_response.find('{')
        arr_start = llm_response.find('[')

        if obj_start == -1 and arr_start == -1:
            reasoning = ""
            json_text = llm_response
        else:
            # Determine which comes first
            if obj_start == -1:
                json_start = arr_start
            elif arr_start == -1:
                json_start = obj_start
            else:
                json_start = min(obj_start, arr_start)

            # Text before JSON is reasoning
            reasoning = llm_response[:json_start].strip()
            json_text = llm_response[json_start:]

            # Clean up common reasoning prefixes
            reasoning = re.sub(r'^(here is|here\'s|the|output:?)\s*', '', reasoning, flags=re.IGNORECASE)
            reasoning = reasoning.strip()

    # Extract JSON data
    try:
        data = extract_json_from_text(json_text, return_dict=True)
    except ValueError as e:
        # Try extracting from full response if json_text extraction failed
        try:
            data = extract_json_from_text(llm_response, return_dict=True)
        except ValueError:
            # Re-raise original error
            raise e

    return {
        'reasoning': reasoning,
        'data': data,
        'raw_response': llm_response
    }

