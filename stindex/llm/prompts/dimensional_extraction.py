"""
Dynamic prompt generation for multi-dimensional extraction.

Generates extraction prompts based on dimension configuration,
supporting any combination of dimensions defined in YAML.
"""

import json
from typing import Dict, List, Optional

from stindex.utils.dimension_loader import DimensionConfig


class DimensionalExtractionPrompt:
    """
    Dynamic prompt builder for multi-dimensional extraction.

    Generates prompts based on dimension configuration instead of hardcoded templates.
    """

    def __init__(
        self,
        dimensions: Dict[str, DimensionConfig],
        document_metadata: Optional[Dict] = None,
        extraction_context: Optional[object] = None  # ExtractionContext (avoid circular import)
    ):
        """
        Initialize prompt builder.

        Args:
            dimensions: Dict of dimension name → DimensionConfig
            document_metadata: Optional document metadata (publication_date, source_location, etc.)
            extraction_context: Optional ExtractionContext for context-aware prompts
        """
        self.dimensions = dimensions
        self.document_metadata = document_metadata or {}
        self.extraction_context = extraction_context

    def system_prompt(self) -> str:
        """Generate system prompt with multi-dimensional extraction instructions."""

        # Base instructions
        prompt = """You are a precise JSON extraction bot. Your ONLY output must be valid JSON.

CRITICAL RULES:
- Output ONLY the JSON object, nothing else
- NO explanations, NO reasoning, NO extra text
- Start your response with { and end with }
- Do not write "Here is the JSON" or similar phrases

"""

        # Add extraction context if available (cmem - memory context)
        if self.extraction_context:
            context_str = self.extraction_context.to_prompt_context()
            if context_str.strip():
                prompt += context_str + "\n"

        # Add document metadata context if available (but not if already in extraction_context)
        if self.document_metadata and not self.extraction_context:
            prompt += "DOCUMENT CONTEXT:\n"
            if self.document_metadata.get("publication_date"):
                prompt += f"- Publication Date: {self.document_metadata['publication_date']}\n"
            if self.document_metadata.get("source_location"):
                prompt += f"- Source Location: {self.document_metadata['source_location']}\n"
            if self.document_metadata.get("source_url"):
                prompt += f"- Source: {self.document_metadata['source_url']}\n"
            prompt += "\n"

        # Add extraction tasks for each dimension
        prompt += "EXTRACTION TASKS:\n\n"

        for i, (dim_name, dim_config) in enumerate(self.dimensions.items(), 1):
            prompt += f"{i}. Extract {dim_name.upper()} ({dim_config.extraction_type}):\n"
            prompt += f"   {dim_config.description}\n"

            # Add dimension-specific instructions
            if dim_config.extraction_type == "normalized":
                prompt += self._get_normalized_instructions(dim_config)
            elif dim_config.extraction_type == "geocoded":
                prompt += self._get_geocoded_instructions(dim_config)
            elif dim_config.extraction_type == "categorical":
                prompt += self._get_categorical_instructions(dim_config)
            elif dim_config.extraction_type == "structured":
                prompt += self._get_structured_instructions(dim_config)

            prompt += "\n"

        prompt += "REMINDER: Return ONLY valid JSON, nothing else."

        return prompt

    def _get_normalized_instructions(self, dim_config: DimensionConfig) -> str:
        """Get instructions for normalized dimensions."""
        instructions = ""

        if dim_config.normalization:
            norm_config = dim_config.normalization

            if dim_config.name == "temporal":
                instructions += "   - Normalize to ISO 8601 format:\n"
                instructions += "     * Dates: YYYY-MM-DD\n"
                instructions += "     * Datetimes: YYYY-MM-DDTHH:MM:SS\n"
                instructions += "     * Durations: P1D, P2M, P3Y\n"
                instructions += "     * Intervals: start/end (e.g., 2025-10-27T11:00:00/2025-10-27T19:00:00)\n"

                if norm_config.get("handle_relative"):
                    if self.document_metadata.get("publication_date"):
                        instructions += f"   - For relative dates (e.g., 'Monday'), use document date {self.document_metadata['publication_date']} as anchor\n"
                    else:
                        instructions += "   - For relative dates, use most recent occurrence\n"

                if norm_config.get("default_year") == "document":
                    instructions += "   - For dates without years: use most recent year in document context\n"
            else:
                instructions += "   - Normalize to standard format\n"

        return instructions

    def _get_geocoded_instructions(self, dim_config: DimensionConfig) -> str:
        """Get instructions for geocoded dimensions."""
        instructions = ""

        if dim_config.disambiguation:
            disamb_config = dim_config.disambiguation

            if disamb_config.get("use_parent_region"):
                instructions += "   - Include parent region for disambiguation (state, country)\n"

            if disamb_config.get("use_source_location") and self.document_metadata.get("source_location"):
                instructions += f"   - Consider source location: {self.document_metadata['source_location']}\n"

            # Add nearby locations context if available via extraction_context
            if self.extraction_context and self.extraction_context.enable_nearby_locations:
                nearby_context = self.extraction_context.get_nearby_locations_context()
                if nearby_context:
                    instructions += "   - " + nearby_context.replace("\n", "\n   - ") + "\n"

            # Add location type examples
            location_types = self._get_field_enum_values(dim_config, "location_type")
            if location_types:
                instructions += f"   - Location types: {', '.join(location_types[:5])}, etc.\n"

        return instructions

    def _get_categorical_instructions(self, dim_config: DimensionConfig) -> str:
        """Get instructions for categorical dimensions."""
        instructions = ""

        # Get category values
        category_values = self._get_field_enum_values(dim_config, "category")
        if category_values:
            instructions += "   - Allowed categories:\n"
            for cat in category_values[:10]:  # Show up to 10
                instructions += f"     * {cat}\n"
            if len(category_values) > 10:
                instructions += f"     * ... ({len(category_values) - 10} more)\n"

        return instructions

    def _get_structured_instructions(self, dim_config: DimensionConfig) -> str:
        """Get instructions for structured dimensions."""
        instructions = "   - Extract structured fields:\n"

        for field in dim_config.fields:
            field_name = field.get("name")
            field_type = field.get("type")
            field_desc = field.get("description", "")
            instructions += f"     * {field_name} ({field_type}): {field_desc}\n"

        return instructions

    def _get_field_enum_values(self, dim_config: DimensionConfig, field_name: str) -> List[str]:
        """Get enum values for a specific field."""
        for field in dim_config.fields:
            if field.get("name") == field_name and field.get("type") == "enum":
                return field.get("values", [])
        return []

    def user_prompt(self, text: str) -> str:
        """Generate user prompt with text to extract from."""
        return text

    def assistant_prompt_example(self) -> str:
        """Generate example assistant response for few-shot learning."""
        # Build example JSON with all dimensions
        example = {}

        for dim_name, dim_config in self.dimensions.items():
            if dim_config.examples:
                # Use first example as demonstration
                example_data = dim_config.examples[0]

                # Format based on dimension type
                if dim_config.extraction_type == "normalized":
                    example[dim_name] = [{
                        "text": example_data.get("input", example_data.get("text", "")),
                        "normalized": example_data.get("output", {}).get("normalized", example_data.get("normalized", "")),
                        dim_config.fields[2]["name"]: example_data.get("output", {}).get(dim_config.fields[2]["name"], example_data.get(dim_config.fields[2]["name"], ""))
                    }]
                elif dim_config.extraction_type == "geocoded":
                    example[dim_name] = [{
                        "text": example_data.get("input", example_data.get("text", "")),
                        "location_type": example_data.get("output", {}).get("location_type", example_data.get("location_type", "")),
                        "parent_region": example_data.get("output", {}).get("parent_region", example_data.get("parent_region", ""))
                    }]
                elif dim_config.extraction_type == "categorical":
                    example[dim_name] = [{
                        "text": example_data.get("input", example_data.get("text", "")),
                        "category": example_data.get("output", {}).get("category", example_data.get("category", "")),
                        "confidence": example_data.get("output", {}).get("confidence", example_data.get("confidence", 1.0))
                    }]
                else:
                    # Generic format
                    example[dim_name] = [example_data.get("output", example_data)]

        return json.dumps(example, indent=2)

    def user_prompt_example(self) -> str:
        """Generate example user prompt for few-shot learning."""
        # Try to construct an example sentence from dimension examples
        example_parts = []

        for dim_name, dim_config in self.dimensions.items():
            if dim_config.examples:
                example_data = dim_config.examples[0]
                input_text = example_data.get("input", "")
                if input_text:
                    example_parts.append(input_text)

        if example_parts:
            return " ".join(example_parts)
        else:
            return "Example document text for extraction."

    def build_messages(
        self,
        text: str,
        use_few_shot: bool = False
    ) -> List[Dict[str, str]]:
        """
        Build message list for extraction.

        Args:
            text: Input text to extract from
            use_few_shot: Whether to include few-shot example

        Returns:
            List of message dicts
        """
        messages = [
            {"role": "system", "content": self.system_prompt()}
        ]

        if use_few_shot:
            # Add few-shot example
            messages.extend([
                {"role": "user", "content": self.user_prompt_example()},
                {"role": "assistant", "content": self.assistant_prompt_example()},
                {"role": "user", "content": text}
            ])
        else:
            messages.append({"role": "user", "content": self.user_prompt(text)})

        return messages

    def get_json_schema(self, schema: Dict) -> str:
        """Get JSON schema instruction string."""
        schema_str = json.dumps(schema, indent=2)
        return f"""

Respond only in raw JSON. Schema:
{schema_str}"""

    def build_messages_with_schema(
        self,
        text: str,
        json_schema: Dict[str, any],
        use_few_shot: bool = False
    ) -> List[Dict[str, str]]:
        """
        Build messages with JSON schema instruction.

        Args:
            text: Input text to extract from
            json_schema: JSON schema for response
            use_few_shot: Whether to include few-shot example

        Returns:
            List of message dicts with schema instruction
        """
        messages = self.build_messages(text, use_few_shot)

        # Add schema instruction to the last user message
        schema_instruction = self.get_json_schema(json_schema)
        messages[-1]["content"] += schema_instruction

        return messages
