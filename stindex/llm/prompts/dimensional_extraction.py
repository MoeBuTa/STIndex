"""
Dynamic prompt generation for multi-dimensional extraction.

Generates extraction prompts based on dimension configuration,
supporting any combination of dimensions defined in YAML.
"""

import json
from typing import Dict, List, Optional

from stindex.extraction.context_manager import ExtractionContext
from stindex.extraction.dimension_loader import DimensionConfig


class DimensionalExtractionPrompt:
    """
    Dynamic prompt builder for multi-dimensional extraction.

    Generates prompts based on dimension configuration instead of hardcoded templates.
    """

    def __init__(
        self,
        dimensions: Dict[str, DimensionConfig],
        document_metadata: Optional[Dict] = None,
        extraction_context: Optional[ExtractionContext] = None  # ExtractionContext (avoid circular import)
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
        self.extraction_context: ExtractionContext = extraction_context

    def system_prompt(self) -> str:
        """Generate system prompt with multi-dimensional extraction instructions."""

        # Build context sections
        extraction_context = self._build_extraction_context()
        document_context = self._build_document_context()
        dimension_tasks = self._build_dimension_tasks()

        # Template with placeholders
        template = """You are a precise JSON extraction bot. Your ONLY output must be valid JSON.

CRITICAL RULES:
- Output ONLY the JSON object, nothing else
- NO explanations, NO reasoning, NO extra text
- Start your response with {{ and end with }}
- Do not write "Here is the JSON" or similar phrases

{extraction_context}{document_context}EXTRACTION TASKS:

{dimension_tasks}
REMINDER: Return ONLY valid JSON, nothing else."""

        return template.format(
            extraction_context=extraction_context,
            document_context=document_context,
            dimension_tasks=dimension_tasks
        )

    def _build_extraction_context(self) -> str:
        """Build extraction context section (cmem - memory context)."""
        if self.extraction_context:
            context_str = self.extraction_context.to_prompt_context()
            if context_str.strip():
                return f"{context_str}\n"
        return ""

    def _build_document_context(self) -> str:
        """Build document metadata context section."""
        # Skip if extraction_context already provides this
        if not self.document_metadata or self.extraction_context:
            return ""

        context_parts = []
        if self.document_metadata.get("publication_date"):
            context_parts.append("- Publication Date: {publication_date}")
        if self.document_metadata.get("source_location"):
            context_parts.append("- Source Location: {source_location}")
        if self.document_metadata.get("source_url"):
            context_parts.append("- Source: {source_url}")

        if context_parts:
            context_template = "DOCUMENT CONTEXT:\n{context}\n\n"
            context_content = "\n".join(context_parts).format(**self.document_metadata)
            return context_template.format(context=context_content)

        return ""

    def _build_dimension_tasks(self) -> str:
        """Build extraction tasks for all dimensions."""
        task_parts = []

        for i, (dim_name, dim_config) in enumerate(self.dimensions.items(), 1):
            # Get dimension-specific instructions
            specific_instructions = ""
            if dim_config.extraction_type == "normalized":
                specific_instructions = self._get_normalized_instructions(dim_config)
            elif dim_config.extraction_type == "geocoded":
                specific_instructions = self._get_geocoded_instructions(dim_config)
            elif dim_config.extraction_type == "categorical":
                specific_instructions = self._get_categorical_instructions(dim_config)
            elif dim_config.extraction_type == "structured":
                specific_instructions = self._get_structured_instructions(dim_config)

            task_template = "{index}. Extract {dimension_name} ({extraction_type}):\n   {description}\n{specific_instructions}"
            task_parts.append(task_template.format(
                index=i,
                dimension_name=dim_name.upper(),
                extraction_type=dim_config.extraction_type,
                description=dim_config.description,
                specific_instructions=specific_instructions
            ))

        return "\n\n".join(task_parts)

    def _get_normalized_instructions(self, dim_config: DimensionConfig) -> str:
        """Get instructions for normalized dimensions."""
        if not dim_config.normalization:
            return ""

        norm_config = dim_config.normalization
        instruction_parts = []

        if dim_config.name == "temporal":
            temporal_template = """   - Normalize to ISO 8601 format:
     * Dates: YYYY-MM-DD
     * Datetimes: YYYY-MM-DDTHH:MM:SS
     * Durations: P1D (days), P2M (months), P3Y (years), PT2H (hours), PT30M (minutes), PT45S (seconds)
     * Intervals: start/end (e.g., 2025-10-27T11:00:00/2025-10-27T19:00:00)"""
            instruction_parts.append(temporal_template)

            if norm_config.get("handle_relative"):
                if self.document_metadata.get("publication_date"):
                    instruction_parts.append(
                        "   - For relative dates (e.g., 'Monday'), use document date {pub_date} as anchor".format(
                            pub_date=self.document_metadata['publication_date']
                        )
                    )
                else:
                    instruction_parts.append("   - For relative dates, use most recent occurrence")

            if norm_config.get("default_year") == "document":
                instruction_parts.append("   - For dates without years: use most recent year in document context")
        else:
            instruction_parts.append("   - Normalize to standard format")

        return "\n".join(instruction_parts) + "\n" if instruction_parts else ""

    def _get_geocoded_instructions(self, dim_config: DimensionConfig) -> str:
        """Get instructions for geocoded dimensions."""
        if not dim_config.disambiguation:
            return ""

        disamb_config = dim_config.disambiguation
        instruction_parts = []

        if disamb_config.get("use_parent_region"):
            instruction_parts.append("   - Include parent region for disambiguation (state, country)")

        if disamb_config.get("use_source_location") and self.document_metadata.get("source_location"):
            instruction_parts.append(
                "   - Consider source location: {source_location}".format(
                    source_location=self.document_metadata['source_location']
                )
            )

        # Add nearby locations context if available via extraction_context
        if self.extraction_context and self.extraction_context.enable_nearby_locations:
            nearby_context = self.extraction_context.get_nearby_locations_context()
            if nearby_context:
                instruction_parts.append("   - " + nearby_context.replace("\n", "\n   - "))

        # Add location type examples
        location_types = self._get_field_enum_values(dim_config, "location_type")
        if location_types:
            instruction_parts.append(
                "   - Location types: {types}, etc.".format(
                    types=", ".join(location_types[:5])
                )
            )

        return "\n".join(instruction_parts) + "\n" if instruction_parts else ""

    def _get_categorical_instructions(self, dim_config: DimensionConfig) -> str:
        """Get instructions for categorical dimensions."""
        # Get category values
        category_values = self._get_field_enum_values(dim_config, "category")
        if not category_values:
            return ""

        instruction_parts = ["   - Allowed categories:"]
        for cat in category_values[:10]:  # Show up to 10
            instruction_parts.append("     * {cat}".format(cat=cat))

        if len(category_values) > 10:
            instruction_parts.append("     * ... ({remaining} more)".format(
                remaining=len(category_values) - 10
            ))

        return "\n".join(instruction_parts) + "\n"

    def _get_structured_instructions(self, dim_config: DimensionConfig) -> str:
        """Get instructions for structured dimensions."""
        instruction_parts = ["   - Extract structured fields:"]

        for field in dim_config.fields:
            field_template = "     * {name} ({type}): {description}"
            instruction_parts.append(field_template.format(
                name=field.get("name"),
                type=field.get("type"),
                description=field.get("description", "")
            ))

        return "\n".join(instruction_parts) + "\n"

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
        template = """

Respond only in raw JSON. Schema:
{schema}"""
        return template.format(schema=schema_str)

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
