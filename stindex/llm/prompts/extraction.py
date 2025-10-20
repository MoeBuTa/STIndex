"""
Extraction prompts for spatiotemporal information.

Uses OOP style with separate methods for system, user, and assistant prompts.
Compatible with Instructor for all LLM providers.
"""

import json
from typing import Dict, List, Optional
from pydantic import BaseModel


class ExtractionPrompt:
    """
    OOP-style prompt builder for spatiotemporal extraction.

    Provides separate methods for system, user, and assistant prompts,
    allowing flexible composition of few-shot examples and multi-turn conversations.
    """
    def __init__(self):
        pass

    @staticmethod
    def system_prompt() -> str:
        """Generate the system prompt with extraction instructions."""
        return """You are a precise JSON extraction bot. Your ONLY output must be valid JSON.

CRITICAL RULES:
- Output ONLY the JSON object, nothing else
- NO explanations, NO reasoning, NO extra text
- Start your response with { and end with }
- Do not write "Here is the JSON" or similar phrases

Task:
1. Find ALL temporal expressions and normalize to ISO 8601:
   - Dates: YYYY-MM-DD
   - Datetimes: YYYY-MM-DDTHH:MM:SS
   - Durations: P1D, P2M, P3Y
   - For dates without years: use most recent year in document

2. Find ALL spatial/location mentions with parent regions:
   - Include nearby parent regions (state, country) for disambiguation
   - Example: "Broome" + "Western Australia" → parent_region: "Western Australia"

REMINDER: Return ONLY valid JSON, nothing else."""

    @staticmethod
    def user_prompt(text: str) -> str:
        """
        Generate the user prompt for extraction request.

        Args:
            text: Input text to extract spatiotemporal information from

        Returns:
            User prompt string with the text to analyze
        """
        return f"{text}"

    @staticmethod
    def assistant_prompt_example() -> str:
        """
        Generate an example assistant response for few-shot learning.

        Returns:
            JSON-formatted example extraction output
        """
        return """{
  "temporal_mentions": [
    {
      "text": "March 15, 2022",
      "normalized": "2022-03-15",
      "temporal_type": "date"
    },
    {
      "text": "March 17",
      "normalized": "2022-03-17",
      "temporal_type": "date"
    }
  ],
  "spatial_mentions": [
    {
      "text": "Broome",
      "location_type": "city",
      "parent_region": "Western Australia"
    },
    {
      "text": "Fitzroy Crossing",
      "location_type": "city",
      "parent_region": "Western Australia"
    }
  ]
}"""

    @staticmethod
    def user_prompt_example() -> str:
        """
        Generate the example user prompt for few-shot learning.

        Returns:
            Example input text for demonstration
        """
        return "On March 15, 2022, a cyclone hit Broome, Western Australia and moved to Fitzroy Crossing by March 17."

    @classmethod
    def build_messages(cls, text: str, use_few_shot: bool = False) -> List[Dict[str, str]]:
        """
        Build the complete message list for extraction.

        Args:
            text: Input text to extract from
            use_few_shot: Whether to include few-shot example

        Returns:
            List of message dicts with role and content
        """
        messages = [
            {"role": "system", "content": cls.system_prompt()}
        ]

        if use_few_shot:
            # Add few-shot example
            messages.extend([
                {"role": "user", "content": cls.user_prompt_example()},
                {"role": "assistant", "content": cls.assistant_prompt_example()},
                {"role": "user", "content": text}
            ])
        else:
            # Simple extraction without example
            messages.append({"role": "user", "content": cls.user_prompt(text)})

        return messages

    @staticmethod
    def get_json_instruction(schema: Dict) -> str:
        """Get JSON formatting instruction for structured output."""
        schema_str = json.dumps(schema, indent=2)
        return f"""

Respond only in raw JSON. Schema:
{schema_str}"""

    @classmethod
    def build_messages_with_schema(
        cls,
        text: str,
        response_model: type[BaseModel],
        use_few_shot: bool = False
    ) -> List[Dict[str, str]]:
        """
        Build messages with JSON schema instruction (for HuggingFace models).

        Args:
            text: Input text to extract from
            response_model: Pydantic model class for schema
            use_few_shot: Whether to include few-shot example

        Returns:
            List of message dicts with schema instruction
        """
        messages = cls.build_messages(text, use_few_shot)

        # Add schema instruction to the last user message
        schema = response_model.model_json_schema()
        json_instruction = cls.get_json_instruction(schema)
        messages[-1]["content"] += json_instruction

        return messages
