"""
Extraction prompts for spatiotemporal information.

Uses OOP style with separate methods for system, user, and assistant prompts.
Compatible with Instructor for all LLM providers.
"""

from typing import Dict, List


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
        """
        Generate the system prompt with extraction instructions.

        Returns:
            System prompt string defining the LLM's role and task
        """
        return """You are an expert at extracting temporal and spatial information from text.

Your task:
1. Find ALL temporal expressions and normalize them to ISO 8601 format
   - Dates: YYYY-MM-DD
   - Datetimes: YYYY-MM-DDTHH:MM:SS
   - Durations: P1D, P2M, P3Y (ISO 8601 duration format)
   - Intervals: 2022-01-01/2022-01-31 (start/end format)
   - For dates without years: use the most recent year mentioned in the document

2. Find ALL spatial/location mentions and identify parent regions for disambiguation
   - If a parent region (state, country) is mentioned nearby, include it
   - Example: "Broome" near "Western Australia" → parent_region: "Western Australia"
   - This helps disambiguate common place names

Provide structured output with:
- temporal_mentions: List of temporal expressions with ISO 8601 normalization
- spatial_mentions: List of locations with parent regions for disambiguation"""

    @staticmethod
    def user_prompt(text: str) -> str:
        """
        Generate the user prompt for extraction request.

        Args:
            text: Input text to extract spatiotemporal information from

        Returns:
            User prompt string with the text to analyze
        """
        return f"Extract temporal and spatial information from this text:\n\n{text}"

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
        return "Extract temporal and spatial information from this text:\n\nOn March 15, 2022, a cyclone hit Broome, Western Australia and moved to Fitzroy Crossing by March 17."

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
                {"role": "user", "content": f"Now extract from this text:\n\n{text}"}
            ])
        else:
            # Simple extraction without example
            messages.append({"role": "user", "content": cls.user_prompt(text)})

        return messages
