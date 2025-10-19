"""
API-based LLM wrappers (OpenAI, Anthropic) for agentic workflows.

Uses LangChain for simplicity.
"""

from typing import Any, Dict

from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI


class OpenAILLM:
    """OpenAI LLM wrapper for agents."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize OpenAI LLM.

        Args:
            config: Configuration dictionary with keys:
                - model_name: Model name (e.g., "gpt-4o-mini")
                - temperature: Temperature
        """
        self.config = config
        self.llm = ChatOpenAI(
            model=config.get("model_name", "gpt-4o-mini"),
            temperature=config.get("temperature", 0.0),
        )

    def generate(self, prompt: str) -> str:
        """Generate text from prompt."""
        response = self.llm.invoke(prompt)
        return response.content

    def with_structured_output(self, schema):
        """Get LLM with structured output."""
        return self.llm.with_structured_output(schema)


class AnthropicLLM:
    """Anthropic LLM wrapper for agents."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Anthropic LLM.

        Args:
            config: Configuration dictionary with keys:
                - model_name: Model name (e.g., "claude-3-5-sonnet-20241022")
                - temperature: Temperature
        """
        self.config = config
        self.llm = ChatAnthropic(
            model=config.get("model_name", "claude-3-5-sonnet-20241022"),
            temperature=config.get("temperature", 0.0),
        )

    def generate(self, prompt: str) -> str:
        """Generate text from prompt."""
        response = self.llm.invoke(prompt)
        return response.content

    def with_structured_output(self, schema):
        """Get LLM with structured output."""
        return self.llm.with_structured_output(schema)
