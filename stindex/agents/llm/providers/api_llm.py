"""
API-based LLM clients (OpenAI, Anthropic).

Handles API providers using Instructor's native support.
"""

import os
from typing import Any, Dict, List, Optional, TypeVar

import instructor
from loguru import logger
from pydantic import BaseModel

from .base import BaseLLM

T = TypeVar("T", bound=BaseModel)


class APILLM(BaseLLM):
    """
    API-based LLM client for OpenAI and Anthropic.

    Uses Instructor's native integration for structured outputs.
    """

    def __init__(
        self,
        provider: str,
        model_name: str,
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_retries: int = 3,
        **kwargs,
    ):
        """
        Initialize API LLM client.

        Args:
            provider: "openai" or "anthropic"
            model_name: Model identifier
            api_key: API key (loaded from env if not provided)
            temperature: Sampling temperature
            max_retries: Max retries for validation failures
            **kwargs: Additional provider-specific arguments
        """
        super().__init__(model_name, temperature, max_retries, **kwargs)
        self.provider = provider.lower()
        self.api_key = api_key
        self.kwargs = kwargs

        logger.info(f"Initializing {provider} LLM client with model: {model_name}")
        self.client = self.initialize()

    def initialize(self) -> Any:
        """Initialize the appropriate API client."""
        if self.provider == "openai":
            return self._init_openai()
        elif self.provider == "anthropic":
            return self._init_anthropic()
        else:
            raise ValueError(f"Unsupported API provider: {self.provider}")

    def _init_openai(self) -> Any:
        """Initialize OpenAI client with Instructor."""
        import openai

        api_key = self.api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key required (set OPENAI_API_KEY env var)")

        base_client = openai.OpenAI(api_key=api_key)
        return instructor.from_openai(base_client)

    def _init_anthropic(self) -> Any:
        """Initialize Anthropic client with Instructor."""
        import anthropic

        api_key = self.api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic API key required (set ANTHROPIC_API_KEY env var)")

        base_client = anthropic.Anthropic(api_key=api_key)
        return instructor.from_anthropic(base_client)

    def extract(
        self,
        response_model: type[T],
        messages: Optional[List[Dict[str, str]]] = None,
        prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> T:
        """
        Extract structured data using API LLM.

        Uses Instructor's chat.completions.create with response_model.
        """
        messages = self._build_messages(messages, prompt, system_prompt)

        try:
            result = self.client.chat.completions.create(
                model=self.model_name,
                response_model=response_model,
                messages=messages,
                temperature=self.temperature,
                max_retries=self.max_retries,
            )
            return result

        except Exception as e:
            logger.error(f"API extraction failed: {str(e)}")
            raise
