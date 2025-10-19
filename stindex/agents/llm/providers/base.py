"""
Base LLM client interface for STIndex.

Defines the contract that all LLM providers must implement.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, TypeVar

from loguru import logger
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class BaseLLM(ABC):
    """
    Abstract base class for LLM clients.

    All LLM providers (API-based or local) must implement this interface.
    """

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.0,
        max_retries: int = 3,
        **kwargs,
    ):
        """
        Initialize base LLM client.

        Args:
            model_name: Model identifier
            temperature: Sampling temperature
            max_retries: Max retries for validation failures
            **kwargs: Provider-specific arguments
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_retries = max_retries
        self.client = None

    @abstractmethod
    def initialize(self) -> Any:
        """
        Initialize the LLM client.

        Must return the initialized client object.
        """
        pass

    @abstractmethod
    def extract(
        self,
        response_model: type[T],
        messages: Optional[List[Dict[str, str]]] = None,
        prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> T:
        """
        Extract structured data using the LLM.

        Args:
            response_model: Pydantic model class for output
            messages: List of message dicts with role and content (preferred)
            prompt: Single prompt string (legacy, converted to messages)
            system_prompt: Optional system instruction (used with prompt param)

        Returns:
            Validated instance of response_model
        """
        pass

    def _build_messages(
        self,
        messages: Optional[List[Dict[str, str]]],
        prompt: Optional[str],
        system_prompt: Optional[str],
    ) -> List[Dict[str, str]]:
        """
        Build message list from various input formats.

        Args:
            messages: Pre-formatted messages
            prompt: Single prompt string
            system_prompt: System prompt for single prompt

        Returns:
            List of message dicts
        """
        if messages is not None:
            return messages

        result = []
        if system_prompt:
            result.append({"role": "system", "content": system_prompt})
        if prompt:
            result.append({"role": "user", "content": prompt})

        if not result:
            raise ValueError("Either messages or prompt must be provided")

        return result

    def generate_batch(
        self,
        messages_batch: List[List[Dict[str, str]]],
        response_model: type[T],
        max_tokens: int = 2048,
        temperature: Optional[float] = None,
    ) -> List[T]:
        """
        Batch extraction of structured data (default implementation: sequential).

        This is a fallback implementation for providers that don't support native batching.
        Subclasses (like HuggingFaceLLM) can override this with optimized batch processing.

        Args:
            messages_batch: List of message lists (one per sample)
            response_model: Pydantic model for structured output
            max_tokens: Maximum tokens to generate per sample
            temperature: Override temperature for this batch

        Returns:
            List of validated Pydantic model instances
        """
        logger.debug(f"Using sequential fallback for batch of {len(messages_batch)} samples")

        results = []
        for messages in messages_batch:
            try:
                result = self.extract(
                    response_model=response_model,
                    messages=messages,
                )
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to extract sample: {e}")
                results.append(None)

        return results
