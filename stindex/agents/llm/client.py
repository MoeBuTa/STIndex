"""
Unified LLM client using manager pattern for clean provider separation.

Provides a simple, consistent interface for structured extraction across:
- OpenAI (GPT models)
- Anthropic (Claude models)
- HuggingFace (transformers with custom wrapper)
"""

from typing import Any, Dict, List, Optional, TypeVar

from loguru import logger
from pydantic import BaseModel

from .providers import APILLM, BaseLLM, HuggingFaceLLM

T = TypeVar("T", bound=BaseModel)


class UnifiedLLMClient:
    """
    Unified LLM client using manager pattern.

    Delegates to appropriate provider-specific client based on configuration.
    """

    def __init__(
        self,
        provider: str = "openai",
        model_name: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_retries: int = 3,
        **kwargs,
    ):
        """
        Initialize unified LLM client.

        Args:
            provider: Provider name ("openai", "anthropic", or "hf")
            model_name: Model identifier
            api_key: API key (loaded from env if not provided)
            temperature: Sampling temperature
            max_retries: Max retries for validation failures
            **kwargs: Additional provider-specific arguments
        """
        self.provider = provider.lower()
        self.model_name = model_name
        self.temperature = temperature
        self.max_retries = max_retries

        logger.info(f"Initializing {provider} LLM client with model: {model_name}")

        # Create appropriate provider client
        self.client = self._create_provider_client(api_key, **kwargs)

        logger.info(f"✓ LLM client initialized successfully")

    def _create_provider_client(self, api_key: Optional[str], **kwargs) -> BaseLLM:
        """
        Factory method to create provider-specific client.

        Args:
            api_key: API key for API providers
            **kwargs: Provider-specific arguments

        Returns:
            Provider-specific LLM client instance
        """
        if self.provider in ["openai", "anthropic"]:
            return APILLM(
                provider=self.provider,
                model_name=self.model_name,
                api_key=api_key,
                temperature=self.temperature,
                max_retries=self.max_retries,
                **kwargs,
            )

        elif self.provider == "hf":
            return HuggingFaceLLM(
                model_name=self.model_name,
                temperature=self.temperature,
                max_retries=self.max_retries,
                **kwargs,
            )

        else:
            raise ValueError(
                f"Unsupported provider: {self.provider}. "
                f"Supported providers: openai, anthropic, hf"
            )

    def extract(
        self,
        response_model: type[T],
        messages: Optional[List[Dict[str, str]]] = None,
        prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> T:
        """
        Extract structured data using the configured provider.

        Args:
            response_model: Pydantic model class for output
            messages: List of message dicts with role and content (preferred)
            prompt: Single prompt string (legacy, converted to messages)
            system_prompt: Optional system instruction (used with prompt param)

        Returns:
            Validated instance of response_model
        """
        try:
            return self.client.extract(
                response_model=response_model,
                messages=messages,
                prompt=prompt,
                system_prompt=system_prompt,
            )

        except Exception as e:
            logger.error(f"Extraction failed: {str(e)}")
            raise


def create_llm_client(config: Dict[str, Any]) -> UnifiedLLMClient:
    """
    Factory function to create LLM client from config.

    Args:
        config: Configuration dictionary with:
            - llm_provider: "openai", "anthropic", or "hf"
            - model_name: Model identifier
            - temperature: Sampling temperature
            - device: Device for HuggingFace models (optional)

    Returns:
        Configured UnifiedLLMClient instance
    """
    provider = config.get("llm_provider", "openai")
    model_name = config.get("model_name", "gpt-4o-mini")
    temperature = config.get("temperature", 0.0)

    kwargs = {}

    # Provider-specific kwargs
    if provider == "hf":
        kwargs["device"] = config.get("device", "auto")
        kwargs["torch_dtype"] = config.get("torch_dtype", "float16")

    return UnifiedLLMClient(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        **kwargs,
    )
