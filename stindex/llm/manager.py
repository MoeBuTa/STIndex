"""LLM provider manager for STIndex."""

from typing import Any, Dict, List

from loguru import logger

from stindex.llm.response.models import LLMResponse


class LLMManager:
    """
    Manager class for LLM provider selection and instantiation.

    Handles provider-specific configuration and initialization.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize LLM manager.

        Args:
            config: Configuration dictionary with:
                - llm_provider: "openai", "anthropic", or "hf"
                - model_name: Model identifier
                - temperature: Sampling temperature
                - max_tokens: Maximum tokens to generate
                - (HF-specific) base_url: URL of HuggingFace/MS-SWIFT deployment server
        """
        self.config = config
        self.provider_name = config.get("llm_provider", "openai")
        self.provider = self._create_provider()

    def _create_provider(self):
        """
        Create LLM provider instance based on configuration.

        Returns:
            Configured LLM provider instance (OpenAILLM, AnthropicLLM, or MSSwiftLLM)

        Raises:
            ValueError: If provider is not supported
        """
        # Prepare provider config
        provider_config = {
            "model_name": self.config.get("model_name", "gpt-4o-mini"),
            "temperature": self.config.get("temperature", 0.0),
            "max_tokens": self.config.get("max_tokens", 2048),
        }

        # Provider-specific kwargs
        if self.provider_name == "hf":
            provider_config["base_url"] = self.config.get("base_url", "http://localhost:8000")

        # Create provider instance
        if self.provider_name == "openai":
            from stindex.llm.openai import OpenAILLM
            logger.info(f"Creating OpenAI provider with model: {provider_config['model_name']}")
            return OpenAILLM(provider_config)

        elif self.provider_name == "anthropic":
            from stindex.llm.anthropic import AnthropicLLM
            logger.info(f"Creating Anthropic provider with model: {provider_config['model_name']}")
            return AnthropicLLM(provider_config)

        elif self.provider_name == "hf":
            from stindex.llm.ms_swift import MSSwiftLLM
            logger.info(f"Creating HuggingFace (MS-SWIFT) provider with model: {provider_config['model_name']}")
            return MSSwiftLLM(provider_config)

        else:
            raise ValueError(
                f"Unsupported provider: {self.provider_name}. "
                f"Supported providers: openai, anthropic, hf"
            )

    def generate(self, messages: List[Dict[str, str]]) -> LLMResponse:
        """
        Generate completion using the configured provider.

        Args:
            messages: List of message dicts with 'role' and 'content' keys

        Returns:
            LLMResponse with standardized structure
        """
        return self.provider.generate(messages)

    def generate_batch(
        self,
        messages_batch: List[List[Dict[str, str]]],
        max_tokens: int = None,
        temperature: float = None,
    ) -> List[LLMResponse]:
        """
        Batch generation using provider's native batch method.

        All providers now support batch generation with async parallel requests.

        Args:
            messages_batch: List of message lists (one per sample)
            max_tokens: Maximum tokens to generate per sample
            temperature: Override temperature for this batch

        Returns:
            List of LLMResponse objects
        """
        return self.provider.generate_batch(
            messages_batch=messages_batch,
            max_tokens=max_tokens,
            temperature=temperature,
        )
