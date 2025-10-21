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
                - device: Device for HuggingFace models (optional)
                - torch_dtype: Data type for HuggingFace models (optional)
                - trust_remote_code: Whether to trust remote code for HuggingFace (optional)
        """
        self.config = config
        self.provider_name = config.get("llm_provider", "openai")
        self.provider = self._create_provider()

    def _create_provider(self):
        """
        Create LLM provider instance based on configuration.

        Returns:
            Configured LLM provider instance (OpenAILLM, AnthropicLLM, or HuggingFaceLLM)

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
            provider_config["device"] = self.config.get("device", "auto")
            provider_config["torch_dtype"] = self.config.get("torch_dtype", "float16")
            provider_config["trust_remote_code"] = self.config.get("trust_remote_code", False)
            provider_config["max_input_length"] = self.config.get("max_input_length", 4096)
            provider_config["top_p"] = self.config.get("top_p", 1.0)
            # Add server configuration for client mode
            provider_config["server_url"] = self.config.get("server_url")
            provider_config["server_urls"] = self.config.get("server_urls")
            provider_config["load_balancing"] = self.config.get("load_balancing", "round_robin")

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
            from stindex.llm.hf import HuggingFaceLLM
            logger.info(f"Creating HuggingFace provider with model: {provider_config['model_name']}")
            return HuggingFaceLLM(provider_config)

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
