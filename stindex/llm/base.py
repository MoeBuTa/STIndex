"""Base LLM client interface and factory for STIndex."""

from abc import ABC, abstractmethod
from typing import Dict


class LLMClient(ABC):
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate a response. Raises on failure."""
        ...


def create_client(config: Dict) -> LLMClient:
    """
    Factory function for creating LLM clients from config.

    Args:
        config: Dict with keys: llm_provider, model_name, temperature, max_tokens,
                and optionally base_url (for hf provider).

    Returns:
        Configured LLMClient instance.

    Raises:
        ValueError: If llm_provider is not recognized.
    """
    provider = config.get("llm_provider", "openai")
    model = config.get("model_name")
    temp = config.get("temperature", 0.0)
    tokens = config.get("max_tokens", 2048)

    if provider == "openai":
        from stindex.llm.openai import OpenAIClient
        return OpenAIClient(model=model, temperature=temp, max_tokens=tokens)
    elif provider == "anthropic":
        from stindex.llm.anthropic import AnthropicClient
        return AnthropicClient(model=model, temperature=temp, max_tokens=tokens)
    elif provider == "gemini":
        from stindex.llm.gemini import GeminiClient
        return GeminiClient(model=model, temperature=temp, max_tokens=tokens)
    elif provider == "hf":
        from stindex.llm.ms_swift import MSSwiftClient
        return MSSwiftClient(
            model=model,
            base_url=config.get("base_url", "http://localhost:8001"),
            temperature=temp,
            max_tokens=tokens,
        )
    else:
        raise ValueError(f"Unknown llm_provider: {provider!r}")
