"""LLM providers for STIndex."""

from stindex.llm.anthropic import AnthropicLLM
from stindex.llm.manager import LLMManager
from stindex.llm.openai import OpenAILLM

# MSSwiftLLM is lazy-loaded by LLMManager to avoid import errors
# when MS-SWIFT dependencies are not installed

__all__ = [
    "LLMManager",
    "OpenAILLM",
    "AnthropicLLM",
]
