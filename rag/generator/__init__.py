"""
RAG Generator Module.

Provides answer generation using LLMs (OpenAI, Anthropic, etc.)
with retrieved context from the retriever.

Features:
- Multiple LLM provider support
- Configurable prompts for different tasks
- Answer extraction and parsing
- Citation generation

Usage:
    from rag.generator import RAGGenerator, GeneratorConfig

    generator = RAGGenerator(config=GeneratorConfig(
        provider="openai",
        model="gpt-4o-mini",
    ))

    answer = generator.generate(
        question="What is the capital of France?",
        context="Paris is the capital and largest city of France...",
    )
"""

from .generator import RAGGenerator, GeneratorConfig, GenerationResult
from .prompts import RAGPromptTemplates

__all__ = [
    "RAGGenerator",
    "GeneratorConfig",
    "GenerationResult",
    "RAGPromptTemplates",
]
