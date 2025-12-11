"""
RAG Generator Implementation.

Generates answers using LLMs with retrieved context.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from loguru import logger

from .prompts import RAGPromptTemplates


@dataclass
class GeneratorConfig:
    """Configuration for RAG generator."""
    # LLM provider settings
    provider: str = "openai"  # openai, anthropic
    model: str = "gpt-4o-mini"
    api_key: Optional[str] = None  # Uses env var if not provided

    # Generation settings
    temperature: float = 0.0
    max_tokens: int = 512
    top_p: float = 1.0

    # Prompt settings
    system_prompt: Optional[str] = None
    prompt_template: str = "default"
    include_citations: bool = True

    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0


@dataclass
class GenerationResult:
    """Result of answer generation."""
    answer: str
    question: str
    context_used: str
    model: str
    provider: str
    usage: Dict[str, int] = field(default_factory=dict)
    raw_response: Optional[str] = None
    citations: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "question": self.question,
            "model": self.model,
            "provider": self.provider,
            "usage": self.usage,
            "citations": self.citations,
            **self.metadata,
        }


class RAGGenerator:
    """
    RAG answer generator using LLM APIs.

    Supports OpenAI and Anthropic APIs with configurable prompts.
    """

    def __init__(
        self,
        config: Optional[GeneratorConfig] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """
        Initialize RAG generator.

        Args:
            config: Full configuration object
            provider: Override provider
            model: Override model
            api_key: Override API key
        """
        self.config = config or GeneratorConfig()

        # Override settings if provided
        if provider:
            self.config.provider = provider
        if model:
            self.config.model = model
        if api_key:
            self.config.api_key = api_key

        # Initialize LLM client
        self._init_client()

        # Load prompt templates
        self.prompts = RAGPromptTemplates()

        logger.info(f"RAGGenerator initialized: provider={self.config.provider}, model={self.config.model}")

    def _init_client(self) -> None:
        """Initialize LLM client based on provider."""
        if self.config.provider == "openai":
            self._init_openai()
        elif self.config.provider == "anthropic":
            self._init_anthropic()
        else:
            raise ValueError(f"Unknown provider: {self.config.provider}")

    def _init_openai(self) -> None:
        """Initialize OpenAI client."""
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai is required: pip install openai")

        api_key = self.config.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not provided. Set OPENAI_API_KEY environment variable.")

        self.client = OpenAI(api_key=api_key)
        self._generate_fn = self._generate_openai

    def _init_anthropic(self) -> None:
        """Initialize Anthropic client."""
        try:
            from anthropic import Anthropic
        except ImportError:
            raise ImportError("anthropic is required: pip install anthropic")

        api_key = self.config.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic API key not provided. Set ANTHROPIC_API_KEY environment variable.")

        self.client = Anthropic(api_key=api_key)
        self._generate_fn = self._generate_anthropic

    def generate(
        self,
        question: str,
        context: str,
        system_prompt: Optional[str] = None,
        **kwargs,
    ) -> GenerationResult:
        """
        Generate answer for a question using provided context.

        Args:
            question: The question to answer
            context: Retrieved context from retriever
            system_prompt: Override system prompt
            **kwargs: Additional generation parameters

        Returns:
            GenerationResult with answer and metadata
        """
        # Build prompt
        system = system_prompt or self.config.system_prompt or self.prompts.get_system_prompt()
        user_prompt = self.prompts.format_qa_prompt(question, context)

        # Merge kwargs with config
        gen_params = {
            "temperature": kwargs.get("temperature", self.config.temperature),
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "top_p": kwargs.get("top_p", self.config.top_p),
        }

        # Generate with retry
        for attempt in range(self.config.max_retries):
            try:
                response, usage = self._generate_fn(system, user_prompt, gen_params)

                # Extract answer
                answer = self._extract_answer(response)

                # Extract citations if enabled
                citations = []
                if self.config.include_citations:
                    citations = self._extract_citations(answer, context)

                return GenerationResult(
                    answer=answer,
                    question=question,
                    context_used=context,
                    model=self.config.model,
                    provider=self.config.provider,
                    usage=usage,
                    raw_response=response,
                    citations=citations,
                )

            except Exception as e:
                logger.warning(f"Generation attempt {attempt + 1} failed: {e}")
                if attempt == self.config.max_retries - 1:
                    raise
                import time
                time.sleep(self.config.retry_delay)

    def _generate_openai(
        self,
        system_prompt: str,
        user_prompt: str,
        params: Dict[str, Any],
    ) -> tuple:
        """Generate with OpenAI API."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=params["temperature"],
            max_tokens=params["max_tokens"],
            top_p=params["top_p"],
        )

        content = response.choices[0].message.content
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }

        return content, usage

    def _generate_anthropic(
        self,
        system_prompt: str,
        user_prompt: str,
        params: Dict[str, Any],
    ) -> tuple:
        """Generate with Anthropic API."""
        response = self.client.messages.create(
            model=self.config.model,
            system=system_prompt,
            max_tokens=params["max_tokens"],
            messages=[
                {"role": "user", "content": user_prompt},
            ],
            temperature=params["temperature"],
            top_p=params["top_p"],
        )

        content = response.content[0].text
        usage = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
        }

        return content, usage

    def _extract_answer(self, response: str) -> str:
        """Extract and clean answer from LLM response."""
        answer = response.strip()

        # Remove common prefixes
        prefixes_to_remove = [
            "Based on the provided context,",
            "According to the context,",
            "From the context provided,",
            "The answer is:",
            "Answer:",
        ]

        for prefix in prefixes_to_remove:
            if answer.lower().startswith(prefix.lower()):
                answer = answer[len(prefix):].strip()
                break

        return answer

    def _extract_citations(self, answer: str, context: str) -> List[str]:
        """
        Extract citations from answer based on context.

        Simple heuristic: find document references in answer.
        """
        import re

        citations = []

        # Look for [Document X] references
        doc_refs = re.findall(r'\[Document\s+(\d+)\]', answer)
        for ref in doc_refs:
            citations.append(f"Document {ref}")

        return citations

    def generate_batch(
        self,
        questions: List[str],
        contexts: List[str],
        **kwargs,
    ) -> List[GenerationResult]:
        """
        Generate answers for multiple questions.

        Args:
            questions: List of questions
            contexts: List of contexts (one per question)
            **kwargs: Additional generation parameters

        Returns:
            List of GenerationResult
        """
        if len(questions) != len(contexts):
            raise ValueError("Number of questions must match number of contexts")

        results = []
        for question, context in zip(questions, contexts):
            try:
                result = self.generate(question, context, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to generate answer for: {question[:50]}... - {e}")
                results.append(GenerationResult(
                    answer=f"Error: {str(e)}",
                    question=question,
                    context_used=context,
                    model=self.config.model,
                    provider=self.config.provider,
                ))

        return results

    def generate_with_reasoning(
        self,
        question: str,
        context: str,
        **kwargs,
    ) -> GenerationResult:
        """
        Generate answer with explicit reasoning steps.

        Useful for multi-hop questions that require step-by-step reasoning.
        """
        system_prompt = self.prompts.get_reasoning_system_prompt()
        user_prompt = self.prompts.format_reasoning_prompt(question, context)

        return self.generate(
            question=question,
            context=context,
            system_prompt=system_prompt,
            **kwargs,
        )


class MockGenerator:
    """
    Mock generator for testing without API calls.
    """

    def __init__(self):
        """Initialize mock generator."""
        self.call_count = 0

    def generate(
        self,
        question: str,
        context: str,
        **kwargs,
    ) -> GenerationResult:
        """Generate mock answer."""
        self.call_count += 1

        # Simple mock: return first sentence of context
        answer = context.split('.')[0] + "." if context else "No context provided."

        return GenerationResult(
            answer=answer,
            question=question,
            context_used=context,
            model="mock",
            provider="mock",
            usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        )
