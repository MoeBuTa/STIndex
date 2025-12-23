"""OpenAI LLM provider for STIndex extraction."""

import asyncio
import os
from typing import Any, Dict, List

from loguru import logger
from openai import AsyncOpenAI, OpenAI

from stindex.llm.response.models import LLMResponse, TokenUsage


class OpenAILLM:
    """OpenAI LLM client for generation tasks."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize OpenAI LLM client.

        Args:
            config: Configuration dictionary with:
                - model_name: Model identifier (e.g., "gpt-4o-mini")
                - temperature: Sampling temperature
                - max_tokens: Maximum tokens to generate
                - api_key: Optional API key (defaults to OPENAI_API_KEY env var)
        """
        self.config = config
        api_key = config.get("api_key") or os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise ValueError("OpenAI API key required (set OPENAI_API_KEY env var or provide via config)")

        self.client = OpenAI(api_key=api_key)
        self.async_client = AsyncOpenAI(api_key=api_key)
        self.model_name = config.get("model_name", "gpt-4o-mini")

        logger.info(f"✓ OpenAI client initialized with model: {self.model_name}")

    def generate(self, messages: List[Dict[str, str]]) -> LLMResponse:
        """
        Generate completion from messages.

        Args:
            messages: List of message dicts with 'role' and 'content' keys

        Returns:
            LLMResponse with standardized structure
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.config.get("temperature", 0.0),
                max_tokens=self.config.get("max_tokens", 2048),
            )

            content = response.choices[0].message.content or ""

            # Extract usage metadata
            usage = None
            if response.usage:
                usage = TokenUsage(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens,
                )

            return LLMResponse(
                model=self.model_name,
                input=messages,
                status="processed",
                content=content,
                usage=usage,
                success=True,
            )

        except Exception as e:
            logger.error(f"OpenAI generation failed: {str(e)}")
            return LLMResponse(
                model=self.model_name,
                input=messages,
                status="error",
                error_msg=str(e),
                success=False,
            )

    def generate_batch(
        self,
        messages_batch: List[List[Dict[str, str]]],
        max_tokens: int = None,
        temperature: float = None,
    ) -> List[LLMResponse]:
        """
        Generate completions for a batch of messages using async parallel requests.

        Args:
            messages_batch: List of message lists (one per sample)
            max_tokens: Maximum tokens to generate per sample (overrides config)
            temperature: Sampling temperature (overrides config)

        Returns:
            List of LLMResponse objects
        """
        return asyncio.run(self._generate_batch_async(messages_batch, max_tokens, temperature))

    async def _generate_batch_async(
        self,
        messages_batch: List[List[Dict[str, str]]],
        max_tokens: int = None,
        temperature: float = None,
    ) -> List[LLMResponse]:
        """Internal async method for batch generation with rate limiting."""
        # Rate limit concurrent requests to avoid overwhelming the server
        max_concurrent = self.config.get("max_concurrent", 64)
        sem = asyncio.Semaphore(max_concurrent)

        async def generate_one(messages: List[Dict[str, str]]) -> LLMResponse:
            """Generate completion for a single message list with rate limiting."""
            async with sem:  # Limit concurrent requests
                try:
                    response = await self.async_client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=temperature if temperature is not None else self.config.get("temperature", 0.0),
                        max_tokens=max_tokens or self.config.get("max_tokens", 2048),
                    )

                    content = response.choices[0].message.content or ""

                    # Extract usage metadata
                    usage = None
                    if response.usage:
                        usage = TokenUsage(
                            prompt_tokens=response.usage.prompt_tokens,
                            completion_tokens=response.usage.completion_tokens,
                            total_tokens=response.usage.total_tokens,
                        )

                    return LLMResponse(
                        model=self.model_name,
                        input=messages,
                        status="processed",
                        content=content,
                        usage=usage,
                        success=True,
                    )

                except Exception as e:
                    logger.error(f"OpenAI batch generation failed for one request: {str(e)}")
                    return LLMResponse(
                        model=self.model_name,
                        input=messages,
                        status="error",
                        error_msg=str(e),
                        success=False,
                    )

        # Run all requests concurrently with error handling
        tasks = [generate_one(messages) for messages in messages_batch]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to error responses
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"OpenAI batch request {i} failed with exception: {result}")
                processed_results.append(LLMResponse(
                    model=self.model_name,
                    input=messages_batch[i],
                    status="error",
                    error_msg=str(result),
                    success=False,
                ))
            else:
                processed_results.append(result)

        logger.info(f"Generated {len(processed_results)} completions (max_concurrent={max_concurrent})")
        return processed_results
