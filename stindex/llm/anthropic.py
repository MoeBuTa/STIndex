"""Anthropic LLM provider for STIndex extraction."""

import asyncio
import os
from typing import Any, Dict, List

from anthropic import Anthropic, AsyncAnthropic
from loguru import logger

from stindex.llm.response.models import LLMResponse, TokenUsage


class AnthropicLLM:
    """Anthropic LLM client for generation tasks."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize Anthropic LLM client.

        Args:
            config: Configuration dictionary with:
                - model_name: Model identifier (e.g., "claude-3-5-sonnet-20241022")
                - temperature: Sampling temperature
                - max_tokens: Maximum tokens to generate
                - api_key: Optional API key (defaults to ANTHROPIC_API_KEY env var)
        """
        self.config = config
        api_key = config.get("api_key") or os.getenv("ANTHROPIC_API_KEY")

        if not api_key:
            raise ValueError("Anthropic API key required (set ANTHROPIC_API_KEY env var or provide via config)")

        self.client = Anthropic(api_key=api_key)
        self.async_client = AsyncAnthropic(api_key=api_key)
        self.model_name = config.get("model_name", "claude-3-5-sonnet-20241022")

        logger.info(f"✓ Anthropic client initialized with model: {self.model_name}")

    def generate(self, messages: List[Dict[str, str]]) -> LLMResponse:
        """
        Generate completion from messages.

        Args:
            messages: List of message dicts with 'role' and 'content' keys
                     Note: Anthropic requires separate system message

        Returns:
            LLMResponse with standardized structure
        """
        try:
            # Anthropic requires system message to be separate from messages
            system_message = None
            filtered_messages = []

            for msg in messages:
                if msg["role"] == "system":
                    system_message = msg["content"]
                else:
                    filtered_messages.append(msg)

            # Build request parameters
            kwargs = {
                "model": self.model_name,
                "messages": filtered_messages,
                "temperature": self.config.get("temperature", 0.0),
                "max_tokens": self.config.get("max_tokens", 2048),
            }

            if system_message:
                kwargs["system"] = system_message

            response = self.client.messages.create(**kwargs)

            # Extract text content from response
            content = ""
            for block in response.content:
                if hasattr(block, "text"):
                    content += block.text

            # Extract usage metadata
            usage = None
            if response.usage:
                usage = TokenUsage(
                    prompt_tokens=response.usage.input_tokens,
                    completion_tokens=response.usage.output_tokens,
                    total_tokens=response.usage.input_tokens + response.usage.output_tokens,
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
            logger.error(f"Anthropic generation failed: {str(e)}")
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
        """Internal async method for batch generation."""
        async def generate_one(messages: List[Dict[str, str]]) -> LLMResponse:
            """Generate completion for a single message list."""
            try:
                # Separate system message from messages
                system_message = None
                filtered_messages = []

                for msg in messages:
                    if msg["role"] == "system":
                        system_message = msg["content"]
                    else:
                        filtered_messages.append(msg)

                # Build request parameters
                kwargs = {
                    "model": self.model_name,
                    "messages": filtered_messages,
                    "temperature": temperature if temperature is not None else self.config.get("temperature", 0.0),
                    "max_tokens": max_tokens or self.config.get("max_tokens", 2048),
                }

                if system_message:
                    kwargs["system"] = system_message

                response = await self.async_client.messages.create(**kwargs)

                # Extract text content from response
                content = ""
                for block in response.content:
                    if hasattr(block, "text"):
                        content += block.text

                # Extract usage metadata
                usage = None
                if response.usage:
                    usage = TokenUsage(
                        prompt_tokens=response.usage.input_tokens,
                        completion_tokens=response.usage.output_tokens,
                        total_tokens=response.usage.input_tokens + response.usage.output_tokens,
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
                logger.error(f"Anthropic batch generation failed for one request: {str(e)}")
                return LLMResponse(
                    model=self.model_name,
                    input=messages,
                    status="error",
                    error_msg=str(e),
                    success=False,
                )

        # Run all requests concurrently
        tasks = [generate_one(messages) for messages in messages_batch]
        results = await asyncio.gather(*tasks)

        logger.info(f"Successfully generated {len(results)} completions in parallel")
        return results
