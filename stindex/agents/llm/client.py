"""
Unified LLM client using Instructor for both HuggingFace and API models.

Provides a simple, consistent interface for structured extraction.
"""

import os
from typing import Any, Dict, List, Optional, TypeVar

import instructor
from loguru import logger
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class UnifiedLLMClient:
    """
    Unified LLM client using Instructor.

    Supports both API models (OpenAI, Anthropic) and HuggingFace models (transformers).
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

        if self.provider == "openai":
            self.client = self._init_openai(api_key, **kwargs)
        elif self.provider == "anthropic":
            self.client = self._init_anthropic(api_key, **kwargs)
        elif self.provider == "hf":
            self.client = self._init_huggingface(**kwargs)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

        logger.info(f"✓ LLM client initialized successfully")

    def _init_openai(self, api_key: Optional[str], **kwargs) -> Any:
        """Initialize OpenAI client with Instructor."""
        import openai

        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key required (set OPENAI_API_KEY env var)")

        base_client = openai.OpenAI(api_key=api_key)
        return instructor.from_openai(base_client)

    def _init_anthropic(self, api_key: Optional[str], **kwargs) -> Any:
        """Initialize Anthropic client with Instructor."""
        import anthropic

        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Anthropic API key required (set ANTHROPIC_API_KEY env var)")

        base_client = anthropic.Anthropic(api_key=api_key)
        return instructor.from_anthropic(base_client)

    def _init_huggingface(self, **kwargs) -> Any:
        """Initialize HuggingFace model with Instructor."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        device = kwargs.get("device", "auto")
        torch_dtype = kwargs.get("torch_dtype", "float16")

        logger.info(f"Loading HuggingFace model: {self.model_name}")

        # Map dtype string to torch dtype
        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        torch_dtype = dtype_map.get(torch_dtype, torch.float16)

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch_dtype,
            device_map=device,
        )
        self.model.eval()

        # For HuggingFace models, we need a custom wrapper for Instructor
        return self._create_hf_wrapper()

    def _create_hf_wrapper(self):
        """Create wrapper for HuggingFace model compatible with Instructor."""
        # HuggingFace models need a custom wrapper since Instructor doesn't
        # have native transformers support (only Ollama, llama-cpp-python)
        class HuggingFaceModelWrapper:
            def __init__(self, model, tokenizer, temperature):
                self.model = model
                self.tokenizer = tokenizer
                self.temperature = temperature

            def create(self, messages, response_model, **kwargs):
                # Format messages into prompt
                prompt = self._format_messages(messages)

                # Add instruction for JSON output
                prompt += "\n\nRespond with valid JSON matching the required schema."

                # Generate
                import torch
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=kwargs.get("max_tokens", 2048),
                        temperature=self.temperature if self.temperature > 0 else 0.01,
                        do_sample=self.temperature > 0,
                    )

                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

                # Remove prompt from output
                if generated_text.startswith(prompt):
                    generated_text = generated_text[len(prompt):].strip()

                # Parse JSON and validate with Pydantic
                import json
                import re

                # Extract JSON
                json_match = re.search(r'\{.*\}', generated_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                    data = json.loads(json_str)
                    return response_model.model_validate(data)
                else:
                    raise ValueError("No valid JSON found in model output")

            def _format_messages(self, messages):
                """Format messages into a prompt string."""
                formatted = []
                for msg in messages:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    formatted.append(f"{role.upper()}: {content}")
                return "\n\n".join(formatted)

        wrapper = HuggingFaceModelWrapper(self.model, self.tokenizer, self.temperature)

        # Patch with instructor
        return instructor.patch(
            create=wrapper.create,
            mode=instructor.Mode.JSON
        )

    def extract(
        self,
        response_model: type[T],
        messages: Optional[List[Dict[str, str]]] = None,
        prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> T:
        """
        Extract structured data using messages or prompt.

        Args:
            response_model: Pydantic model class for output
            messages: List of message dicts with role and content (preferred)
            prompt: Single prompt string (legacy, converted to messages)
            system_prompt: Optional system instruction (used with prompt param)

        Returns:
            Validated instance of response_model
        """
        # Convert old prompt format to messages if needed
        if messages is None:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            if prompt:
                messages.append({"role": "user", "content": prompt})

        if not messages:
            raise ValueError("Either messages or prompt must be provided")

        try:
            if self.provider in ["openai", "anthropic"]:
                result = self.client.chat.completions.create(
                    model=self.model_name,
                    response_model=response_model,
                    messages=messages,
                    temperature=self.temperature,
                    max_retries=self.max_retries,
                )
            else:
                # HuggingFace model
                result = self.client.create(
                    messages=messages,
                    response_model=response_model,
                    max_tokens=2048,
                )

            return result

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
    if provider == "hf":
        kwargs["device"] = config.get("device", "auto")
        kwargs["torch_dtype"] = config.get("torch_dtype", "float16")

    return UnifiedLLMClient(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        **kwargs,
    )
