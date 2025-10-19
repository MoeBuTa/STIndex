"""
HuggingFace Transformers LLM client.

Uses transformers library for local model inference with custom JSON extraction.
"""

import json
import re
from typing import Any, Dict, List, Optional, TypeVar

import torch
from loguru import logger
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import BaseLLM

T = TypeVar("T", bound=BaseModel)


class HuggingFaceLLM(BaseLLM):
    """
    HuggingFace Transformers LLM client.

    Note: Instructor doesn't natively support HuggingFace transformers.
    This custom wrapper provides similar functionality to Instructor's
    JSON mode, with multi-strategy JSON extraction and Pydantic validation.
    """

    def __init__(
        self,
        model_name: str,
        temperature: float = 0.0,
        max_retries: int = 3,
        device: str = "auto",
        torch_dtype: str = "float16",
        **kwargs,
    ):
        """
        Initialize HuggingFace LLM client.

        Args:
            model_name: HuggingFace model ID or local path
            temperature: Sampling temperature
            max_retries: Max retries for validation failures
            device: Device placement ("auto", "cuda", "cpu")
            torch_dtype: Data type ("float16", "float32", "bfloat16")
            **kwargs: Additional arguments
        """
        super().__init__(model_name, temperature, max_retries, **kwargs)
        self.device = device
        self.torch_dtype_str = torch_dtype

        logger.info(f"Initializing HuggingFace LLM client with model: {model_name}")
        self.tokenizer = None
        self.model = None
        self.client = self.initialize()

    def initialize(self) -> Any:
        """
        Initialize HuggingFace model and tokenizer.

        Returns custom wrapper that mimics Instructor's interface.
        """
        logger.info(f"Loading HuggingFace model: {self.model_name}")

        # Map dtype string to torch dtype
        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        torch_dtype = dtype_map.get(self.torch_dtype_str, torch.float16)

        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch_dtype,
            device_map=self.device,
        )
        self.model.eval()  # Set to evaluation mode

        # Return custom wrapper
        return _HuggingFaceWrapper(self.model, self.tokenizer, self.temperature)

    def extract(
        self,
        response_model: type[T],
        messages: Optional[List[Dict[str, str]]] = None,
        prompt: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> T:
        """
        Extract structured data using HuggingFace model.

        Uses custom wrapper's create method with JSON extraction.
        """
        messages = self._build_messages(messages, prompt, system_prompt)

        try:
            result = self.client.create(
                messages=messages,
                response_model=response_model,
                max_tokens=2048,
            )
            return result

        except Exception as e:
            logger.error(f"HuggingFace extraction failed: {str(e)}")
            raise

    def generate_batch(
        self,
        messages_batch: List[List[Dict[str, str]]],
        response_model: type[T],
        max_tokens: int = 2048,
        temperature: Optional[float] = None,
    ) -> List[T]:
        """
        Batch extraction of structured data using HuggingFace model.

        Args:
            messages_batch: List of message lists (one per sample)
            response_model: Pydantic model for structured output
            max_tokens: Maximum tokens to generate per sample
            temperature: Override temperature for this batch

        Returns:
            List of validated Pydantic model instances
        """
        logger.info(f"Processing batch of {len(messages_batch)} samples")

        # Format all prompts
        prompts = []
        schema = response_model.model_json_schema()
        schema_str = json.dumps(schema, indent=2)

        for messages in messages_batch:
            prompt = self.client._format_messages(messages)
            prompt += f"\n\nRespond with valid JSON matching this exact schema:\n{schema_str}\n"
            prompt += "\nIMPORTANT: Return ONLY valid JSON, no additional text or explanations. Ensure all required fields are included."
            prompts.append(prompt)

        # Tokenize with padding (left padding for decoder-only models)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        ).to(self.model.device)

        # Store prompt lengths for each sample
        prompt_lengths = inputs["attention_mask"].sum(dim=1).tolist()

        # Generate batch
        gen_temperature = temperature if temperature is not None else self.temperature
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=gen_temperature if gen_temperature > 0 else 0.01,
                do_sample=gen_temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Extract completions (remove prompt portion)
        results = []
        for idx, output in enumerate(outputs):
            # Remove prompt tokens
            completion_ids = output[prompt_lengths[idx]:]
            generated_text = self.tokenizer.decode(
                completion_ids, skip_special_tokens=True
            )

            # Extract and validate JSON
            try:
                result = self.client._extract_json(generated_text, response_model)
                results.append(result)
            except Exception as e:
                logger.error(
                    f"Failed to extract JSON from batch item {idx}: {str(e)}"
                )
                logger.debug(f"Generated text: {generated_text[:500]}...")
                raise

        logger.info(f"Successfully processed batch of {len(results)} samples")
        return results


class _HuggingFaceWrapper:
    """
    Internal wrapper for HuggingFace model with Instructor-inspired interface.

    Provides:
    - response_model parameter for structured outputs
    - JSON schema injection into prompt
    - Multi-strategy JSON extraction
    - Pydantic validation
    """

    def __init__(self, model, tokenizer, temperature):
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = temperature

    def create(self, messages, response_model, **kwargs):
        """Generate structured output from messages."""
        # Format messages into prompt
        prompt = self._format_messages(messages)

        # Add schema instruction
        schema = response_model.model_json_schema()
        schema_str = json.dumps(schema, indent=2)
        prompt += f"\n\nRespond with valid JSON matching this exact schema:\n{schema_str}\n"
        prompt += "\nIMPORTANT: Return ONLY valid JSON, no additional text or explanations. Ensure all required fields are included."

        # Generate text
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

        # Extract and validate JSON
        return self._extract_json(generated_text, response_model)

    def _format_messages(self, messages):
        """Format messages into a prompt string."""
        formatted = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            formatted.append(f"{role.upper()}: {content}")
        return "\n\n".join(formatted)

    def _extract_json(self, text, response_model):
        """
        Extract JSON using multiple strategies.

        Inspired by Instructor's MD_JSON mode:
        1. Parse entire output as JSON
        2. Extract from Markdown code blocks
        3. Find first complete JSON object using brace counting
        """
        # Strategy 1: Try parsing entire output
        try:
            data = json.loads(text)
            return response_model.model_validate(data)
        except json.JSONDecodeError:
            pass

        # Strategy 2: Extract from Markdown code blocks
        code_block_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
        code_match = re.search(code_block_pattern, text, re.DOTALL)
        if code_match:
            json_str = code_match.group(1).strip()
            try:
                data = json.loads(json_str)
                return response_model.model_validate(data)
            except json.JSONDecodeError as e:
                logger.debug(f"Failed to parse JSON from code block: {e}")

        # Strategy 3: Find first complete JSON object
        start_idx = text.find('{')
        if start_idx == -1:
            raise ValueError("No JSON object found in model output")

        # Use brace counter to find matching closing brace
        brace_count = 0
        json_str = None
        for i in range(start_idx, len(text)):
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    json_str = text[start_idx:i+1]
                    break

        # Final validation
        if json_str:
            try:
                data = json.loads(json_str)
                return response_model.model_validate(data)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse extracted JSON: {e}")
                logger.debug(f"Generated text: {text}...")
                logger.debug(f"Extracted JSON: {json_str}...")
                raise ValueError(f"Invalid JSON in model output: {e}")
        else:
            logger.error(f"No JSON found in output: {text}...")
            raise ValueError("No valid JSON found in model output")
