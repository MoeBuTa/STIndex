"""
Local LLM adapter for Qwen3-8B and other HuggingFace models.
"""

import json
import os
import re
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


class LocalQwenLLM:
    """
    Local Qwen3-8B LLM wrapper compatible with LangChain interface.

    Supports structured output generation for entity extraction tasks.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        model_name: str = "Qwen/Qwen3-8B",
        device: str = "auto",
        torch_dtype: str = "float16",
        max_tokens: int = 2048,
        temperature: float = 0.0,
        trust_remote_code: bool = True,
    ):
        """
        Initialize LocalQwenLLM.

        Args:
            model_path: Local path to model, if None uses HF cache
            model_name: HuggingFace model name
            device: Device to use (cuda/cpu/auto)
            torch_dtype: Torch dtype (float16/float32/int8)
            max_tokens: Maximum generation length
            temperature: Sampling temperature
            trust_remote_code: Trust remote code for model loading
        """
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.temperature = temperature

        # Determine model path
        if model_path is None:
            # Use HuggingFace cache
            model_path = model_name
        elif not os.path.exists(model_path):
            # Try to find in HF cache
            hf_cache = os.path.expanduser("~/.cache/huggingface/hub")
            model_id = f"models--{model_name.replace('/', '--')}"
            cached_path = os.path.join(hf_cache, model_id)
            if os.path.exists(cached_path):
                model_path = cached_path
                # Find snapshot directory
                snapshots = os.path.join(cached_path, "snapshots")
                if os.path.exists(snapshots):
                    snapshot_dirs = [d for d in os.listdir(snapshots) if os.path.isdir(os.path.join(snapshots, d))]
                    if snapshot_dirs:
                        model_path = os.path.join(snapshots, snapshot_dirs[0])

        print(f"Loading model from: {model_path}")

        # Set dtype
        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        torch_dtype = dtype_map.get(torch_dtype, torch.float16)

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code
        )

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device,
            trust_remote_code=trust_remote_code
        )

        self.model.eval()

        # Generation config
        self.generation_config = GenerationConfig(
            max_new_tokens=max_tokens,
            temperature=temperature if temperature > 0 else 0.01,
            do_sample=temperature > 0,
            top_p=0.95 if temperature > 0 else None,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )

        print(f"Model loaded successfully on {device}")

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stop_sequences: Optional[List[str]] = None,
    ) -> str:
        """
        Generate text from prompt.

        Args:
            prompt: Input prompt
            max_tokens: Override max tokens
            temperature: Override temperature
            stop_sequences: Stop generation at these sequences

        Returns:
            Generated text
        """
        # Prepare input
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Update generation config
        gen_config = GenerationConfig(
            max_new_tokens=max_tokens or self.max_tokens,
            temperature=(temperature if temperature is not None else self.temperature) or 0.01,
            do_sample=(temperature or self.temperature) > 0,
            top_p=0.95 if (temperature or self.temperature) > 0 else None,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=gen_config,
            )

        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Remove prompt from output
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()

        # Handle stop sequences
        if stop_sequences:
            for stop_seq in stop_sequences:
                if stop_seq in generated_text:
                    generated_text = generated_text.split(stop_seq)[0]

        return generated_text

    def generate_structured(
        self,
        prompt: str,
        schema: Optional[Dict[str, Any]] = None,
        max_attempts: int = 3,
    ) -> Dict[str, Any]:
        """
        Generate structured JSON output.

        Args:
            prompt: Input prompt (should request JSON output)
            schema: Expected JSON schema (for validation)
            max_attempts: Number of retry attempts

        Returns:
            Parsed JSON dictionary
        """
        for attempt in range(max_attempts):
            try:
                # Generate text
                output = self.generate(prompt, stop_sequences=["```", "\n\n\n"])

                # Extract JSON from output
                json_match = self._extract_json(output)

                if json_match:
                    parsed = json.loads(json_match)
                    return parsed
                else:
                    # Try parsing entire output
                    parsed = json.loads(output)
                    return parsed

            except json.JSONDecodeError as e:
                if attempt == max_attempts - 1:
                    # Last attempt, return error structure
                    print(f"Failed to parse JSON after {max_attempts} attempts: {e}")
                    return {"error": "Failed to generate valid JSON", "raw_output": output}
                continue

        return {"error": "Max attempts reached"}

    def _extract_json(self, text: str) -> Optional[str]:
        """Extract JSON from text, handling code blocks and extra text."""
        # Try to find JSON in code blocks
        code_block_pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
        match = re.search(code_block_pattern, text, re.DOTALL)
        if match:
            return match.group(1)

        # Try to find JSON object
        json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
        matches = re.findall(json_pattern, text, re.DOTALL)
        if matches:
            # Return the longest match (likely the complete JSON)
            return max(matches, key=len)

        return None

    def invoke(self, input_dict: Dict[str, Any]) -> str:
        """
        LangChain-compatible invoke method.

        Args:
            input_dict: Dictionary with 'text' or 'prompt' key

        Returns:
            Generated text
        """
        prompt = input_dict.get("text") or input_dict.get("prompt") or str(input_dict)
        return self.generate(prompt)

    def batch_generate(
        self,
        prompts: List[str],
        max_tokens: Optional[int] = None,
        batch_size: int = 4,
    ) -> List[str]:
        """
        Generate text for multiple prompts in batches.

        Args:
            prompts: List of input prompts
            max_tokens: Override max tokens
            batch_size: Batch size for generation

        Returns:
            List of generated texts
        """
        results = []

        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]

            # Tokenize batch
            inputs = self.tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            # Generate
            gen_config = GenerationConfig(
                max_new_tokens=max_tokens or self.max_tokens,
                temperature=self.temperature or 0.01,
                do_sample=self.temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )

            with torch.no_grad():
                outputs = self.model.generate(**inputs, generation_config=gen_config)

            # Decode batch
            batch_results = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

            # Remove prompts from outputs
            for prompt, generated in zip(batch, batch_results):
                if generated.startswith(prompt):
                    generated = generated[len(prompt):].strip()
                results.append(generated)

        return results

    def __call__(self, prompt: str, **kwargs) -> str:
        """Make the class callable."""
        return self.generate(prompt, **kwargs)


class LocalLLMWrapper:
    """
    Wrapper to make LocalQwenLLM compatible with LangChain's with_structured_output.
    """

    def __init__(self, llm: LocalQwenLLM):
        self.llm = llm

    def with_structured_output(self, schema: Any):
        """Return a version that outputs structured data matching schema."""
        return StructuredOutputLLM(self.llm, schema)


class StructuredOutputLLM:
    """LLM that enforces structured output."""

    def __init__(self, llm: LocalQwenLLM, schema: Any):
        self.llm = llm
        self.schema = schema
        self.schema_name = schema.__name__ if hasattr(schema, '__name__') else "Output"

    def invoke(self, messages: Union[str, List, Dict]) -> Any:
        """Generate structured output from messages."""
        # Format prompt
        if isinstance(messages, str):
            prompt = messages
        elif isinstance(messages, dict):
            prompt = messages.get("text") or str(messages)
        elif isinstance(messages, list):
            # Handle LangChain message format
            prompt = self._format_messages(messages)
        else:
            prompt = str(messages)

        # Add JSON formatting instruction
        json_prompt = f"{prompt}\n\nOutput your response as valid JSON matching this structure. Do not include any explanatory text, only the JSON object."

        # Generate structured output
        result = self.llm.generate_structured(json_prompt)

        # Convert to schema instance if possible
        if hasattr(self.schema, 'model_validate'):
            # Pydantic v2
            return self.schema.model_validate(result)
        elif hasattr(self.schema, 'parse_obj'):
            # Pydantic v1
            return self.schema.parse_obj(result)
        else:
            # Return raw dict
            return result

    def _format_messages(self, messages: List) -> str:
        """Format LangChain messages into a prompt string."""
        formatted = []
        for msg in messages:
            if hasattr(msg, 'content'):
                content = msg.content
            elif isinstance(msg, tuple):
                role, content = msg
                content = f"[{role}]: {content}"
            else:
                content = str(msg)
            formatted.append(content)
        return "\n\n".join(formatted)
