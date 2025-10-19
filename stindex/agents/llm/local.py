"""
Local LLM wrapper for agentic workflows.

Wraps LocalQwenLLM for use in agents.
"""

from typing import Any, Dict, Optional

from stindex.llm.local_llm import LocalQwenLLM


class LocalLLM:
    """Local LLM wrapper for agents."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize LocalLLM.

        Args:
            config: Configuration dictionary with keys:
                - model_name: Model name (e.g., "Qwen/Qwen3-8B")
                - model_path: Optional local path
                - device: Device (cuda/cpu/auto)
                - temperature: Temperature
        """
        self.config = config
        self.llm = LocalQwenLLM(
            model_name=config.get("model_name", "Qwen/Qwen3-8B"),
            model_path=config.get("model_path"),
            device=config.get("device", "auto"),
            temperature=config.get("temperature", 0.0),
        )

    def generate(self, prompt: str) -> str:
        """
        Generate text from prompt.

        Args:
            prompt: Input prompt

        Returns:
            Generated text
        """
        return self.llm.generate(prompt)

    def generate_structured(self, prompt: str) -> Dict[str, Any]:
        """
        Generate structured JSON output.

        Args:
            prompt: Input prompt (should request JSON output)

        Returns:
            Parsed JSON dictionary
        """
        return self.llm.generate_structured(prompt)
