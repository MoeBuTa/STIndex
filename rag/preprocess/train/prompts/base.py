"""
Base prompt class for LLM prompts
"""

from abc import ABC, abstractmethod
from typing import Any, Dict


class BasePrompt(ABC):
    """Base class for all prompt templates."""

    def __init__(self):
        """Initialize the base prompt."""
        pass

    @abstractmethod
    def get_system_prompt(self) -> str:
        """
        Get the system prompt.

        Returns:
            System prompt string
        """
        pass

    @abstractmethod
    def get_user_prompt(self, **kwargs) -> str:
        """
        Get the user prompt with formatted variables.

        Args:
            **kwargs: Variables to format into the prompt

        Returns:
            Formatted user prompt string
        """
        pass

    def format_prompt(self, **kwargs) -> Dict[str, str]:
        """
        Format both system and user prompts.

        Args:
            **kwargs: Variables to format into the prompts

        Returns:
            Dictionary with 'system_prompt' and 'user_prompt' keys
        """
        return {
            "system_prompt": self.get_system_prompt(),
            "user_prompt": self.get_user_prompt(**kwargs),
        }
