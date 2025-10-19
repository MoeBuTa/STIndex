"""
Base prompt templates.
"""

from typing import List, Dict


class BasePrompt:
    """Base class for prompt templates."""

    def __init__(self, system: str, template: str, examples: List[Dict] = None):
        """
        Initialize prompt.

        Args:
            system: System message
            template: User message template
            examples: Few-shot examples
        """
        self.system = system
        self.template = template
        self.examples = examples or []

    def format(self, **kwargs) -> str:
        """
        Format template with variables.

        Args:
            **kwargs: Template variables

        Returns:
            Formatted prompt
        """
        return self.template.format(**kwargs)

    def with_examples(self, n_examples: int = 3) -> str:
        """
        Add few-shot examples to template.

        Args:
            n_examples: Number of examples to include

        Returns:
            Template with examples
        """
        if not self.examples:
            return self.template

        examples_text = "\n\n".join([
            f"Example {i+1}:\nInput: {ex['input']}\nOutput: {ex['output']}"
            for i, ex in enumerate(self.examples[:n_examples])
        ])

        return f"{examples_text}\n\n{self.template}"

    def get_full_prompt(self, with_examples: bool = True, **kwargs) -> str:
        """
        Get complete prompt with system message and formatted template.

        Args:
            with_examples: Include few-shot examples
            **kwargs: Template variables

        Returns:
            Complete prompt
        """
        template = self.with_examples() if with_examples else self.template
        user_message = template.format(**kwargs)
        return f"{self.system}\n\n{user_message}"
