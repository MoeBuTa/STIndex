"""Anthropic LLM provider for STIndex."""

import os

from anthropic import Anthropic

from stindex.llm.base import LLMClient


class AnthropicClient(LLMClient):
    def __init__(self, model="claude-sonnet-4-5-20250929", temperature=0.0, max_tokens=2048):
        self.client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=self.temperature,
        )
        return "".join(b.text for b in response.content if hasattr(b, "text"))
