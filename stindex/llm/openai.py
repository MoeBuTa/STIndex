"""OpenAI LLM provider for STIndex."""

import os

from openai import OpenAI

from stindex.llm.base import LLMClient


class OpenAIClient(LLMClient):
    def __init__(self, model="gpt-4o-mini", temperature=0.0, max_tokens=2048):
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content
