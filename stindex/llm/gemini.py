"""Google Gemini LLM provider for STIndex."""

import os

from stindex.llm.base import LLMClient


class GeminiClient(LLMClient):
    def __init__(self, model="gemini-2.0-flash", temperature=0.0, max_tokens=2048):
        from google import genai
        self.client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        self.genai = genai
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model,
            contents=user_prompt,
            config=self.genai.types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
            ),
        )
        return response.text
