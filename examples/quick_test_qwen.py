"""
Quick test to verify Qwen3-8B model can be loaded.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from stindex.llm.local_llm import LocalQwenLLM


def main():
    print("Testing Qwen3-8B model loading...")
    print("-" * 60)

    try:
        # Initialize model
        print("\n1. Loading model from HuggingFace cache...")
        llm = LocalQwenLLM(
            model_name="Qwen/Qwen3-8B",
            device="auto",
            temperature=0.0,
        )
        print("✓ Model loaded successfully!")

        # Test generation
        print("\n2. Testing text generation...")
        prompt = "Extract temporal information from: On March 15, 2022, a cyclone hit."
        output = llm.generate(prompt, max_tokens=100)
        print(f"✓ Generation successful!")
        print(f"\nPrompt: {prompt}")
        print(f"Output: {output[:200]}...")

        # Test structured output
        print("\n3. Testing structured JSON generation...")
        json_prompt = """Extract temporal mentions from this text and output as JSON:
Text: "On January 1, 2024, the project started."

Output format:
{
  "temporal_mentions": [
    {"text": "...", "context": "..."}
  ]
}

Output (JSON only):"""

        result = llm.generate_structured(json_prompt)
        print(f"✓ Structured output successful!")
        print(f"Result: {result}")

        print("\n" + "=" * 60)
        print("✓ All tests passed! Qwen3-8B is working correctly.")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
