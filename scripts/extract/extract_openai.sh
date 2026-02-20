#!/bin/bash
# Extract spatiotemporal indices using OpenAI API.
# Usage: bash scripts/extract_openai.sh "Your text here"
set -e

MODEL="gpt-4o-mini"
TEMPERATURE=0.0
MAX_TOKENS=2048
TEXT="${1:-}"

if [ -z "$TEXT" ]; then
    echo "Usage: $0 \"<text to extract from>\""
    exit 1
fi

stindex extract "$TEXT" \
    --config openai \
    --model "$MODEL" \
    --temperature "$TEMPERATURE" \
    --max-tokens "$MAX_TOKENS"
