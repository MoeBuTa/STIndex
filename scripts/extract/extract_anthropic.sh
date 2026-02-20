#!/bin/bash
# Extract spatiotemporal indices using Anthropic Claude API.
# Usage: bash scripts/extract_anthropic.sh "Your text here"
set -e

MODEL="claude-sonnet-4-5-20250929"
TEMPERATURE=0.0
MAX_TOKENS=2048
TEXT="${1:-}"

if [ -z "$TEXT" ]; then
    echo "Usage: $0 \"<text to extract from>\""
    exit 1
fi

stindex extract "$TEXT" \
    --config anthropic \
    --model "$MODEL" \
    --temperature "$TEMPERATURE" \
    --max-tokens "$MAX_TOKENS"
