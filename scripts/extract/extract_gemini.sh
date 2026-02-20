#!/bin/bash
# Extract spatiotemporal indices using Google Gemini API.
# Usage: bash scripts/extract/extract_gemini.sh "Your text here"
set -e

MODEL="gemini-2.0-flash"
TEMPERATURE=0.0
MAX_TOKENS=2048
TEXT="${1:-}"

if [ -z "$TEXT" ]; then
    echo "Usage: $0 \"<text to extract from>\""
    exit 1
fi

stindex extract "$TEXT" \
    --config gemini \
    --model "$MODEL" \
    --temperature "$TEMPERATURE" \
    --max-tokens "$MAX_TOKENS"
