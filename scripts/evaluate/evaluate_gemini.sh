#!/bin/bash
# Run context-aware evaluation using Google Gemini API.
# Usage: bash scripts/evaluate/evaluate_gemini.sh [--sample-limit N] [--dataset path]
set -e

MODEL="gemini-2.0-flash"
TEMPERATURE=0.0
MAX_TOKENS=2048

stindex evaluate \
    --config gemini \
    --model "$MODEL" \
    --temperature "$TEMPERATURE" \
    --max-tokens "$MAX_TOKENS" \
    "$@"
