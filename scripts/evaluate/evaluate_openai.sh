#!/bin/bash
# Run context-aware evaluation using OpenAI API.
# Usage: bash scripts/evaluate_openai.sh [--sample-limit N] [--dataset path]
set -e

MODEL="gpt-4o-mini"
TEMPERATURE=0.0
MAX_TOKENS=2048

stindex evaluate \
    --config openai \
    --model "$MODEL" \
    --temperature "$TEMPERATURE" \
    --max-tokens "$MAX_TOKENS" \
    "$@"
