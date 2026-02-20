#!/bin/bash
# Run context-aware evaluation using HuggingFace model via MS-SWIFT server.
# Requires the server to be running: bash scripts/deploy_ms_swift.sh
# Usage: bash scripts/evaluate_hf.sh [--sample-limit N] [--dataset path]
set -e

MODEL="Qwen3-4B-Instruct-2507"
BASE_URL="http://localhost:8001"
TEMPERATURE=0.0
MAX_TOKENS=4096

stindex evaluate \
    --config hf \
    --model "$MODEL" \
    --base-url "$BASE_URL" \
    --temperature "$TEMPERATURE" \
    --max-tokens "$MAX_TOKENS" \
    "$@"
