#!/bin/bash
# Extract spatiotemporal indices using HuggingFace model via MS-SWIFT server.
# Requires the server to be running: bash scripts/deploy_ms_swift.sh
# Usage: bash scripts/extract_hf.sh "Your text here"
set -e

MODEL="Qwen3-4B-Instruct-2507"
BASE_URL="http://localhost:8001"
TEMPERATURE=0.0
MAX_TOKENS=4096
TEXT="${1:-}"

if [ -z "$TEXT" ]; then
    echo "Usage: $0 \"<text to extract from>\""
    exit 1
fi

stindex extract "$TEXT" \
    --config hf \
    --model "$MODEL" \
    --base-url "$BASE_URL" \
    --temperature "$TEMPERATURE" \
    --max-tokens "$MAX_TOKENS"
