#!/bin/bash
# Test the HuggingFace LLM server

SERVER_URL="${STINDEX_HF_SERVER_URL:-http://localhost:8001}"

echo "Testing HuggingFace LLM server at $SERVER_URL"
echo ""

# Test health endpoint
echo "1. Health check:"
curl -s "$SERVER_URL/health" | python -m json.tool
echo ""
echo ""

# Test generation endpoint
echo "2. Generation test:"
curl -s -X POST "$SERVER_URL/generate" \
    -H "Content-Type: application/json" \
    -d '{
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say hello in one sentence."}
        ],
        "max_tokens": 50,
        "temperature": 0.0
    }' | python -m json.tool
