#!/bin/bash
# Restart HuggingFace LLM server

echo "Restarting HuggingFace LLM server..."

# Stop existing server
./scripts/stop_server.sh

# Wait a moment for GPU memory to be fully released
echo "Waiting for GPU memory to be released..."
sleep 3

# Start new server
./scripts/start_server.sh "$@"
