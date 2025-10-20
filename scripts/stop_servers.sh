#!/bin/bash
# Stop all HuggingFace server instances

echo "Stopping all HuggingFace servers..."

# Find all uvicorn processes running stindex.server.hf_server
PIDS=$(ps aux | grep "uvicorn stindex.server.hf_server" | grep -v grep | awk '{print $2}')

if [ -z "$PIDS" ]; then
    echo "No HuggingFace servers found running."
else
    echo "Found servers with PIDs: $PIDS"
    echo "$PIDS" | xargs kill
    echo "All servers stopped."
fi
