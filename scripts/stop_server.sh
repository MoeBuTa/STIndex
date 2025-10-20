#!/bin/bash
# Stop the HuggingFace LLM server and release GPU memory

echo "Stopping HuggingFace LLM server..."

# Find and kill the server process
PIDS=$(pgrep -f "stindex.server.hf.app")

if [ -z "$PIDS" ]; then
    echo "No HuggingFace server process found."
    exit 0
fi

echo "Found process(es): $PIDS"

# Send SIGTERM for graceful shutdown
for PID in $PIDS; do
    echo "Stopping process $PID..."
    kill "$PID"
done

# Wait for processes to stop (max 10 seconds)
echo "Waiting for graceful shutdown..."
for i in {1..10}; do
    PIDS=$(pgrep -f "stindex.server.hf.app")
    if [ -z "$PIDS" ]; then
        echo "✓ HF server stopped successfully"
        echo "✓ Model released from GPU memory"
        exit 0
    fi
    sleep 1
done

# Force kill if still running
echo "Force killing remaining processes..."
pkill -9 -f "stindex.server.hf.app"
echo "✓ HF server stopped (forced)"
echo "✓ Model released from GPU memory"
