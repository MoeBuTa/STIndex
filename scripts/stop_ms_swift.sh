#!/bin/bash
# Stop HuggingFace model deployment server (MS-SWIFT)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PID_FILE="$PROJECT_ROOT/logs/.hf_server.pid"

echo "Stopping HuggingFace deployment server..."

# Check if PID file exists
if [ ! -f "$PID_FILE" ]; then
    echo "No PID file found at $PID_FILE"
    echo "Trying to kill any swift deploy processes..."
    pkill -f "swift deploy" || true
    echo "Done."
    exit 0
fi

# Read PID from file
SERVER_PID=$(cat "$PID_FILE")

# Check if process is running
if ! ps -p "$SERVER_PID" > /dev/null 2>&1; then
    echo "Server with PID $SERVER_PID is not running"
    rm -f "$PID_FILE"
    echo "Cleaned up stale PID file"
    exit 0
fi

# Kill the process
echo "Stopping server with PID: $SERVER_PID"
kill "$SERVER_PID"

# Wait for process to terminate (max 10 seconds)
WAIT_COUNT=0
while ps -p "$SERVER_PID" > /dev/null 2>&1 && [ $WAIT_COUNT -lt 10 ]; do
    sleep 1
    WAIT_COUNT=$((WAIT_COUNT + 1))
done

# Force kill if still running
if ps -p "$SERVER_PID" > /dev/null 2>&1; then
    echo "Process did not terminate gracefully, force killing..."
    kill -9 "$SERVER_PID" || true
    sleep 1
fi

# Clean up PID file
rm -f "$PID_FILE"

echo "✓ HuggingFace deployment server stopped."
