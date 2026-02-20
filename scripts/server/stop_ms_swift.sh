#!/bin/bash
# Stop HuggingFace model deployment server (MS-SWIFT)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
PID_FILE="$PROJECT_ROOT/logs/.hf_server.pid"

echo "Stopping HuggingFace deployment server..."

# Function to kill all swift deploy and VLLM processes
kill_all_swift_processes() {
    echo "Killing all swift deploy and VLLM processes..."

    # Find all swift deploy process PIDs
    SWIFT_PIDS=$(ps aux | grep -E "swift/cli/deploy\.py|swift deploy" | grep -v grep | awk '{print $2}' || true)

    # Find all VLLM worker process PIDs
    VLLM_PIDS=$(ps aux | grep -E "VLLM::Worker|vllm\.worker" | grep -v grep | awk '{print $2}' || true)

    # Combine all PIDs
    ALL_PIDS="$SWIFT_PIDS $VLLM_PIDS"
    ALL_PIDS=$(echo $ALL_PIDS | tr ' ' '\n' | sort -u | tr '\n' ' ')

    if [ -z "$ALL_PIDS" ] || [ "$ALL_PIDS" = " " ]; then
        echo "No swift deploy or VLLM processes found"
        return 0
    fi

    echo "Found processes: $ALL_PIDS"

    # Try graceful kill first
    for PID in $ALL_PIDS; do
        if [ -n "$PID" ]; then
            echo "Stopping process $PID..."
            kill "$PID" 2>/dev/null || true
        fi
    done

    # Wait for processes to terminate (max 10 seconds)
    WAIT_COUNT=0
    while [ $WAIT_COUNT -lt 10 ]; do
        SWIFT_REMAINING=$(ps aux | grep -E "swift/cli/deploy\.py|swift deploy" | grep -v grep | awk '{print $2}' || true)
        VLLM_REMAINING=$(ps aux | grep -E "VLLM::Worker|vllm\.worker" | grep -v grep | awk '{print $2}' || true)
        REMAINING="$SWIFT_REMAINING $VLLM_REMAINING"
        REMAINING=$(echo $REMAINING | tr ' ' '\n' | sort -u | tr '\n' ' ')

        if [ -z "$REMAINING" ] || [ "$REMAINING" = " " ]; then
            echo "All processes stopped gracefully"
            return 0
        fi
        sleep 1
        WAIT_COUNT=$((WAIT_COUNT + 1))
    done

    # Force kill any remaining processes
    SWIFT_REMAINING=$(ps aux | grep -E "swift/cli/deploy\.py|swift deploy" | grep -v grep | awk '{print $2}' || true)
    VLLM_REMAINING=$(ps aux | grep -E "VLLM::Worker|vllm\.worker" | grep -v grep | awk '{print $2}' || true)
    REMAINING="$SWIFT_REMAINING $VLLM_REMAINING"
    REMAINING=$(echo $REMAINING | tr ' ' '\n' | sort -u | tr '\n' ' ')

    if [ -n "$REMAINING" ] && [ "$REMAINING" != " " ]; then
        echo "Force killing remaining processes: $REMAINING"
        for PID in $REMAINING; do
            if [ -n "$PID" ]; then
                kill -9 "$PID" 2>/dev/null || true
            fi
        done
        sleep 1
    fi

    echo "All swift deploy and VLLM processes terminated"
}

# Check if PID file exists
if [ ! -f "$PID_FILE" ]; then
    echo "No PID file found at $PID_FILE"
    kill_all_swift_processes
    echo "Done."
    exit 0
fi

# Read PID from file
SERVER_PID=$(cat "$PID_FILE")
echo "PID file contains: $SERVER_PID"

# Kill all swift deploy processes (not just the parent)
kill_all_swift_processes

# Clean up PID file
rm -f "$PID_FILE"

echo "✓ HuggingFace deployment server stopped."
