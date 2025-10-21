#!/bin/bash
# Stop all HuggingFace server instances

set -e  # Exit on error

# Configuration
PID_FILE=${STINDEX_PID_FILE:-logs/hf_servers.pids}

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Stopping HuggingFace servers...${NC}"
echo ""

SERVERS_STOPPED=false

# Try to read from PID file first
if [ -f "$PID_FILE" ]; then
    echo "Reading PIDs from $PID_FILE..."
    PIDS=$(cat "$PID_FILE")

    if [ -n "$PIDS" ]; then
        echo "Stopping servers with PIDs: $PIDS"

        # Try graceful shutdown first
        kill $PIDS 2>/dev/null || true

        # Wait a moment
        sleep 2

        # Force kill if still running
        kill -9 $PIDS 2>/dev/null || true

        # Remove PID file
        rm -f "$PID_FILE"
        echo -e "${GREEN}✓ Servers stopped and PID file removed${NC}"
        SERVERS_STOPPED=true
    else
        echo "PID file is empty"
        rm -f "$PID_FILE"
    fi
else
    echo -e "${YELLOW}No PID file found at $PID_FILE${NC}"
fi

# Fallback: find all uvicorn processes running stindex.server.hf_server
echo ""
echo "Checking for any remaining HuggingFace server processes..."
REMAINING_PIDS=$(ps aux | grep "uvicorn stindex.server.hf_server" | grep -v grep | awk '{print $2}')

if [ -n "$REMAINING_PIDS" ]; then
    echo "Found remaining servers with PIDs: $REMAINING_PIDS"
    echo "$REMAINING_PIDS" | xargs kill -9 2>/dev/null || true
    echo -e "${GREEN}✓ Remaining servers stopped${NC}"
    SERVERS_STOPPED=true
else
    if [ "$SERVERS_STOPPED" = false ]; then
        echo -e "${GREEN}No HuggingFace servers found running${NC}"
    else
        echo "No remaining processes found"
    fi
fi

echo ""
echo -e "${GREEN}Done!${NC}"
