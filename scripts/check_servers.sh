#!/bin/bash
# Check status of HuggingFace servers
# Displays health status, GPU usage, and process information

set -e  # Exit on error

# Configuration with defaults
BASE_PORT=${STINDEX_HF_BASE_PORT:-8001}
NUM_GPUS=${STINDEX_NUM_GPUS:-auto}
PID_FILE=${STINDEX_PID_FILE:-logs/hf_servers.pids}

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== HuggingFace Server Status ===${NC}"
echo ""

# Auto-detect GPUs if needed
if [ "$NUM_GPUS" = "auto" ]; then
    if command -v nvidia-smi &> /dev/null; then
        DETECTED_GPUS=$(nvidia-smi --list-gpus | wc -l)
        if [ $DETECTED_GPUS -gt 0 ]; then
            NUM_GPUS=$DETECTED_GPUS
        else
            NUM_GPUS=1
        fi
    else
        NUM_GPUS=1
    fi
fi

# Check if PID file exists
if [ ! -f "$PID_FILE" ]; then
    echo -e "${YELLOW}⚠ No PID file found: $PID_FILE${NC}"
    echo "Servers may not be running."
    echo ""
    echo "To start servers, run:"
    echo "  ./scripts/start_servers.sh"
    echo ""
else
    # Read PIDs from file
    PIDS=($(cat "$PID_FILE"))
    echo -e "${CYAN}PIDs from file:${NC} ${PIDS[@]}"
    echo ""

    # Check which processes are actually running
    RUNNING_COUNT=0
    echo -e "${CYAN}Process Status:${NC}"
    for PID in "${PIDS[@]}"; do
        if ps -p $PID > /dev/null 2>&1; then
            echo -e "  PID $PID: ${GREEN}✓ Running${NC}"
            RUNNING_COUNT=$((RUNNING_COUNT + 1))
        else
            echo -e "  PID $PID: ${RED}✗ Not running${NC}"
        fi
    done
    echo ""
fi

# Check health endpoints
echo -e "${CYAN}Health Check:${NC}"
HEALTHY_COUNT=0
for ((i=0; i<$NUM_GPUS; i++)); do
    PORT=$((BASE_PORT + i))

    # Try to get health status
    HEALTH_RESPONSE=$(curl -s http://localhost:$PORT/health 2>&1 || echo "")

    if [ -n "$HEALTH_RESPONSE" ] && echo "$HEALTH_RESPONSE" | grep -q "status"; then
        # Extract status if it's JSON
        if command -v jq &> /dev/null; then
            STATUS=$(echo "$HEALTH_RESPONSE" | jq -r '.status' 2>/dev/null || echo "ok")
            MODEL=$(echo "$HEALTH_RESPONSE" | jq -r '.model_name' 2>/dev/null || echo "")
            DEVICE=$(echo "$HEALTH_RESPONSE" | jq -r '.device' 2>/dev/null || echo "")

            echo -e "  Port $PORT: ${GREEN}✓ Healthy${NC}"
            if [ -n "$MODEL" ] && [ "$MODEL" != "null" ]; then
                echo -e "    Model: $MODEL"
            fi
            if [ -n "$DEVICE" ] && [ "$DEVICE" != "null" ]; then
                echo -e "    Device: $DEVICE"
            fi
        else
            echo -e "  Port $PORT: ${GREEN}✓ Healthy${NC} (install jq for details)"
        fi
        HEALTHY_COUNT=$((HEALTHY_COUNT + 1))
    else
        echo -e "  Port $PORT: ${RED}✗ Unreachable${NC}"
    fi
done
echo ""

# Summary
echo -e "${CYAN}Summary:${NC}"
if [ -f "$PID_FILE" ]; then
    echo "  Running processes: $RUNNING_COUNT / ${#PIDS[@]}"
fi
echo "  Healthy servers: $HEALTHY_COUNT / $NUM_GPUS"
echo ""

# GPU Status (if nvidia-smi available)
if command -v nvidia-smi &> /dev/null; then
    echo -e "${CYAN}GPU Status:${NC}"
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | \
    while IFS=, read -r idx name util mem_used mem_total; do
        # Color code GPU utilization
        if [ $util -gt 80 ]; then
            COLOR=$GREEN
        elif [ $util -gt 30 ]; then
            COLOR=$YELLOW
        else
            COLOR=$NC
        fi
        echo -e "  GPU $idx ($name): ${COLOR}${util}%${NC} | Memory: ${mem_used}/${mem_total} MB"
    done
    echo ""
fi

# Check for relevant Python processes
echo -e "${CYAN}Related Processes:${NC}"
PROCESS_COUNT=$(ps aux | grep -E "uvicorn.*hf_server|stindex.server.hf_server" | grep -v grep | wc -l)
if [ $PROCESS_COUNT -gt 0 ]; then
    echo -e "  Found $PROCESS_COUNT HuggingFace server process(es):"
    ps aux | grep -E "uvicorn.*hf_server|stindex.server.hf_server" | grep -v grep | \
    awk '{printf "    PID %-6s | CPU: %-5s | MEM: %-5s | %s\n", $2, $3"%", $4"%", $11}'
else
    echo -e "  ${YELLOW}No HuggingFace server processes found${NC}"
fi
echo ""

# Quick action hints
echo -e "${CYAN}Quick Actions:${NC}"
if [ $HEALTHY_COUNT -eq $NUM_GPUS ]; then
    echo -e "  ${GREEN}✓ All servers are healthy!${NC}"
    echo ""
    echo "  Test extraction:"
    echo "    stindex extract \"March 15, 2022 in Sydney\" --config hf"
    echo ""
    echo "  View logs:"
    echo "    tail -f logs/server_*.log"
    echo ""
    echo "  Stop servers:"
    echo "    ./scripts/stop_servers.sh"
else
    echo -e "  ${YELLOW}⚠ Some servers are not responding${NC}"
    echo ""
    echo "  View logs:"
    echo "    tail -f logs/server_*.log"
    echo ""
    echo "  Restart servers:"
    echo "    ./scripts/restart_servers.sh"
    echo ""
    echo "  Stop and start fresh:"
    echo "    ./scripts/stop_servers.sh && ./scripts/start_servers.sh"
fi

echo ""
