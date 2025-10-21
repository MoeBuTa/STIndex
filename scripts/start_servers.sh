#!/bin/bash
# Unified HuggingFace server management script
# Automatically detects GPUs and starts appropriate number of servers

set -e  # Exit on error

# Configuration with defaults
BASE_PORT=${STINDEX_HF_BASE_PORT:-8001}
CONFIG=${STINDEX_CONFIG:-hf}
NUM_GPUS=${STINDEX_NUM_GPUS:-auto}
PID_FILE=${STINDEX_PID_FILE:-logs/hf_servers.pids}

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== HuggingFace Server Manager ===${NC}"
echo ""

# Create logs directory
mkdir -p logs

# Auto-detect GPUs if needed
if [ "$NUM_GPUS" = "auto" ]; then
    if command -v nvidia-smi &> /dev/null; then
        DETECTED_GPUS=$(nvidia-smi --list-gpus | wc -l)
        if [ $DETECTED_GPUS -gt 0 ]; then
            NUM_GPUS=$DETECTED_GPUS
            echo -e "${GREEN}✓ Auto-detected $NUM_GPUS GPU(s)${NC}"
        else
            NUM_GPUS=1
            echo -e "${YELLOW}⚠ No GPUs detected, starting 1 server (CPU mode)${NC}"
        fi
    else
        NUM_GPUS=1
        echo -e "${YELLOW}⚠ nvidia-smi not found, starting 1 server${NC}"
    fi
fi

echo ""
echo "Configuration:"
echo "  Base port: $BASE_PORT"
echo "  Config: $CONFIG"
echo "  Number of servers: $NUM_GPUS"
echo ""

# Check if servers are already running
if [ -f "$PID_FILE" ]; then
    echo -e "${YELLOW}⚠ Found existing PID file: $PID_FILE${NC}"
    echo "Servers may already be running. Stop them first with:"
    echo "  ./scripts/stop_servers.sh"
    echo ""
    read -p "Stop existing servers and continue? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        ./scripts/stop_servers.sh
        sleep 2
    else
        echo "Aborted."
        exit 1
    fi
fi

# Array to store PIDs
PIDS=()

# Start servers
echo -e "${BLUE}Starting servers...${NC}"
echo ""

for ((i=0; i<$NUM_GPUS; i++)); do
    PORT=$((BASE_PORT + i))
    GPU_ID=$i

    if [ $NUM_GPUS -eq 1 ]; then
        echo -e "${GREEN}→${NC} Starting server on port $PORT..."
        LOG_FILE="logs/server_port${PORT}.log"
    else
        echo -e "${GREEN}→${NC} Starting server on GPU $GPU_ID (port $PORT)..."
        LOG_FILE="logs/server_gpu${GPU_ID}_port${PORT}.log"
    fi

    # Set environment variables and launch server
    if [ $NUM_GPUS -eq 1 ]; then
        # Single server: let it use all GPUs or auto-detect
        STINDEX_CONFIG=$CONFIG \
        python -m uvicorn stindex.server.hf_server:app \
            --host 0.0.0.0 \
            --port $PORT \
            --log-level info \
            &> "$LOG_FILE" &
    else
        # Multi-GPU: pin each server to specific GPU
        CUDA_VISIBLE_DEVICES=$GPU_ID \
        STINDEX_CONFIG=$CONFIG \
        python -m uvicorn stindex.server.hf_server:app \
            --host 0.0.0.0 \
            --port $PORT \
            --log-level info \
            &> "$LOG_FILE" &
    fi

    PIDS+=($!)

    # Small delay to stagger startup and avoid port conflicts
    sleep 1
done

# Save PIDs to file
echo "${PIDS[@]}" > "$PID_FILE"

echo ""
echo -e "${GREEN}✓ All servers launched in background!${NC}"
echo ""
echo "Server URLs:"
for ((i=0; i<$NUM_GPUS; i++)); do
    PORT=$((BASE_PORT + i))
    if [ $NUM_GPUS -eq 1 ]; then
        echo "  http://localhost:$PORT"
    else
        echo "  GPU $i: http://localhost:$PORT"
    fi
done

echo ""
echo "PIDs: ${PIDS[@]} (saved to $PID_FILE)"
if [ $NUM_GPUS -eq 1 ]; then
    echo "Logs: logs/server_port*.log"
else
    echo "Logs: logs/server_gpu*_port*.log"
fi

echo ""
echo "Commands:"
echo "  View logs:      tail -f logs/server_*.log"
echo "  Stop servers:   ./scripts/stop_servers.sh"
echo "  Restart:        ./scripts/restart_servers.sh"
echo "  Check health:   curl http://localhost:$BASE_PORT/health"

# Wait a moment for servers to start
echo ""
echo -e "${YELLOW}Waiting for servers to initialize...${NC}"
sleep 3

# Check health of servers
echo ""
echo "Health check:"
for ((i=0; i<$NUM_GPUS; i++)); do
    PORT=$((BASE_PORT + i))
    if curl -s http://localhost:$PORT/health > /dev/null 2>&1; then
        echo -e "  Port $PORT: ${GREEN}✓ Running${NC}"
    else
        echo -e "  Port $PORT: ${YELLOW}⚠ Starting...${NC} (check logs if it doesn't come up)"
    fi
done

echo ""
echo -e "${GREEN}Done!${NC}"
