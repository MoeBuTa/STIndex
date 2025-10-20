#!/bin/bash
# Launch multiple HuggingFace server instances on different GPUs
# Each server runs on a separate GPU and listens on a different port

# Configuration
BASE_PORT=${STINDEX_HF_BASE_PORT:-8001}
CONFIG=${STINDEX_CONFIG:-hf}
NUM_GPUS=${STINDEX_NUM_GPUS:-$(nvidia-smi --list-gpus | wc -l)}

echo "Starting multi-GPU HuggingFace servers..."
echo "  Base port: $BASE_PORT"
echo "  Config: $CONFIG"
echo "  Number of GPUs: $NUM_GPUS"

# Array to store PIDs
PIDS=()

# Start server on each GPU
for ((i=0; i<$NUM_GPUS; i++)); do
    PORT=$((BASE_PORT + i))
    GPU_ID=$i

    echo "Starting server on GPU $GPU_ID (port $PORT)..."

    # Set environment variables and launch server
    CUDA_VISIBLE_DEVICES=$GPU_ID \
    STINDEX_CONFIG=$CONFIG \
    python -m uvicorn stindex.server.app:app \
        --host 0.0.0.0 \
        --port $PORT \
        --log-level info \
        &> "logs/server_gpu${GPU_ID}_port${PORT}.log" &

    PIDS+=($!)

    # Small delay to stagger startup
    sleep 2
done

echo ""
echo "All servers launched!"
echo "Server URLs:"
for ((i=0; i<$NUM_GPUS; i++)); do
    PORT=$((BASE_PORT + i))
    echo "  GPU $i: http://localhost:$PORT"
done

echo ""
echo "PIDs: ${PIDS[@]}"
echo "Logs: logs/server_gpu*_port*.log"
echo ""
echo "To stop all servers, run:"
echo "  kill ${PIDS[@]}"
echo "Or use: ./scripts/stop_servers.sh"

# Wait for all background processes
wait
