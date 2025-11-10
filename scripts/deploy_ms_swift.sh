#!/bin/bash
# HuggingFace Model Deployment Script (via MS-SWIFT)
# Deploys HuggingFace model using MS-SWIFT with vLLM backend

set -e

# Configuration
CONFIG_FILE="cfg/hf.yml"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root
cd "$PROJECT_ROOT"

echo "============================================================"
echo "HuggingFace Model Deployment (MS-SWIFT + vLLM)"
echo "============================================================"
echo ""

# Load configuration using Python
read -r MODEL PORT TENSOR_PARALLEL GPU_MEM TRUST_CODE DTYPE MAX_LEN RESULT_PATH <<< $(python3 << 'EOF'
import yaml
import os

with open("cfg/hf.yml") as f:
    cfg = yaml.safe_load(f)
dep = cfg["deployment"]
vllm = dep.get("vllm", {})
result_path = dep.get("result_path", "data/output/result")

# Auto-detect GPUs if tensor_parallel_size is "auto" or not set
tensor_parallel = vllm.get('tensor_parallel_size', 1)
if tensor_parallel == "auto":
    import subprocess
    try:
        # Get number of available GPUs
        gpu_count = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True
        ).strip().split('\n')
        tensor_parallel = len(gpu_count) if gpu_count and gpu_count[0] else 1
    except Exception:
        tensor_parallel = 1

print(f"{dep['model']} {dep['port']} {tensor_parallel} {vllm.get('gpu_memory_utilization', 0.9)} {str(vllm.get('trust_remote_code', True)).lower()} {vllm.get('dtype', 'auto')} {vllm.get('max_model_len', '')} {result_path}")
EOF
)

echo "Configuration:"
echo "  Model: $MODEL"
echo "  Port: $PORT"
echo "  Tensor Parallel Size: $TENSOR_PARALLEL (GPUs)"
echo "  GPU Memory Utilization: $GPU_MEM"
echo "  Trust Remote Code: $TRUST_CODE"
echo "  Dtype: $DTYPE"
echo "  Max Model Len: $MAX_LEN"
echo "  Result Path: $RESULT_PATH"
echo ""

# Create result directory
mkdir -p "$PROJECT_ROOT/$RESULT_PATH"

# Build command
CMD="swift deploy \
    --model $MODEL \
    --port $PORT \
    --infer_backend vllm \
    --use_hf \
    --vllm_tensor_parallel_size $TENSOR_PARALLEL \
    --vllm_gpu_memory_utilization $GPU_MEM \
    --result_path $RESULT_PATH"

# Add optional parameters
if [ -n "$MAX_LEN" ]; then
    CMD="$CMD --max_model_len $MAX_LEN"
fi

# Note: dtype and trust_remote_code are handled automatically by vLLM/MS-SWIFT
# These settings don't need to be passed via CLI flags

# Setup directories and files
LOG_DIR="$PROJECT_ROOT/logs"
PID_FILE="$LOG_DIR/.hf_server.pid"
LOG_FILE="$LOG_DIR/hf_server.log"

mkdir -p "$LOG_DIR"

# Check if server is already running
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "Error: Server already running with PID $OLD_PID"
        echo "Stop it first with: ./scripts/stop_ms_swift.sh"
        exit 1
    else
        # Stale PID file, remove it
        rm -f "$PID_FILE"
    fi
fi

echo "Command:"
echo "  $CMD"
echo ""
echo "Log file: $LOG_FILE"
echo ""
echo "Starting MS-SWIFT deployment in background..."
echo ""

# Execute in background and save PID
$CMD > "$LOG_FILE" 2>&1 &
SERVER_PID=$!
echo $SERVER_PID > "$PID_FILE"

echo "Server started with PID: $SERVER_PID"
echo "Waiting for server to be ready..."
echo ""

# Wait for server to be ready (check health endpoint)
MAX_WAIT=180  # Maximum wait time in seconds (increased for model loading + CUDA graphs)
WAIT_COUNT=0
SLEEP_INTERVAL=2

while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
    # Check if process is still running
    if ! ps -p $SERVER_PID > /dev/null 2>&1; then
        echo "Error: Server process died. Check logs at: $LOG_FILE"
        rm -f "$PID_FILE"
        exit 1
    fi

    # Try to connect to the server
    if curl -s "http://localhost:$PORT/v1/models" > /dev/null 2>&1; then
        echo "✓ Server is ready and responding!"
        echo ""
        echo "Server details:"
        echo "  PID: $SERVER_PID"
        echo "  Port: $PORT"
        echo "  Model: $MODEL"
        echo "  Log: $LOG_FILE"
        echo ""
        echo "Test with: curl http://localhost:$PORT/v1/models"
        echo "Stop with: ./scripts/stop_ms_swift.sh"
        echo ""
        exit 0
    fi

    # Show progress
    if [ $((WAIT_COUNT % 10)) -eq 0 ]; then
        echo "Still waiting... (${WAIT_COUNT}s elapsed)"
    fi

    sleep $SLEEP_INTERVAL
    WAIT_COUNT=$((WAIT_COUNT + SLEEP_INTERVAL))
done

echo "Warning: Server did not respond within ${MAX_WAIT} seconds"
echo "Server may still be starting up. Check logs at: $LOG_FILE"
echo "PID: $SERVER_PID"
exit 1
