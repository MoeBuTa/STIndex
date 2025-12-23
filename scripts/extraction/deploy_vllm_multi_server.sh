#!/bin/bash
set -e

echo "Deploying 4 separate vLLM servers (one per GPU)..."

# Flash Attention settings
export VLLM_ATTENTION_BACKEND="FLASH_ATTN"
export VLLM_USE_TRITON_FLASH_ATTN="0"

# Detect number of GPUs
NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "✓ Detected $NUM_GPUS GPUs"
echo "  Deploying $NUM_GPUS separate vLLM servers"

# Load config parameters from YAML
read -r MODEL GPU_MEM MAX_LEN <<< $(python3 << EOF
import yaml
with open("cfg/extraction/inference/hf_parallel.yml") as f:
    cfg = yaml.safe_load(f)
dep = cfg["deployment"]
vllm = dep.get("vllm", {})
print(f"{dep['model']} {vllm.get('gpu_memory_utilization', 0.85)} {vllm.get('max_model_len', 32768)}")
EOF
)

# Setup logging
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOGDIR="logs/vllm_multi_${TIMESTAMP}"
mkdir -p "$LOGDIR"

echo "Configuration:"
echo "  Model: $MODEL"
echo "  GPUs: $NUM_GPUS separate servers"
echo "  GPU Memory: $GPU_MEM"
echo "  Max Length: $MAX_LEN"
echo ""

# Deploy separate server for each GPU
for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
    PORT=$((8001 + GPU_ID))

    echo "Starting vLLM Server $((GPU_ID + 1))..."
    echo "  GPU: $GPU_ID"
    echo "  Port: $PORT"

    # Build command for single GPU server
    CMD="CUDA_VISIBLE_DEVICES=$GPU_ID swift deploy \
        --model $MODEL \
        --port $PORT \
        --infer_backend vllm \
        --use_hf \
        --vllm_gpu_memory_utilization $GPU_MEM \
        --vllm_max_model_len $MAX_LEN"

    # Start server in background
    nohup bash -c "
        # Activate conda environment
        source ~/.bashrc
        conda activate replay

        echo 'vLLM Server $((GPU_ID + 1))'
        echo 'GPU: $GPU_ID'
        echo 'Port: $PORT'
        echo 'Model: $MODEL'
        echo \"Started: \$(date)\"
        echo ''

        $CMD

        echo '✓ Server stopped'
        echo \"Finished: \$(date)\"
    " > "$LOGDIR/server_$((GPU_ID + 1)).log" 2>&1 &

    SERVER_PID=$!
    echo "  PID: $SERVER_PID"
    echo "  Log: $LOGDIR/server_$((GPU_ID + 1)).log"
    echo ""

    # Wait a bit before starting next server
    sleep 2
done

echo "✓ All $NUM_GPUS vLLM servers starting..."
echo ""
echo "Logs directory: $LOGDIR"
echo ""
echo "Wait for servers to initialize (~2-3 minutes each)..."
echo "Check status:"
echo "  curl http://localhost:8001/v1/models  # Server 1"
echo "  curl http://localhost:8002/v1/models  # Server 2"
echo "  curl http://localhost:8003/v1/models  # Server 3"
echo "  curl http://localhost:8004/v1/models  # Server 4"
echo ""
echo "Monitor logs:"
echo "  tail -f $LOGDIR/server_*.log"
