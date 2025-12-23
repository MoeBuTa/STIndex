#!/bin/bash
set -e

echo "Deploying vLLM server with auto-detected GPU configuration..."

# Flash Attention settings
export VLLM_ATTENTION_BACKEND="FLASH_ATTN"
export VLLM_USE_TRITON_FLASH_ATTN="0"

# Detect number of GPUs using nvidia-smi
NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "✓ Detected $NUM_GPUS GPUs"
echo "  Setting tensor_parallel_size=$NUM_GPUS"

# Load other config parameters from YAML
read -r MODEL PORT GPU_MEM MAX_LEN <<< $(python3 << EOF
import yaml
with open("cfg/extraction/inference/hf_parallel.yml") as f:
    cfg = yaml.safe_load(f)
dep = cfg["deployment"]
vllm = dep.get("vllm", {})
print(f"{dep['model']} {dep['port']} {vllm.get('gpu_memory_utilization', 0.85)} {vllm.get('max_model_len', 32768)}")
EOF
)

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOGDIR="logs/vllm_deploy_${TIMESTAMP}"
mkdir -p "$LOGDIR"

# Build command
CMD="swift deploy \
    --model $MODEL \
    --port $PORT \
    --infer_backend vllm \
    --use_hf \
    --vllm_tensor_parallel_size $NUM_GPUS \
    --vllm_gpu_memory_utilization $GPU_MEM \
    --vllm_max_model_len $MAX_LEN"

echo "Configuration:"
echo "  Model: $MODEL"
echo "  Port: $PORT"
echo "  Tensor Parallel Size: $NUM_GPUS"
echo "  GPU Memory: $GPU_MEM"
echo "  Max Length: $MAX_LEN"
echo ""

# Deploy vLLM server
echo "Starting vLLM server (port $PORT)..."
nohup bash -c "
    echo 'vLLM Server Deployment'
    echo 'Model: $MODEL'
    echo 'GPUs: $NUM_GPUS (tensor_parallel_size=$NUM_GPUS)'
    echo \"Started: \$(date)\"
    echo ''

    $CMD

    echo '✓ vLLM server stopped'
    echo \"Finished: \$(date)\"
" > "$LOGDIR/vllm_server.log" 2>&1 &
SERVER_PID=$!

echo "✓ vLLM server starting..."
echo "  PID: $SERVER_PID"
echo "  Port: $PORT"
echo "  Tensor Parallelism: $NUM_GPUS"
echo "  Log: $LOGDIR/vllm_server.log"

echo ""
echo "Monitor log:"
echo "  tail -f $LOGDIR/vllm_server.log"

echo ""
echo "Wait for server to be ready (~2-3 minutes)..."
echo "Check: curl http://localhost:$PORT/v1/models"

