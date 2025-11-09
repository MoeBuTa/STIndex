#!/bin/bash
# MS-SWIFT Deployment Script for STIndex
# Deploys model using MS-SWIFT with vLLM backend

set -e

# Configuration
CONFIG_FILE="cfg/ms_swift.yml"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Change to project root
cd "$PROJECT_ROOT"

echo "============================================================"
echo "MS-SWIFT Deployment (vLLM Backend)"
echo "============================================================"
echo ""

# Load configuration using Python
read -r MODEL PORT TENSOR_PARALLEL GPU_MEM TRUST_CODE DTYPE <<< $(python3 << 'EOF'
import yaml
with open("cfg/ms_swift.yml") as f:
    cfg = yaml.safe_load(f)
dep = cfg["deployment"]
vllm = dep.get("vllm", {})
print(f"{dep['model']} {dep['port']} {vllm.get('tensor_parallel_size', 1)} {vllm.get('gpu_memory_utilization', 0.9)} {str(vllm.get('trust_remote_code', True)).lower()} {vllm.get('dtype', 'auto')}")
EOF
)

echo "Configuration:"
echo "  Model: $MODEL"
echo "  Port: $PORT"
echo "  Tensor Parallel Size: $TENSOR_PARALLEL"
echo "  GPU Memory Utilization: $GPU_MEM"
echo "  Trust Remote Code: $TRUST_CODE"
echo "  Dtype: $DTYPE"
echo ""

# Build command
CMD="swift deploy \
    --model $MODEL \
    --port $PORT \
    --infer_backend vllm \
    --vllm_tensor_parallel_size $TENSOR_PARALLEL \
    --vllm_gpu_memory_utilization $GPU_MEM \
    --vllm_dtype $DTYPE"

# Add optional flags
if [ "$TRUST_CODE" = "true" ]; then
    CMD="$CMD --vllm_trust_remote_code"
fi

echo "Command:"
echo "  $CMD"
echo ""
echo "Starting MS-SWIFT deployment..."
echo ""

# Execute
exec $CMD
