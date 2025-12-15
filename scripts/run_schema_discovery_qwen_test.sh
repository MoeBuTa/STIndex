#!/bin/bash
# Direct execution script for schema discovery (no Slurm)
# Uses swift deploy with persistent server

# Environment setup
export VLLM_ATTENTION_BACKEND="FLASH_ATTN"
export VLLM_USE_TRITON_FLASH_ATTN="0"
export HF_HOME="/group/pmc063/wzhang/caches/huggingface"
export CUDA_VISIBLE_DEVICES=0,1
# Force vLLM to use stable V0 engine instead of experimental V1
export VLLM_USE_V1="0"

# Activate conda environment
source ~/.bashrc
conda activate replay

# Run test mode (3 clusters, sequential processing)
# Note: Make sure to deploy the server first with: ./scripts/deploy_ms_swift.sh
# Don't override --model since hf.yml already has the correct model_name
python scripts/regenerate_schemas.py \
    --dataset mirage \
    --test \
    --llm-provider hf \
    --max-workers 1 \
    2>&1 | tee logs/schema_discovery_qwen_test_$(date +%Y%m%d_%H%M%S).log

echo "Test complete!"
