#!/bin/bash
# Direct execution script for full schema discovery (10 clusters)
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

# Run full mode (10 clusters, sequential processing with max_workers=1)
# Note: Make sure to deploy the server first with: ./scripts/deploy_ms_swift.sh
# Don't override --model since hf.yml already has the correct model_name
python scripts/regenerate_schemas.py \
    --dataset mirage \
    --full \
    --llm-provider hf \
    --max-workers 1 \
    2>&1 | tee logs/schema_discovery_qwen_full_$(date +%Y%m%d_%H%M%S).log

echo "Full schema discovery complete!"
