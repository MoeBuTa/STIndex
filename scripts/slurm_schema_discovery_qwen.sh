#!/bin/bash
#SBATCH --job-name=stindex-schema-qwen
#SBATCH --partition=data-inst
#SBATCH --ntasks=24
#SBATCH --mem=128G
#SBATCH --gres=gpu:h100:2
#SBATCH --time=24:00:00
#SBATCH --output=logs/schema_discovery_qwen_%j.log

# Environment setup
export VLLM_ATTENTION_BACKEND="FLASH_ATTN"
export VLLM_USE_TRITON_FLASH_ATTN="0"
export HF_HOME="/group/pmc063/wzhang/caches/huggingface"

# Activate conda environment
source ~/.bashrc
conda activate replay

# Run schema discovery using existing regenerate_schemas.py script
python scripts/regenerate_schemas.py \
    --dataset mirage \
    --full \
    --llm-provider hf_batch \
    --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --max-workers 5

echo "Schema discovery complete!"
