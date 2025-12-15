#!/bin/bash
#SBATCH --job-name=stindex-schema-qwen-test
#SBATCH --partition=data-inst
#SBATCH --ntasks=24
#SBATCH --mem=128G
#SBATCH --gres=gpu:h100:2
#SBATCH --time=4:00:00
#SBATCH --output=logs/schema_discovery_qwen_test_%j.log

# Environment setup
export VLLM_ATTENTION_BACKEND="FLASH_ATTN"
export VLLM_USE_TRITON_FLASH_ATTN="0"
export HF_HOME="/group/pmc063/wzhang/caches/huggingface"

# Activate conda environment
source ~/.bashrc
conda activate replay

# Run test mode (3 clusters) using existing regenerate_schemas.py script
python scripts/regenerate_schemas.py \
    --dataset mirage \
    --test \
    --llm-provider hf_batch \
    --model Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --max-workers 3

echo "Test complete!"
