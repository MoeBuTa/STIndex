#!/bin/bash
# Download and format raw datasets using pikerag

set -e

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

echo "========================================================"
echo "  Downloading and Formatting RAG Datasets"
echo "========================================================"

cd "$PROJECT_ROOT/rag/preprocess/train/pikerag"

echo "Running pikerag data processor..."
python main.py data_process/config/datasets.yaml

cd "$PROJECT_ROOT"

echo ""
echo "✓ Download complete!"
echo "  Output: data/original/{hotpotqa,two_wiki,musique}/train.jsonl"
