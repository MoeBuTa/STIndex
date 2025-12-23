#!/usr/bin/env bash
#
# Download and preprocess StatPearls corpus from NCBI
#
# Usage:
#   bash scripts/download_medcorp_full.sh
#
# This script:
# 1. Downloads StatPearls from NCBI FTP (~6.2 GB)
# 2. Extracts and preprocesses to JSONL format
# 3. Regenerates train.jsonl with Textbooks + StatPearls (427K docs)

set -e  # Exit on error

echo "========================================================================"
echo "MedCorp Full Download: Textbooks + StatPearls"
echo "========================================================================"
echo ""

# Step 1: Download StatPearls from NCBI
echo "[Step 1/3] Downloading StatPearls from NCBI FTP..."
python -m rag.preprocess.corpus.download_statpearls

echo ""
echo "========================================================================"

# Step 2: Preprocess StatPearls to JSONL
echo "[Step 2/3] Preprocessing StatPearls NXML files..."
python -m rag.preprocess.corpus.preprocess_statpearls

echo ""
echo "========================================================================"

# Step 3: Regenerate train.jsonl with both corpora
echo "[Step 3/3] Regenerating train.jsonl with Textbooks + StatPearls..."
python -m rag.preprocess.train.pikerag.data_process.main \
    --dataset medcorp \
    --output data/original/medcorp/train.jsonl

echo ""
echo "========================================================================"
echo "✓ MedCorp full corpus ready!"
echo ""
echo "Verify with:"
echo "  wc -l data/original/medcorp/train.jsonl"
echo "  # Expected: 427,049 lines (125,847 textbooks + 301,202 statpearls)"
echo ""
echo "  cat data/original/medcorp/train.jsonl | jq -r '.metadata.source_corpus' | sort | uniq -c"
echo "  # Expected: counts for 'textbooks' and 'statpearls'"
echo ""
echo "Next steps:"
echo "  1. Build FAISS indices: python -m rag.preprocess.train.ingestion.vector_ingest"
echo "  2. Compute embeddings for RRF-4 retriever"
echo "========================================================================"
