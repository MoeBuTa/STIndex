#!/bin/bash
# Entry point for corpus extraction.
# Edit CONFIG below to point to your extraction config (LLM provider + corpus settings).
# Usage: bash scripts/extraction/extract_corpus.sh

set -e

CONFIG="cfg/extraction/inference/extract.yml"

# Setup logging
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOGFILE="logs/extract_corpus_${TIMESTAMP}.log"
mkdir -p logs

echo "========================================"
echo "Corpus Extraction + Metadata (Background)"
echo "========================================"
echo "Config: $CONFIG"
echo "Log: $LOGFILE"
echo "Monitor: tail -f $LOGFILE"
echo ""

# Run in background
nohup bash -c "
    echo '========================================'
    echo 'Corpus Extraction + Metadata'
    echo '========================================'
    echo 'Config:  $CONFIG'
    echo 'Started: \$(date)'
    echo ''

    python -m stindex.exe.extract_corpus \
        --config '$CONFIG'

    echo ''
    echo '✓ Corpus extraction complete!'
    echo 'Finished: \$(date)'
" > "$LOGFILE" 2>&1 &

PID=$!
echo "Process ID: $PID"
echo "✓ Started in background"
