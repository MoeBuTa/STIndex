#!/bin/bash
# Sequential pipeline orchestrator
# Runs schema discovery and corpus extraction sequentially
# Usage: bash scripts/run_sequential_pipeline.sh

set -e

# Setup logging
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOGFILE="logs/sequential_pipeline_${TIMESTAMP}.log"
mkdir -p logs

echo "========================================"
echo "Sequential Pipeline Orchestrator"
echo "========================================"
echo "Log: $LOGFILE"
echo "Monitor: tail -f $LOGFILE"
echo ""

# Run in background
nohup bash -c "
    set -e

    echo '========================================'
    echo 'Sequential Pipeline Orchestrator'
    echo '========================================'
    echo "Started: $(date)"
    echo ''

    # Step 1: Schema Discovery
    echo '----------------------------------------'
    echo 'Step 1/2: Schema Discovery'
    echo '----------------------------------------'
    echo "Started: $(date)"

    python -m stindex.exe.discover_schema --config cfg/discovery/textbook_schema.yml

    if [ \$? -eq 0 ]; then
        echo '✓ Schema discovery complete!'
        echo "Finished: $(date)"
    else
        echo '✗ Schema discovery failed!'
        exit 1
    fi

    echo ''

    # Step 2: Corpus Extraction
    echo '----------------------------------------'
    echo 'Step 2/2: Corpus Extraction + Metadata'
    echo '----------------------------------------'
    echo "Started: $(date)"

    python -m stindex.exe.extract_corpus --config cfg/extraction/corpus_extraction_textbook.yml

    if [ \$? -eq 0 ]; then
        echo '✓ Corpus extraction complete!'
        echo "Finished: $(date)"
    else
        echo '✗ Corpus extraction failed!'
        exit 1
    fi

    echo ''
    echo '========================================'
    echo '✓ All 2 steps completed successfully!'
    echo '========================================'
    echo "Finished: $(date)"

" > "$LOGFILE" 2>&1 &

PID=$!
echo "Process ID: $PID"
echo "✓ Sequential pipeline started in background"
echo ""
echo "The pipeline will run 2 steps automatically:"
echo "  1. Schema Discovery (from all MIRAGE questions)"
echo "  2. Corpus Extraction (using discovered schema)"
