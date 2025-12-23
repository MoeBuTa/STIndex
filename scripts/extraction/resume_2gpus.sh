#!/bin/bash
# Resume extraction with 2 GPUs (auto-resume from previous progress)
set -e

echo "Resuming parallel corpus extraction with 2 GPUs..."

# Configuration
CORPUS_PATH="data/original/medcorp/train_textbooks_only.jsonl"
TOTAL_DOCS=125847
NUM_GPUS=2

echo "✓ Using $NUM_GPUS GPUs"
echo "✓ Total documents: $TOTAL_DOCS"

# Split work evenly between 2 workers
DOCS_PER_WORKER=$((TOTAL_DOCS / 2))
echo "  Documents per worker: $DOCS_PER_WORKER"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOGDIR="logs/extraction_2gpus_${TIMESTAMP}"
mkdir -p "$LOGDIR"

echo ""
echo "Log directory: $LOGDIR"
echo "🔄 Auto-resume enabled: Workers will continue from previous progress"
echo ""

# Worker 1: First half
WORKER_ID=1
OFFSET=0
LIMIT=$DOCS_PER_WORKER
PORT=8001
BASE_URL="http://localhost:$PORT"

echo "Starting Worker $WORKER_ID..."
echo "  Initial offset: $OFFSET"
echo "  Initial limit: $LIMIT"
echo "  Server: $BASE_URL"
echo "  Resume: Will auto-detect and continue from previous work"

nohup bash -c "
    source ~/.bashrc
    conda activate replay

    echo 'Worker $WORKER_ID: Starting extraction'
    echo '  Offset: $OFFSET'
    echo '  Limit: $LIMIT'
    echo '  Server: $BASE_URL'
    echo 'Started: \$(date)'

    python -m stindex.exe.extract_corpus \
        --config cfg/extraction/corpus_extraction_parallel.yml \
        --worker-id $WORKER_ID \
        --offset $OFFSET \
        --limit $LIMIT \
        --num-gpus $NUM_GPUS \
        --base-url $BASE_URL

    echo '✓ Worker $WORKER_ID Complete!'
    echo 'Finished: \$(date)'
" > "$LOGDIR/worker_${WORKER_ID}.log" 2>&1 &

WORKER_PID=$!
echo "  PID: $WORKER_PID"
echo "  Log: $LOGDIR/worker_${WORKER_ID}.log"
echo ""

# Worker 2: Second half
WORKER_ID=2
OFFSET=$DOCS_PER_WORKER
LIMIT=$((TOTAL_DOCS - OFFSET))
PORT=8002
BASE_URL="http://localhost:$PORT"

echo "Starting Worker $WORKER_ID..."
echo "  Initial offset: $OFFSET"
echo "  Initial limit: $LIMIT"
echo "  Server: $BASE_URL"
echo "  Resume: Will auto-detect and continue from previous work"

nohup bash -c "
    source ~/.bashrc
    conda activate replay

    echo 'Worker $WORKER_ID: Starting extraction'
    echo '  Offset: $OFFSET'
    echo '  Limit: $LIMIT'
    echo '  Server: $BASE_URL'
    echo 'Started: \$(date)'

    python -m stindex.exe.extract_corpus \
        --config cfg/extraction/corpus_extraction_parallel.yml \
        --worker-id $WORKER_ID \
        --offset $OFFSET \
        --limit $LIMIT \
        --num-gpus $NUM_GPUS \
        --base-url $BASE_URL

    echo '✓ Worker $WORKER_ID Complete!'
    echo 'Finished: \$(date)'
" > "$LOGDIR/worker_${WORKER_ID}.log" 2>&1 &

WORKER_PID=$!
echo "  PID: $WORKER_PID"
echo "  Log: $LOGDIR/worker_${WORKER_ID}.log"
echo ""

echo "✓ Both workers started in background"
echo ""
echo "Monitor progress:"
echo "  tail -f $LOGDIR/worker_*.log"
echo ""
echo "Check progress:"
echo "  wc -l data/extraction_results_parallel/corpus_extraction_worker*.jsonl"
echo ""
