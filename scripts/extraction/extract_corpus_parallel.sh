#!/bin/bash
# Parallel corpus extraction across all available GPUs with auto-resume.
# Requires one vLLM server per GPU: bash scripts/server/deploy_vllm_multi_server.sh
# Edit CORPUS_PATH and CONFIG below before running.
# Usage: bash scripts/extraction/extract_corpus_parallel.sh
set -e

echo "Starting parallel corpus extraction with auto-resume..."

# Detect number of GPUs using nvidia-smi
NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "✓ Detected $NUM_GPUS GPUs"
echo "  Spawning $NUM_GPUS parallel workers"

# Count total documents in corpus
CORPUS_PATH="data/original/medcorp/train_textbooks_only.jsonl"
TOTAL_DOCS=$(wc -l < "$CORPUS_PATH")
echo "✓ Total documents: $TOTAL_DOCS"

# Calculate documents per worker
DOCS_PER_WORKER=$((TOTAL_DOCS / NUM_GPUS))
REMAINDER=$((TOTAL_DOCS % NUM_GPUS))
echo "  Documents per worker: $DOCS_PER_WORKER"
if [ $REMAINDER -gt 0 ]; then
    echo "  Last worker will process $REMAINDER additional documents"
fi

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOGDIR="logs/extraction_parallel_${TIMESTAMP}"
mkdir -p "$LOGDIR"

echo ""
echo "Log directory: $LOGDIR"
echo ""

# Spawn workers dynamically
for WORKER_ID in $(seq 1 $NUM_GPUS); do
    # Calculate offset and limit for this worker
    OFFSET=$(( (WORKER_ID - 1) * DOCS_PER_WORKER ))

    if [ $WORKER_ID -eq $NUM_GPUS ]; then
        # Last worker gets remaining documents
        LIMIT=$((DOCS_PER_WORKER + REMAINDER))
    else
        LIMIT=$DOCS_PER_WORKER
    fi

    # Each worker uses its own vLLM server port
    PORT=$((8000 + WORKER_ID))
    BASE_URL="http://localhost:$PORT"

    echo "Starting Worker $WORKER_ID..."
    echo "  Offset: $OFFSET"
    echo "  Limit: $LIMIT"
    echo "  Server: $BASE_URL"
    echo "  Resume: Enabled (auto-resume from previous work)"

    # Start worker in background
    nohup bash -c "
        # Activate conda environment
        source ~/.bashrc
        conda activate replay

        echo 'Worker $WORKER_ID: Starting extraction'
        echo '  Offset: $OFFSET'
        echo '  Limit: $LIMIT'
        echo '  Server: $BASE_URL'
        echo '  Resume: Enabled'
        echo 'Started: \$(date)'

        python -m stindex.exe.extract_corpus \
            --config cfg/extraction/inference/extract.yml \
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
done

echo "✓ All $NUM_GPUS workers started in background"
echo ""
echo "🔄 Auto-resume enabled:"
echo "  - Workers will automatically resume from previous progress"
echo "  - Existing output files will be appended to"
echo "  - No manual merge needed"
echo ""
echo "Monitor progress:"
echo "  tail -f $LOGDIR/worker_*.log"
echo ""
echo "Check progress:"
echo "  for w in \$(seq 1 $NUM_GPUS); do"
echo "    count=\$(wc -l data/extraction_results_parallel/corpus_extraction_worker\${w}_*.jsonl 2>/dev/null | tail -1 | awk '{print \$1}')"
echo "    echo \"Worker \$w: \$count docs\""
echo "  done"
echo ""
