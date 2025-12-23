#!/bin/bash
# Simple bash-based progress monitor for parallel extraction

LOGDIR="${1:-logs/extraction_parallel_20251217_184229}"

echo "Monitoring: $LOGDIR"
echo "Press Ctrl+C to exit"
echo ""

while true; do
    clear
    echo "================================================================================"
    echo "PARALLEL EXTRACTION PROGRESS - $(date '+%Y-%m-%d %H:%M:%S')"
    echo "================================================================================"
    echo ""

    # Check each worker
    for w in 1 2 3 4; do
        LOGFILE="$LOGDIR/worker_$w.log"
        if [ -f "$LOGFILE" ]; then
            echo "Worker $w:"
            # Get latest progress line
            tail -100 "$LOGFILE" | grep "Extracting dimensions:" | tail -1 || echo "  Initializing..."
            echo ""
        else
            echo "Worker $w: Log not found"
            echo ""
        fi
    done

    echo "================================================================================"
    echo "Refresh: 10s | Log: $LOGDIR"
    echo "================================================================================"

    sleep 10
done
