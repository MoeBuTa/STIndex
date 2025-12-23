#!/bin/bash
# Verify parallel extraction test results
# Usage: bash scripts/extraction/verify_test_results.sh <log_directory>

if [ -z "$1" ]; then
    echo "Usage: bash $0 <log_directory>"
    echo "Example: bash $0 logs/extraction_test_20251217_141520"
    exit 1
fi

LOGDIR="$1"

echo "========================================"
echo "Parallel Extraction Test Verification"
echo "========================================"
echo "Log Directory: $LOGDIR"
echo ""

# Check worker completion
echo "=== Worker Completion Status ==="
for i in 1 2 3 4; do
    logfile="$LOGDIR/worker_${i}.log"

    if [ ! -f "$logfile" ]; then
        echo "Worker $i: ✗ Log file not found"
        continue
    fi

    # Count chunks processed
    chunks=$(grep -c "✓ LLM extracted" "$logfile" 2>/dev/null || echo "0")

    # Check completion
    if grep -q "Success rate: 100" "$logfile" 2>/dev/null; then
        status="✓ COMPLETED"
    else
        status="⏳ Running"
    fi

    # Get document range
    range=$(grep "Loaded.*documents" "$logfile" | head -n 1)

    echo "Worker $i: $status | $chunks chunks | $range"
done

echo ""

# Check output files
echo "=== Output Files ==="
output_dir="data/extraction_results_parallel"
if [ -d "$output_dir" ]; then
    for i in 1 2 3 4; do
        # Find the most recent output file for this worker
        jsonl_file=$(ls -t "$output_dir"/corpus_extraction_worker${i}_*.jsonl 2>/dev/null | grep -v timing | head -n 1)

        if [ -f "$jsonl_file" ]; then
            lines=$(wc -l < "$jsonl_file" 2>/dev/null || echo "0")
            size=$(du -h "$jsonl_file" 2>/dev/null | cut -f1 || echo "0")
            echo "Worker $i: $lines records | $size | $(basename $jsonl_file)"
        else
            echo "Worker $i: ✗ Output file not found"
        fi
    done
else
    echo "✗ Output directory not found: $output_dir"
fi

echo ""

# Check GPU-hours summary
echo "=== GPU-Hours Summary ==="
for i in 1 2 3 4; do
    logfile="$LOGDIR/worker_${i}.log"

    if [ -f "$logfile" ]; then
        # Look for timing summary in the log
        gpu_hours=$(grep "gpu_hours" "$logfile" 2>/dev/null | tail -n 1 || echo "N/A")

        if [ "$gpu_hours" != "N/A" ]; then
            echo "Worker $i: $gpu_hours"
        else
            echo "Worker $i: (still processing)"
        fi
    fi
done

echo ""
echo "========================================"
echo "Verification Complete"
echo "========================================"
