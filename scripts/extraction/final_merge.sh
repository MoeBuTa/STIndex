#!/bin/bash
# Final merge: Combine previous 76,900 docs + new worker outputs
# Run this after the current 2-GPU extraction completes

set -e

echo "=========================================="
echo "Final Merge: All Extraction Results"
echo "=========================================="
echo ""

# Find the latest worker output files from current run
WORKER1_NEW=$(ls -t data/extraction_results_parallel/corpus_extraction_worker1_2025*.jsonl 2>/dev/null | grep -v "timing\|partial" | head -1)
WORKER2_NEW=$(ls -t data/extraction_results_parallel/corpus_extraction_worker2_2025*.jsonl 2>/dev/null | grep -v "timing\|partial" | head -1)

# Previous merged file
PREVIOUS="data/extraction_results_parallel/merged/combined_progress.jsonl"

# Output
FINAL_OUTPUT="data/extraction_results_parallel/final_complete.jsonl"

echo "Sources:"
echo "  1. Previous merged: $PREVIOUS"
if [ -f "$PREVIOUS" ]; then
    PREV_COUNT=$(wc -l < "$PREVIOUS")
    echo "     Lines: $PREV_COUNT"
else
    echo "     ✗ File not found!"
    exit 1
fi

echo "  2. New Worker 1: $WORKER1_NEW"
if [ -f "$WORKER1_NEW" ]; then
    W1_COUNT=$(wc -l < "$WORKER1_NEW")
    echo "     Lines: $W1_COUNT"
else
    echo "     ✗ File not found (extraction may not be complete)"
    W1_COUNT=0
fi

echo "  3. New Worker 2: $WORKER2_NEW"
if [ -f "$WORKER2_NEW" ]; then
    W2_COUNT=$(wc -l < "$WORKER2_NEW")
    echo "     Lines: $W2_COUNT"
else
    echo "     ✗ File not found (extraction may not be complete)"
    W2_COUNT=0
fi

echo ""
echo "Merging..."

# Combine all files
cat "$PREVIOUS" > "$FINAL_OUTPUT"
if [ -f "$WORKER1_NEW" ]; then
    cat "$WORKER1_NEW" >> "$FINAL_OUTPUT"
fi
if [ -f "$WORKER2_NEW" ]; then
    cat "$WORKER2_NEW" >> "$FINAL_OUTPUT"
fi

# Count final
FINAL_COUNT=$(wc -l < "$FINAL_OUTPUT")
EXPECTED=$((PREV_COUNT + W1_COUNT + W2_COUNT))

echo ""
echo "=========================================="
echo "✓ Merge Complete!"
echo "=========================================="
echo "  Output: $FINAL_OUTPUT"
echo "  Total lines: $FINAL_COUNT"
echo "  Expected: $EXPECTED"
echo ""

if [ $FINAL_COUNT -eq $EXPECTED ]; then
    echo "✓ Line count matches!"
else
    echo "⚠️  Line count mismatch (expected $EXPECTED, got $FINAL_COUNT)"
fi

# Show file size
SIZE=$(du -h "$FINAL_OUTPUT" | cut -f1)
echo "  File size: $SIZE"
echo ""

# Verify no duplicates by checking unique doc_ids
echo "Checking for duplicates..."
UNIQUE_IDS=$(jq -r '.doc_id' "$FINAL_OUTPUT" | sort -u | wc -l)
echo "  Total documents: $FINAL_COUNT"
echo "  Unique doc_ids: $UNIQUE_IDS"

if [ $UNIQUE_IDS -eq $FINAL_COUNT ]; then
    echo "  ✓ No duplicates found!"
else
    DUPLICATES=$((FINAL_COUNT - UNIQUE_IDS))
    echo "  ⚠️  Found $DUPLICATES duplicate documents"
fi

echo ""
echo "Final corpus extraction complete!"
echo "All 125,847 documents processed."
