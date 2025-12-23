#!/bin/bash
# Entry point for schema discovery
# Config: cfg/discovery/textbook_schema.yml
# Usage: bash scripts/discovery/discover_schema.sh

set -e

CONFIG="cfg/discovery/textbook_schema.yml"

# Setup logging
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOGFILE="logs/discover_schema_${TIMESTAMP}.log"
mkdir -p logs

echo "========================================"
echo "Schema Discovery (Background)"
echo "========================================"
echo "Config: $CONFIG"
echo "Log: $LOGFILE"
echo "Monitor: tail -f $LOGFILE"
echo ""

# Run in background
nohup bash -c "
    echo '========================================'
    echo 'Schema Discovery'
    echo '========================================'
    echo 'Config: $CONFIG'
    echo 'Started: \$(date)'
    echo ''

    python -m stindex.exe.discover_schema --config '$CONFIG'

    echo ''
    echo '✓ Schema discovery complete!'
    echo 'Finished: \$(date)'
" > "$LOGFILE" 2>&1 &

PID=$!
echo "Process ID: $PID"
echo "✓ Started in background"
