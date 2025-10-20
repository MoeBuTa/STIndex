#!/bin/bash
# Check HuggingFace server status and GPU memory usage

SERVER_URL="${STINDEX_HF_SERVER_URL:-http://localhost:8001}"

echo "=== HuggingFace LLM Server Status ==="
echo ""

# Check if process is running
PIDS=$(pgrep -f "stindex.server.hf.app")
if [ -z "$PIDS" ]; then
    echo "Status: NOT RUNNING"
    echo ""
else
    echo "Status: RUNNING"
    echo "Process ID(s): $PIDS"
    echo ""

    # Try to get health info from API
    echo "API Health Check:"
    if command -v curl &> /dev/null; then
        curl -s "$SERVER_URL/health" 2>/dev/null | python -m json.tool 2>/dev/null || echo "  (API not reachable yet - model may still be loading)"
    else
        echo "  (curl not available)"
    fi
    echo ""
fi

# Show GPU memory usage
echo "=== GPU Memory Usage ==="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | \
    awk -F', ' '{printf "GPU %s (%s): %s/%s MB (%.1f%% util)\n", $1, $2, $3, $4, $5}'
else
    echo "nvidia-smi not available"
fi
