#!/bin/bash
# Start single HuggingFace server instance on default GPU

# Default values
PORT=${STINDEX_HF_PORT:-8001}
CONFIG=${STINDEX_CONFIG:-hf}

echo "Starting HuggingFace server..."
echo "  Port: $PORT"
echo "  Config: $CONFIG"

export STINDEX_CONFIG=$CONFIG

# Create logs directory if not exists
mkdir -p logs

# Start server in background
nohup python -m uvicorn stindex.server.app:app \
    --host 0.0.0.0 \
    --port $PORT \
    --log-level info \
    > logs/server_port${PORT}.log 2>&1 &

PID=$!

echo "✓ HF server started (PID: $PID)"
echo "  URL: http://localhost:$PORT"
echo "  Log: logs/server_port${PORT}.log"
echo ""
echo "Commands:"
echo "  View logs:   tail -f logs/server_port${PORT}.log"
echo "  Stop server: ./scripts/stop_servers.sh"
