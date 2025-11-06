#!/bin/bash
# STIndex vLLM Server Restart Script
# Stops and restarts all vLLM servers

echo "============================================================"
echo "Restarting STIndex vLLM Servers"
echo "============================================================"
echo ""

# Stop servers
echo "Step 1: Stopping existing servers..."
echo ""
bash scripts/stop_vllm_servers.sh

echo ""
echo "============================================================"
echo ""

# Start servers
echo "Step 2: Starting servers..."
echo ""
bash scripts/start_vllm_servers.sh

echo ""
echo "✓ Restart complete"
echo ""
