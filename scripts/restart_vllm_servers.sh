#!/bin/bash
# STIndex vLLM Server Restart Script
# Stops and restarts all vLLM servers
#
# Usage:
#   bash scripts/restart_vllm_servers.sh           # Wait for servers to be ready (default)
#   bash scripts/restart_vllm_servers.sh --no-wait # Restart without waiting for ready
#   bash scripts/restart_vllm_servers.sh &         # Run in background

# Parse arguments
WAIT_FLAG=""
for arg in "$@"; do
    case $arg in
        --no-wait)
            WAIT_FLAG="--no-wait"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--no-wait]"
            echo ""
            echo "Options:"
            echo "  --no-wait    Restart servers without waiting for them to be ready"
            echo "  -h, --help   Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                 # Wait for servers to be ready (default)"
            echo "  $0 --no-wait       # Restart and return immediately"
            exit 0
            ;;
        *)
            echo "Unknown option: $arg"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

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
bash scripts/start_vllm_servers.sh $WAIT_FLAG

echo ""
echo "✓ Restart complete"
echo ""
