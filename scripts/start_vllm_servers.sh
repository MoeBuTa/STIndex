#!/bin/bash
# STIndex vLLM Server Startup Script
# Uses Python ServerManager for intelligent GPU allocation
#
# Usage:
#   bash scripts/start_vllm_servers.sh           # Wait for servers to be ready (default)
#   bash scripts/start_vllm_servers.sh --no-wait # Start servers and return immediately
#   bash scripts/start_vllm_servers.sh &         # Run in background (equivalent to --no-wait)

set -e

# Parse arguments
WAIT_FOR_READY=true
for arg in "$@"; do
    case $arg in
        --no-wait)
            WAIT_FOR_READY=false
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--no-wait]"
            echo ""
            echo "Options:"
            echo "  --no-wait    Start servers without waiting for them to be ready"
            echo "  -h, --help   Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                 # Wait for servers to be ready (default)"
            echo "  $0 --no-wait       # Start and return immediately"
            echo "  $0 &               # Run in background"
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
echo "STIndex vLLM Server Startup (Router + Backends)"
echo "============================================================"
echo ""
echo "Using intelligent GPU memory allocation..."
echo "Wait for ready: $WAIT_FOR_READY"
echo ""

# Export variable for Python to access
export WAIT_FOR_READY

# Use Python ServerManager to start servers
python3 << EOF
import sys
import os
from pathlib import Path
from stindex.server.server_manager import ServerManager

try:
    manager = ServerManager(config_path="vllm")

    # Get wait_for_ready from bash variable
    wait_for_ready = os.environ.get('WAIT_FOR_READY', 'true').lower() == 'true'

    print("Starting servers with automatic GPU allocation...")
    print("")

    success = manager.ensure_servers_running(wait_for_ready=wait_for_ready)

    if success:
        print("")
        print("=" * 60)
        if wait_for_ready:
            print("✓ Startup Complete (Router + Backend Servers)")
        else:
            print("✓ Servers Started (initializing in background)")
        print("=" * 60)
        print("")
        print("Architecture:")
        print(f"  Client → Router (port {manager.router_port}) → Backend vLLM Servers")
        print("")
        print("To monitor logs:")
        print(f"  tail -f {manager.log_dir}/router.log")
        for model in manager.enabled_models:
            safe_name = model['name'].replace('/', '_').replace(':', '_')
            print(f"  tail -f {manager.log_dir}/{safe_name}.log")
        print("")
        print("To check server status:")
        print("  bash scripts/check_vllm_servers.sh")
        print("")
        print("To stop all servers:")
        print("  bash scripts/stop_vllm_servers.sh")
        print("")
        if not wait_for_ready:
            print("Note: Servers are initializing in the background.")
            print("      Full initialization may take 1-2 minutes depending on model size.")
            print("      Use 'bash scripts/check_vllm_servers.sh' to monitor status.")
        else:
            print("Note: Full initialization may take 1-2 minutes depending on model size.")
            print("      Monitor logs to see when models are fully loaded.")
        print("")
        sys.exit(0)
    else:
        print("")
        print("✗ Server startup failed. Check logs for details:")
        print(f"  {manager.log_dir}/")
        print("")
        sys.exit(1)

except Exception as e:
    print(f"✗ Error: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF
