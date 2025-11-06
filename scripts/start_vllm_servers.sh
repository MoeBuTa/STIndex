#!/bin/bash
# STIndex vLLM Server Startup Script
# Uses Python ServerManager for intelligent GPU allocation

set -e

echo "============================================================"
echo "STIndex vLLM Server Startup (Router + Backends)"
echo "============================================================"
echo ""
echo "Using intelligent GPU memory allocation..."
echo ""

# Use Python ServerManager to start servers
python3 << 'EOF'
import sys
from pathlib import Path
from stindex.server.server_manager import ServerManager

try:
    manager = ServerManager(config_path="vllm")

    print("Starting servers with automatic GPU allocation...")
    print("")

    success = manager.ensure_servers_running(wait_for_ready=True)

    if success:
        print("")
        print("=" * 60)
        print("✓ Startup Complete (Router + Backend Servers)")
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
