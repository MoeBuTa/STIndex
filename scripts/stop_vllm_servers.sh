#!/bin/bash
# STIndex vLLM Server Stop Script
# Gracefully stops router and all running vLLM backend servers using Python ServerManager

echo "============================================================"
echo "Stopping STIndex vLLM Servers (Router + Backends)"
echo "============================================================"
echo ""

# Use Python ServerManager to stop servers
python3 << 'EOF'
import sys
from stindex.server.server_manager import ServerManager

try:
    manager = ServerManager(config_path="vllm")

    print("Stopping all servers...")
    print("")

    manager.stop_servers()

    print("")
    print("=" * 60)
    print("✓ All servers stopped")
    print("=" * 60)
    print("")

    # Check GPU status
    import subprocess
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
             "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True
        )
        if result.stdout.strip():
            print("⚠ Warning: GPUs still in use by processes:")
            for line in result.stdout.strip().split("\n"):
                print(f"  {line}")
            print("")
            print("  Run 'nvidia-smi' to check details")
        else:
            print("✓ GPUs released")
    except Exception:
        print("⚠ Could not check GPU status")

    print("")
    sys.exit(0)

except Exception as e:
    print(f"✗ Error: {e}", file=sys.stderr)
    import traceback
    traceback.print_exc()
    sys.exit(1)
EOF
