#!/bin/bash
# STIndex vLLM Server Status Check Script
# Checks health of router and all running vLLM backend servers
#
# Usage:
#   bash scripts/check_vllm_servers.sh

# Parse arguments (for --help)
for arg in "$@"; do
    case $arg in
        -h|--help)
            echo "Usage: $0"
            echo ""
            echo "Checks status and health of all vLLM servers."
            echo ""
            echo "Displays:"
            echo "  - Router status and health"
            echo "  - Backend server status (per model)"
            echo "  - GPU utilization and memory usage"
            echo "  - Recent log entries"
            echo ""
            echo "Exit codes:"
            echo "  0 - All servers running and healthy"
            echo "  1 - No servers running or some unhealthy"
            exit 0
            ;;
        *)
            echo "Unknown option: $arg"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

PID_DIR="logs/vllm/pids"
LOG_DIR="logs/vllm"

echo "============================================================"
echo "STIndex vLLM Server Status (Router + Backends)"
echo "============================================================"
echo ""

# Check for PID files
if [ ! -d "${PID_DIR}" ] || [ ! "$(ls -A ${PID_DIR}/*.pid 2>/dev/null)" ]; then
    echo "✗ No servers found (no PID files in ${PID_DIR})"
    echo ""
    echo "Start servers with: bash scripts/start_vllm_servers.sh"
    exit 1
fi

# Check router first
echo "Router Server:"
echo ""

if [ -f "${PID_DIR}/router.pid" ]; then
    ROUTER_PID=$(cat "${PID_DIR}/router.pid")

    # Get router config
    ROUTER_PORT=$(python3 -c "
import yaml
with open('cfg/vllm.yml', 'r') as f:
    config = yaml.safe_load(f)
print(config.get('router', {}).get('port', 8000))
")

    # Check if process is running
    if ps -p $ROUTER_PID > /dev/null 2>&1; then
        echo "✓ Router"
        echo "    PID: $ROUTER_PID (running)"
        echo "    Port: $ROUTER_PORT"

        # Check health endpoint
        HEALTH_STATUS="⚠ unknown"
        if command -v curl &> /dev/null; then
            HEALTH_RESPONSE=$(curl -s http://localhost:${ROUTER_PORT}/ 2>/dev/null)
            if [ $? -eq 0 ]; then
                HEALTH_STATUS="✓ healthy"

                # Get available models
                MODELS=$(echo "$HEALTH_RESPONSE" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    models = data.get('available_models', [])
    if models:
        print(', '.join(models))
except:
    pass
" 2>/dev/null)

                if [ -n "$MODELS" ]; then
                    echo "    Models: $MODELS"
                fi
            else
                HEALTH_STATUS="✗ not responding"
            fi
        fi
        echo "    Health: $HEALTH_STATUS"

    else
        echo "✗ Router"
        echo "    PID: $ROUTER_PID (not running - stale PID file)"
        echo "    Port: $ROUTER_PORT"
    fi
else
    echo "⚠ Router not found (no PID file)"
fi

echo ""
echo "============================================================"
echo "Backend Servers:"
echo "============================================================"
echo ""

# Read enabled models from vllm.yml for port mapping
ENABLED_MODELS=$(python3 -c "
import yaml
import json

with open('cfg/vllm.yml', 'r') as f:
    config = yaml.safe_load(f)

models = config.get('server', {}).get('models', [])
enabled = []
for model in models:
    if model.get('enabled', False):
        safe_name = model['name'].replace('/', '_').replace(':', '_')
        enabled.append({
            'name': model['name'],
            'safe_name': safe_name,
            'port': model['port'],
        })

print(json.dumps(enabled))
")

# Check each server
RUNNING_COUNT=0
HEALTHY_COUNT=0
TOTAL=0

for pid_file in "${PID_DIR}"/*.pid; do
    # Skip router.pid (already checked)
    if [ "$(basename $pid_file)" = "router.pid" ]; then
        continue
    fi

    if [ -f "$pid_file" ]; then
        TOTAL=$((TOTAL + 1))
        PID=$(cat "$pid_file")
        MODEL_SAFE_NAME=$(basename "$pid_file" .pid)

        # Get full model name and port
        MODEL_INFO=$(echo "$ENABLED_MODELS" | python3 -c "
import sys, json
models = json.load(sys.stdin)
safe_name = '$MODEL_SAFE_NAME'
for m in models:
    if m['safe_name'] == safe_name:
        print(f\"{m['name']}|{m['port']}\")
        break
")

        MODEL_NAME=$(echo "$MODEL_INFO" | cut -d'|' -f1)
        PORT=$(echo "$MODEL_INFO" | cut -d'|' -f2)

        # Check if process is running
        if ps -p $PID > /dev/null 2>&1; then
            RUNNING_COUNT=$((RUNNING_COUNT + 1))
            echo "✓ $MODEL_NAME"
            echo "    PID: $PID (running)"
            echo "    Port: $PORT"

            # Check health endpoint
            HEALTH_STATUS="⚠ unknown"
            if command -v curl &> /dev/null; then
                HEALTH_RESPONSE=$(curl -s -w "\n%{http_code}" http://localhost:${PORT}/health 2>/dev/null)
                HTTP_CODE=$(echo "$HEALTH_RESPONSE" | tail -n1)

                if [ "$HTTP_CODE" = "200" ]; then
                    HEALTHY_COUNT=$((HEALTHY_COUNT + 1))
                    HEALTH_STATUS="✓ healthy"
                elif [ -n "$HTTP_CODE" ]; then
                    HEALTH_STATUS="✗ unhealthy (HTTP $HTTP_CODE)"
                else
                    HEALTH_STATUS="⚠ not responding"
                fi
            fi
            echo "    Health: $HEALTH_STATUS"

            # Show memory usage if available
            if ps -p $PID -o rss= > /dev/null 2>&1; then
                MEM_KB=$(ps -p $PID -o rss= | tr -d ' ')
                MEM_MB=$((MEM_KB / 1024))
                echo "    Memory: ${MEM_MB} MB"
            fi

        else
            echo "✗ $MODEL_NAME"
            echo "    PID: $PID (not running - stale PID file)"
            echo "    Port: $PORT"
        fi

        echo ""
    fi
done

# Summary
echo "============================================================"
echo "Summary"
echo "============================================================"
echo "Router: $([ -f "${PID_DIR}/router.pid" ] && ps -p $(cat ${PID_DIR}/router.pid) > /dev/null 2>&1 && echo "✓ running" || echo "✗ not running")"
echo "Backend servers: $TOTAL"
echo "  Running: $RUNNING_COUNT"
echo "  Healthy: $HEALTHY_COUNT"
echo ""

# GPU Status
if command -v nvidia-smi &> /dev/null; then
    echo "============================================================"
    echo "GPU Status"
    echo "============================================================"

    # Show GPU usage
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | \
        while IFS=, read -r idx name mem_used mem_total util; do
            mem_used=$(echo $mem_used | xargs)
            mem_total=$(echo $mem_total | xargs)
            util=$(echo $util | xargs)
            echo "GPU $idx: $name"
            echo "  Memory: ${mem_used} MB / ${mem_total} MB"
            echo "  Utilization: ${util}%"
        done

    echo ""

    # Show processes using GPUs
    GPU_PROCS=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null)
    if [ ! -z "$GPU_PROCS" ]; then
        echo "GPU Processes:"
        echo "$GPU_PROCS" | while IFS=, read -r pid name mem; do
            pid=$(echo $pid | xargs)
            name=$(echo $name | xargs)
            mem=$(echo $mem | xargs)
            echo "  PID $pid: $name ($mem)"
        done
    else
        echo "No GPU processes found"
    fi

    echo ""
fi

# Recent Errors
echo "============================================================"
echo "Recent Logs (last 5 lines per server)"
echo "============================================================"

if [ -d "${LOG_DIR}" ]; then
    for log_file in "${LOG_DIR}"/*.log; do
        if [ -f "$log_file" ]; then
            echo ""
            echo "$(basename $log_file):"
            tail -n 5 "$log_file" 2>/dev/null | sed 's/^/  /'
        fi
    done
else
    echo "No logs found"
fi

echo ""

# Exit status
if [ $RUNNING_COUNT -eq $TOTAL ] && [ $HEALTHY_COUNT -eq $TOTAL ]; then
    echo "✓ All servers running and healthy"
    exit 0
elif [ $RUNNING_COUNT -eq 0 ]; then
    echo "✗ No servers running"
    exit 1
else
    echo "⚠ Some servers not healthy"
    exit 1
fi
