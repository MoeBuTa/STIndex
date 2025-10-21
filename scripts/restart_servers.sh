#!/bin/bash
# Restart HuggingFace servers (stop then start)

set -e  # Exit on error

# Colors for output
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Restarting HuggingFace Servers ===${NC}"
echo ""

# Stop existing servers
echo "Step 1/2: Stopping servers..."
./scripts/stop_servers.sh

# Wait a moment to ensure clean shutdown
echo ""
echo "Waiting for clean shutdown..."
sleep 2

# Start servers with same configuration
echo ""
echo "Step 2/2: Starting servers..."
./scripts/start_servers.sh

echo ""
echo -e "${BLUE}Restart complete!${NC}"
