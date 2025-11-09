#!/bin/bash
# Stop MS-SWIFT deployment server

set -e

echo "Stopping MS-SWIFT server..."

# Find and kill swift deploy processes
pkill -f "swift deploy" || true

echo "MS-SWIFT server stopped."
