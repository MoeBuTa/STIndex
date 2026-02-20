#!/bin/bash
echo "Stopping vLLM server..."
pkill -f "swift.llm run_deploy"
echo "✓ vLLM server stopped"
