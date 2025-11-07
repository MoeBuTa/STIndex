#!/bin/bash

# Complete pipeline for public health surveillance case study
# Uses STIndex preprocessing, extraction, and visualization modules

set -e  # Exit on error

echo "=============================================="
echo "Public Health Surveillance Case Study"
echo "STIndex End-to-End Pipeline"
echo "=============================================="
echo ""

# Check if we're in the project root
if [ ! -f "stindex/__init__.py" ]; then
    echo "Error: Please run from project root directory"
    exit 1
fi

# Check if vLLM server is running (optional, will auto-start if needed)
echo "[1/2] Checking vLLM server..."
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✓ vLLM server is running"
else
    echo "⚠️  vLLM server not running, will auto-start if needed"
fi
echo ""

# Run the complete pipeline
echo "[2/2] Running pipeline (preprocessing → extraction → visualization)..."
python case_studies/public_health/scripts/run_case_study.py

echo ""
echo "=============================================="
echo "Pipeline Complete!"
echo "=============================================="
echo ""
echo "Results saved to:"
echo "  - Chunks: case_studies/public_health/data/chunks/"
echo "  - Extraction: case_studies/public_health/data/results/"
echo "  - Visualizations: case_studies/public_health/data/visualizations/"
echo ""
