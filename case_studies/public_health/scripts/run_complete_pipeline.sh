#!/bin/bash

# Quick Start Script for Public Health Surveillance Case Study
# Runs the complete pipeline from extraction to visualization

set -e  # Exit on error

echo "================================================================================"
echo "Public Health Surveillance - Complete Pipeline"
echo "================================================================================"
echo ""

# Check if vLLM server is running
echo "[1/5] Checking vLLM server..."
if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "❌ vLLM server not running!"
    echo "Please start it with: ./scripts/start_server.sh"
    exit 1
fi
echo "✓ vLLM server is running"
echo ""

# Check for Google Maps API key
if [ -z "$GOOGLE_MAPS_API_KEY" ]; then
    echo "⚠️  GOOGLE_MAPS_API_KEY not set"
    echo "   Geocoding will use Nominatim + city extraction fallback"
    echo "   For better accuracy, set: export GOOGLE_MAPS_API_KEY='your_key'"
else
    echo "✓ Google Maps API key found"
fi
echo ""

# Parse arguments
SAMPLE_LIMIT=""
if [ "$1" == "--test" ]; then
    SAMPLE_LIMIT="--sample-limit 10"
    echo "📝 Running in TEST mode (first 10 documents only)"
    echo ""
fi

# Step 1: Run test extraction
echo "[2/5] Running test extraction..."
python case_studies/public_health/scripts/test_dimensional_extraction.py --test extraction
echo ""

# Step 2: Batch extraction
echo "[3/5] Running batch extraction on all documents..."
python case_studies/public_health/scripts/extract_all_documents.py $SAMPLE_LIMIT
echo ""

# Step 3: Create visualization
echo "[4/5] Creating animated map visualization..."
python case_studies/public_health/visualization/map_generator.py
echo ""

# Step 4: Generate statistical plots
echo "[5/5] Generating statistical plots..."
python case_studies/public_health/visualization/generate_plots.py
echo ""

# Success!
echo "================================================================================"
echo "✓ Pipeline Complete!"
echo "================================================================================"
echo ""
echo "📊 View Results:"
echo "  - Extraction results: case_studies/public_health/data/results/batch_extraction_results.json"
echo "  - Animated map: case_studies/public_health/data/results/health_events_map.html"
echo "  - Statistical plots: case_studies/public_health/data/results/plots/"
echo ""
echo "🌐 Open visualizations in browser:"
echo "  firefox case_studies/public_health/data/results/health_events_map.html"
echo "  firefox case_studies/public_health/data/results/plots/interactive_timeline.html"
echo ""
echo "📈 View report with all visualizations:"
echo "  firefox REPORT.html"
echo ""
echo "================================================================================"
