#!/bin/bash
echo "Stopping all extraction workers..."
pkill -f "stindex.exe.extract_corpus"
echo "✓ All extraction workers stopped"
