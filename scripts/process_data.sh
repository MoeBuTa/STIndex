#!/bin/bash

# =============================================================================
# STIndex RAG Data Processing Pipeline
# =============================================================================
# This script processes RAG datasets through multiple stages:
#
# Stage 1: Download and format raw datasets (pikerag)
# Stage 2: Generate training data (GRPO, SFT)
# Stage 3: Extract unique documents from GRPO for RAG corpus
# Stage 4: (Optional) Vector ingestion for retrieval
#
# Output Structure:
#   data/
#   ├── original/               # Stage 1: Formatted QA datasets
#   │   ├── hotpotqa/
#   │   ├── two_wiki/
#   │   └── musique/
#   ├── data_conversation/      # Stage 2: Test data for inference
#   ├── data_train/             # Stage 2: Training data
#   │   ├── grpo/
#   │   └── sft/
#   ├── corpus/                 # Stage 3: RAG corpus (from GRPO)
#   │   └── grpo/
#   │       └── chunks.jsonl    # Unique documents for retrieval
#   └── vector/                 # Stage 4: Vector index
#       └── rag/
# =============================================================================

set -e  # Exit on error

echo "========================================================"
echo "  STIndex RAG Data Processing Pipeline"
echo "========================================================"

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Configuration
DATASETS="hotpotqa two_wiki musique"

# Parse command line arguments
SKIP_DOWNLOAD=false
SKIP_TRAINING=false
SKIP_CORPUS=false
SKIP_VECTOR=true  # Skip vector ingestion by default (takes time)
VECTOR_LIMIT=""   # Empty means all documents

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-download)
            SKIP_DOWNLOAD=true
            shift
            ;;
        --skip-training)
            SKIP_TRAINING=true
            shift
            ;;
        --skip-corpus)
            SKIP_CORPUS=true
            shift
            ;;
        --with-vector)
            SKIP_VECTOR=false
            shift
            ;;
        --vector-limit)
            VECTOR_LIMIT="--limit $2"
            shift 2
            ;;
        --datasets)
            DATASETS="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-download    Skip Stage 1 (download and format)"
            echo "  --skip-training    Skip Stage 2 (training data generation)"
            echo "  --skip-corpus      Skip Stage 3 (corpus extraction from GRPO)"
            echo "  --with-vector      Enable Stage 4 (vector ingestion)"
            echo "  --vector-limit N   Limit vector ingestion to N documents"
            echo "  --datasets         Datasets to process (default: hotpotqa two_wiki musique)"
            echo "  --help             Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# =============================================================================
# Stage 1: Download and Format Raw Datasets
# =============================================================================
if [ "$SKIP_DOWNLOAD" = false ]; then
    echo ""
    echo "========================================================"
    echo "  Stage 1: Download and Format Raw Datasets"
    echo "========================================================"

    cd "$PROJECT_ROOT/rag/preprocess/pikerag"

    echo "Running pikerag data processor..."
    python main.py data_process/config/datasets.yaml

    cd "$PROJECT_ROOT"

    echo "✓ Stage 1 complete: Raw datasets downloaded and formatted"
else
    echo ""
    echo "Skipping Stage 1 (--skip-download)"
fi

# =============================================================================
# Stage 2: Generate Training and Evaluation Data
# =============================================================================
if [ "$SKIP_TRAINING" = false ]; then
    echo ""
    echo "========================================================"
    echo "  Stage 2: Generate Training Data (GRPO, SFT)"
    echo "========================================================"

    echo "Generating conversation data for inference/evaluation..."
    python "$PROJECT_ROOT/rag/preprocess/dataset/dataset_generator.py" \
        --datasets $DATASETS \
        --train-limits 10000 10000 5000 \
        --test-limit 500 \
        --grpo-output-name grpo_25000

    echo ""
    echo "Generating SFT training data (2000 samples)..."
    python "$PROJECT_ROOT/rag/preprocess/dataset/dataset_generator_sft.py" \
        --mode direct \
        --datasets $DATASETS \
        --output-name sft_2000

    echo "✓ Stage 2 complete: Training data generated"
else
    echo ""
    echo "Skipping Stage 2 (--skip-training)"
fi

# =============================================================================
# Stage 3: Extract Documents from GRPO for RAG Corpus
# =============================================================================
if [ "$SKIP_CORPUS" = false ]; then
    echo ""
    echo "========================================================"
    echo "  Stage 3: Extract RAG Corpus from GRPO"
    echo "========================================================"

    echo "Extracting unique documents from GRPO training data..."
    python "$PROJECT_ROOT/rag/preprocess/extract_grpo_docs.py" \
        --input "$PROJECT_ROOT/data/data_train/grpo/grpo_25000.jsonl" \
        --output "$PROJECT_ROOT/data/corpus/grpo/chunks.jsonl"

    echo "✓ Stage 3 complete: RAG corpus extracted"
else
    echo ""
    echo "Skipping Stage 3 (--skip-corpus)"
fi

# =============================================================================
# Stage 4: Vector Ingestion (Optional)
# =============================================================================
if [ "$SKIP_VECTOR" = false ]; then
    echo ""
    echo "========================================================"
    echo "  Stage 4: Vector Ingestion (FAISS)"
    echo "========================================================"

    echo "Running vector ingestion..."
    python -m rag.preprocess.ingestion.vector_ingest \
        --input "$PROJECT_ROOT/data/corpus/grpo/chunks.jsonl" \
        --output "$PROJECT_ROOT/data/vector/rag" \
        --model "BAAI/bge-m3" \
        --batch-size 32 \
        --index-type flat \
        $VECTOR_LIMIT

    echo "✓ Stage 4 complete: Vector index built"
else
    echo ""
    echo "Skipping Stage 4 (use --with-vector to enable)"
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "========================================================"
echo "  Data Processing Complete!"
echo "========================================================"
echo ""
echo "Generated files:"
echo ""
echo "Stage 1 - Formatted QA datasets:"
echo "  data/original/{hotpotqa,two_wiki,musique}/{dev,train}.jsonl"
echo ""
echo "Stage 2 - Training data:"
echo "  data/data_train/grpo/grpo_25000.jsonl    # GRPO training"
echo "  data/data_train/sft/sft_2000.jsonl       # SFT training"
echo "  data/data_conversation/{dataset}/test.jsonl  # Evaluation"
echo ""
echo "Stage 3 - RAG corpus (from GRPO):"
echo "  data/corpus/grpo/chunks.jsonl            # Unique documents"
echo "  data/corpus/grpo/extraction_stats.json   # Statistics"
echo ""

# Print corpus statistics if available
if [ -f "$PROJECT_ROOT/data/corpus/grpo/extraction_stats.json" ]; then
    echo "RAG Corpus Statistics:"
    cat "$PROJECT_ROOT/data/corpus/grpo/extraction_stats.json" | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(f'  Total QA samples: {d.get(\"total_samples\", \"N/A\"):,}')
print(f'  Document references: {d.get(\"total_doc_refs\", \"N/A\"):,}')
print(f'  Unique documents: {d.get(\"unique_documents\", \"N/A\"):,}')
"
    echo ""
fi

if [ "$SKIP_VECTOR" = false ] && [ -f "$PROJECT_ROOT/data/vector/rag/index_config.json" ]; then
    echo "Vector Index:"
    echo "  data/vector/rag/faiss_index.bin"
    echo "  data/vector/rag/id_mapping.json"
    echo "  data/vector/rag/chunks_metadata.jsonl"
    echo ""
fi
