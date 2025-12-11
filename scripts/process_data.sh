#!/bin/bash

# =============================================================================
# STIndex RAG Data Processing Pipeline
# =============================================================================
# This script processes RAG datasets through multiple stages:
#
# Stage 1: Download and format raw datasets (pikerag)
# Stage 2: Extract documents and questions from datasets
# Stage 3: (Optional) Generate training data (GRPO, SFT)
# Stage 4: (Optional) Vector ingestion for retrieval
#
# Output Structure:
#   data/
#   ├── original/               # Stage 1: Formatted QA datasets
#   │   ├── hotpotqa/train.jsonl
#   │   ├── two_wiki/train.jsonl
#   │   └── musique/train.jsonl
#   ├── corpus/                 # Stage 2: Merged corpus
#   │   ├── documents.jsonl     # 1.2M unique documents
#   │   └── {dataset}/train/    # Per-dataset documents
#   ├── questions/              # Stage 2: Merged questions
#   │   ├── questions.jsonl     # 275K questions
#   │   └── {dataset}/train/    # Per-dataset questions
#   ├── data_train/             # Stage 3: Training data
#   │   ├── grpo/
#   │   └── sft/
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
DATASETS="hotpotqa two_wiki musique mirage medcorp"

# Parse command line arguments
SKIP_DOWNLOAD=false
SKIP_CORPUS=false
SKIP_TRAINING=true  # Skip training data by default
SKIP_VECTOR=true    # Skip vector ingestion by default
VECTOR_LIMIT=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-download)
            SKIP_DOWNLOAD=true
            shift
            ;;
        --skip-corpus)
            SKIP_CORPUS=true
            shift
            ;;
        --with-training)
            SKIP_TRAINING=false
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
            echo "  --skip-corpus      Skip Stage 2 (corpus extraction)"
            echo "  --with-training    Enable Stage 3 (training data generation)"
            echo "  --with-vector      Enable Stage 4 (vector ingestion)"
            echo "  --vector-limit N   Limit vector ingestion to N documents"
            echo "  --datasets         Datasets to process (default: hotpotqa two_wiki musique mirage medcorp)"
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

    cd "$PROJECT_ROOT/rag/preprocess/train/pikerag"

    echo "Running pikerag data processor..."
    python main.py data_process/config/datasets.yaml

    cd "$PROJECT_ROOT"

    echo "✓ Stage 1 complete: Raw datasets downloaded and formatted"
else
    echo ""
    echo "Skipping Stage 1 (--skip-download)"
fi

# =============================================================================
# Stage 2: Extract Documents and Questions
# =============================================================================
if [ "$SKIP_CORPUS" = false ]; then
    echo ""
    echo "========================================================"
    echo "  Stage 2: Extract Documents and Questions"
    echo "========================================================"

    echo "Extracting documents and questions from train datasets..."
    python -m rag.preprocess.corpus.extract_documents \
        --datasets $DATASETS \
        --input-dir "$PROJECT_ROOT/data/original" \
        --corpus-dir "$PROJECT_ROOT/data/corpus" \
        --questions-dir "$PROJECT_ROOT/data/questions"

    echo "✓ Stage 2 complete: Corpus and questions extracted"
else
    echo ""
    echo "Skipping Stage 2 (--skip-corpus)"
fi

# =============================================================================
# Stage 3: Generate Training Data (Optional)
# =============================================================================
if [ "$SKIP_TRAINING" = false ]; then
    echo ""
    echo "========================================================"
    echo "  Stage 3: Generate Training Data (GRPO, SFT)"
    echo "========================================================"

    echo "Generating GRPO training data..."
    python "$PROJECT_ROOT/rag/preprocess/train/dataset/dataset_generator.py" \
        --datasets $DATASETS \
        --train-limits 10000 10000 5000 \
        --test-limit 500 \
        --grpo-output-name grpo_25000

    echo ""
    echo "Generating SFT training data..."
    python "$PROJECT_ROOT/rag/preprocess/train/dataset/dataset_generator_sft.py" \
        --mode direct \
        --datasets $DATASETS \
        --output-name sft_2000

    echo "✓ Stage 3 complete: Training data generated"
else
    echo ""
    echo "Skipping Stage 3 (use --with-training to enable)"
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
    python -m rag.preprocess.train.ingestion.vector_ingest \
        --input "$PROJECT_ROOT/data/corpus/documents.jsonl" \
        --output "$PROJECT_ROOT/data/vector/rag" \
        --model "BAAI/bge-m3" \
        --batch-size 32 \
        --index-type ivf \
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
echo "  data/original/{hotpotqa,two_wiki,musique,mirage,medcorp}/train.jsonl"
echo ""
echo "Stage 2 - Corpus and Questions:"
echo "  data/corpus/documents.jsonl      # Merged corpus (1.2M docs)"
echo "  data/questions/questions.jsonl   # Merged questions (275K)"
echo ""

# Print corpus statistics if available
if [ -f "$PROJECT_ROOT/data/corpus/stats.json" ]; then
    echo "Corpus Statistics:"
    cat "$PROJECT_ROOT/data/corpus/stats.json" | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(f'  Total documents: {d.get(\"total_unique_documents\", \"N/A\"):,}')
for name, count in d.get('documents_per_dataset', {}).items():
    print(f'    - {name}: {count:,}')
"
    echo ""
fi

if [ -f "$PROJECT_ROOT/data/questions/stats.json" ]; then
    echo "Questions Statistics:"
    cat "$PROJECT_ROOT/data/questions/stats.json" | python3 -c "
import sys, json
d = json.load(sys.stdin)
print(f'  Total questions: {d.get(\"total_questions\", \"N/A\"):,}')
for name, count in d.get('questions_per_dataset', {}).items():
    print(f'    - {name}: {count:,}')
"
    echo ""
fi

if [ "$SKIP_TRAINING" = false ]; then
    echo "Stage 3 - Training data:"
    echo "  data/data_train/grpo/grpo_25000.jsonl"
    echo "  data/data_train/sft/sft_2000.jsonl"
    echo ""
fi

if [ "$SKIP_VECTOR" = false ] && [ -f "$PROJECT_ROOT/data/vector/rag/index_config.json" ]; then
    echo "Stage 4 - Vector Index:"
    echo "  data/vector/rag/faiss_index.bin"
    echo "  data/vector/rag/id_mapping.json"
    echo "  data/vector/rag/documents_metadata.jsonl"
    echo ""
fi
