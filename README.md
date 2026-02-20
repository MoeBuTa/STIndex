# STIndex - Spatiotemporal Information Extraction

[![PyPI version](https://img.shields.io/pypi/v/stindex.svg)](https://pypi.org/project/stindex/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Home Homepage](https://img.shields.io/badge/Home-Homepage-green.svg)](https://stindex.ai4wa.com/)
[![Demo Dashboard](https://img.shields.io/badge/Demo-Dashboard-green.svg)](https://stindex.ai4wa.com/dashboard)

STIndex is a multi-dimensional information extraction system that uses LLMs to extract temporal, spatial, and custom dimensional data from unstructured text. Features an end-to-end pipeline with preprocessing, extraction, and evaluation.

**🌐 [Try the Demo Dashboard](https://stindex.ai4wa.com/)**

## Quick Start

### Installation

```bash
pip install stindex

# Install spaCy language model (required for NER)
python -m spacy download en_core_web_sm
```

### Basic Extraction

```bash
# Extract spatiotemporal entities (uses default config + PROVIDER_DEFAULTS)
stindex extract "On March 15, 2022, a cyclone hit Broome, Western Australia."

# Use a specific provider
stindex extract "Text here..." --config openai   # or anthropic, hf

# Override model and parameters at runtime
stindex extract "Text here..." --config hf --model Qwen/Qwen3-8B \
    --base-url http://localhost:8001 --temperature 0.0 --max-tokens 4096
```

### Python API

```python
from stindex import DimensionalExtractor

extractor = DimensionalExtractor(config_path="openai")
result = extractor.extract("March 15, 2022 in Broome, Western Australia")

for e in result.temporal_entities:
    print(e["text"], "→", e["normalized"])   # "March 15, 2022" → "2022-03-15"

for e in result.spatial_entities:
    print(e["text"], e["latitude"], e["longitude"])  # "Broome" -17.96 122.22
```

### End-to-End Pipeline

```python
from stindex import InputDocument, STIndexPipeline

docs = [
    InputDocument.from_url("https://example.com/article"),
    InputDocument.from_file("/path/to/document.pdf"),
    InputDocument.from_text("Your text here")
]

pipeline = STIndexPipeline(dimension_config="dimensions", output_dir="data/output")
results = pipeline.run_pipeline(docs)
```

### Schema Discovery

Automatically discover dimensional schemas from Q&A datasets:

```python
from stindex.pipeline.discovery_pipeline import SchemaDiscoveryPipeline

discovery = SchemaDiscoveryPipeline(
    questions_path="data/original/mirage/train.jsonl",
    corpus_path="data/original/medcorp/train.jsonl",
    output_path="cfg/discovered_medical_schema.yml",
    n_clusters=10
)
schema = discovery.run()

# Use discovered schema for extraction
pipeline = STIndexPipeline(dimension_config="cfg/discovered_medical_schema.yml")
results = pipeline.run_pipeline(docs)
```

**Supported datasets:** MIRAGE, MedCorp, HotpotQA, 2WikiMQA, MuSiQue

---

## LLM Providers

Provider defaults are defined in `stindex/utils/config.py` and can be overridden at runtime via CLI flags or Python API. No separate YAML files needed — just select a provider.

| Provider | Default Model | Config File |
|----------|--------------|-------------|
| `openai` | `gpt-4o-mini` | `cfg/extraction/inference/openai.yml` |
| `anthropic` | `claude-sonnet-4-5-20250929` | `cfg/extraction/inference/anthropic.yml` |
| `hf` | `Qwen3-4B-Instruct-2507` | `cfg/extraction/inference/hf.yml` |

Select provider in `cfg/extraction/inference/extract.yml`:
```yaml
llm:
  llm_provider: hf   # or openai, anthropic
```

Or override everything at runtime:
```bash
stindex extract "Text..." --config openai --model gpt-4o --temperature 0.0

stindex extract "Text..." --config hf \
    --model Qwen3-4B-Instruct-2507 --base-url http://localhost:8001
```

---

## HuggingFace Server (MS-SWIFT + vLLM)

Deploy a model using MS-SWIFT with vLLM backend. Configure in `cfg/extraction/inference/hf.yml`:

```yaml
deployment:
  model: Qwen/Qwen3-4B-Instruct-2507   # HuggingFace model ID or local path
  port: 8001
  vllm:
    tensor_parallel_size: 1            # Number of GPUs (or "auto")
    gpu_memory_utilization: 0.7
    max_model_len: 16384
```

```bash
# Start server (reads cfg/extraction/inference/hf.yml)
./scripts/server/deploy_ms_swift.sh

# Stop server
./scripts/server/stop_ms_swift.sh

# Monitor
tail -f logs/hf_server.log
```

---

## Configuration

```
cfg/
├── extraction/
│   ├── inference/
│   │   ├── extract.yml        # Main config: llm_provider, feature toggles
│   │   ├── dimensions.yml     # Dimension schema definitions
│   │   ├── reflection.yml     # Two-pass reflection thresholds
│   │   ├── hf.yml             # HF server deployment config
│   │   ├── openai.yml         # Provider selector (llm_provider: openai)
│   │   └── anthropic.yml      # Provider selector (llm_provider: anthropic)
│   └── postprocess/
│       ├── spatial.yml        # Geocoding settings (Nominatim, Google Maps)
│       └── temporal.yml       # Temporal normalization (ISO 8601)
├── preprocess/
│   ├── chunking.yml           # Chunking strategy and parameters
│   ├── parsing.yml            # Document parsing (PDF, HTML, DOCX)
│   └── scraping.yml           # Web scraping (rate limits, caching)
└── discovery/
    └── textbook_schema.yml    # Example discovered schema
```

### Key Config: `extract.yml`

```yaml
llm:
  llm_provider: openai     # Selects provider; model/temp/tokens use PROVIDER_DEFAULTS

spatial:
  enable_osm_context: false  # Fetch nearby POIs for disambiguation (slow)

temporal:
  enable_relative_resolution: true  # Resolve "yesterday" → absolute date

reflection:
  enabled: false             # Two-pass LLM quality filtering (adds latency)

categorical:
  enable_validation: true    # Validate categories against allowed values
```

### Key Config: `dimensions.yml`

Defines extraction schemas. Temporal and spatial are always enabled. Additional dimensions (e.g., `event`, `entity`) can be toggled:

```yaml
dimensions:
  temporal:
    enabled: true
    extraction_type: normalized
    # hierarchy: timestamp → date → month → year
  spatial:
    enabled: true
    extraction_type: geocoded
    # hierarchy: location → city → state → country
  event:
    enabled: false   # Set to true to extract event categories
    extraction_type: categorical
```

---

## Evaluation

Compare baseline vs. context-aware extraction on annotated datasets:

```bash
# Run on built-in evaluation set
stindex evaluate \
    --dataset data/evaluation/context_aware_eval.json \
    --config hf --model Qwen3-4B-Instruct-2507 --base-url http://localhost:8001

# Limit to 20 samples for quick testing
stindex evaluate --dataset data/evaluation/context_aware_eval.json \
    --config openai --sample-limit 20

# Resume from checkpoint
stindex evaluate --dataset data/evaluation/context_aware_eval_extended.json \
    --config hf --output-dir data/output/evaluations/my_run
```

Results saved to `--output-dir` (default: `data/output/evaluations/`):
```
data/output/evaluations/
└── {run_dir}/
    ├── baseline_{timestamp}.csv          # Per-chunk baseline results
    ├── context_aware_{timestamp}.csv     # Per-chunk context-aware results
    └── comparison_summary_{timestamp}.json
```

Metrics reported (following CoNLL-2003 and TempEval-3 standards):
- **Temporal**: Precision, Recall, F1, Normalization Accuracy
- **Spatial**: Precision, Recall, F1, Geocoding Success Rate, Mean Distance Error, Accuracy@25km

---

## Scripts

```
scripts/
├── server/
│   ├── deploy_ms_swift.sh         # Start HF model server
│   └── stop_ms_swift.sh           # Stop HF model server
├── extract/
│   ├── extract_openai.sh          # Single-text extraction via OpenAI
│   ├── extract_anthropic.sh       # Single-text extraction via Anthropic
│   └── extract_hf.sh              # Single-text extraction via HF server
├── evaluate/
│   ├── evaluate_openai.sh         # Evaluation via OpenAI
│   └── evaluate_hf.sh             # Evaluation via HF server
├── extraction/
│   ├── extract_corpus.sh          # Corpus extraction (background)
│   ├── extract_corpus_parallel.sh # Parallel corpus extraction (multi-GPU)
│   ├── monitor_progress.sh        # Monitor parallel extraction progress
│   └── stop_extraction_parallel.sh
├── discovery/
│   └── discover_schema.sh         # Run schema discovery pipeline
└── rag/
    ├── filter_questions.sh        # Filter evaluation questions
    └── preprocess_corpus.sh       # Preprocess corpus for RAG
```

---

## Slurm (HPC)

```bash
# Single GPU
salloc -p data-inst -n 24 --mem=128G --gres=gpu:h100:1

# Multi-GPU (for tensor parallelism)
salloc -p data-inst -n 48 --mem=256G --gres=gpu:h100:2
```

---

## License

MIT License
