# STIndex - Spatiotemporal Information Extraction

[![PyPI version](https://img.shields.io/pypi/v/stindex.svg)](https://pypi.org/project/stindex/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Demo Dashboard](https://img.shields.io/badge/Demo-Dashboard-green.svg)](https://stindex.ai4wa.com/)

STIndex is a multi-dimensional information extraction system that uses LLMs to extract temporal, spatial, and custom dimensional data from unstructured text. Features end-to-end pipeline with preprocessing, extraction, and visualization.

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
# Extract spatiotemporal entities
stindex extract "On March 15, 2022, a cyclone hit Broome, Western Australia."

# Use specific LLM provider
stindex extract "Text here..." --config openai  # or anthropic, hf
```

### End-to-End Pipeline

```python
from stindex import InputDocument, STIndexPipeline

# Create input documents (URL, file, or text)
docs = [
    InputDocument.from_url("https://example.com/article"),
    InputDocument.from_file("/path/to/document.pdf"),
    InputDocument.from_text("Your text here")
]

# Run full pipeline: preprocessing → extraction → warehouse → visualization
pipeline = STIndexPipeline(
    dimension_config="dimensions",
    output_dir="data/output",
    enable_warehouse=True,  # NEW in v0.6.0: Load data into warehouse
    warehouse_config="warehouse"
)
results = pipeline.run_pipeline(docs, load_to_warehouse=True)
# Automatically generates zip archive: data/visualizations/stindex_report_{timestamp}.zip
# Contains: HTML report + all plots, maps, and source files
```

### Python API (Direct Extraction)

```python
from stindex import DimensionalExtractor

# Initialize with default config (cfg/extract.yml)
extractor = DimensionalExtractor()

# Or specify a config
extractor = DimensionalExtractor(config_path="openai")

# Extract entities
result = extractor.extract("March 15, 2022 in Broome, Australia")

# Access results
print(f"Temporal: {len(result.temporal_entities)} entities")
print(f"Spatial: {len(result.spatial_entities)} entities")

# Raw LLM output available for debugging
if result.extraction_config:
    raw_output = result.extraction_config.get("raw_llm_output") if isinstance(result.extraction_config, dict) else result.extraction_config.raw_llm_output
    print(f"Raw output: {raw_output}")
```

## Server Deployment

### MS-SWIFT Server (Model Sharding with Tensor Parallelism)

Deploy a single MS-SWIFT server that uses all available GPUs via tensor parallelism:

```bash
# Deploy server (auto-detects GPUs by default)
./scripts/deploy_ms_swift.sh

# Stop server
./scripts/stop_ms_swift.sh

# Check logs
tail -f logs/hf_server.log
```

**Configuration** (`cfg/hf.yml`):
- `deployment.port`: Server port (default: 8001)
- `deployment.model`: HuggingFace model ID or local path
- `deployment.result_path`: Directory for inference logs (default: `data/output/result`)
- `deployment.vllm.tensor_parallel_size`:
  - `auto` (default): Auto-detect all available GPUs
  - Or set manually: `1`, `2`, `4`, etc.
- `deployment.vllm.gpu_memory_utilization`: GPU memory fraction (default: 0.7)

**Output Logs**:
- Server logs: `logs/hf_server.log`
- Inference logs: `data/output/result/{model_name}/deploy_result/{timestamp}.jsonl`

Each inference log contains:
- `response`: Complete LLM output (including `<think>` tags and JSON)
- `infer_request`: Input messages and generation config
- `generation_config`: Sampling parameters used

## Configuration

Configuration files in `cfg/`:
- `extract.yml`: Main configuration (sets LLM provider)
- `evaluate.yml`: Evaluation settings
- `dimensions.yml`: Multi-dimensional extraction configuration
- `warehouse.yml`: Data warehouse configuration (connection, ETL, embeddings)
- `openai.yml`: OpenAI API settings (GPT-4)
- `anthropic.yml`: Anthropic API settings (Claude)
- `hf.yml`: HuggingFace/MS-SWIFT server settings
  - **Client config** (`llm`): API endpoint and generation parameters
  - **Server config** (`deployment`): Model deployment settings
    - `result_path`: Inference log directory (default: `data/output/result`)
    - `vllm.tensor_parallel_size`: GPU configuration (`auto` or number)

### Switching Providers

Edit `cfg/extract.yml`:
```yaml
llm:
  llm_provider: hf  # or openai, anthropic
```

Or specify at runtime:
```python
extractor = DimensionalExtractor(config_path="openai")
```

### Quick Evaluation

```bash
# Sequential mode (default)
stindex evaluate

# With specific config
stindex evaluate --llm-config openai

# Limit samples
stindex evaluate --sample-limit 10
```

### Output Structure

Results are organized by dataset and model:
```
data/output/evaluations/
└── {dataset_name}-{model_name}/
    ├── eval_{timestamp}_{config}.csv         # Detailed results
    └── eval_{timestamp}_{config}.summary.json # Aggregate metrics
```

### TODOs

 - Backend server implementation
 - Data warehouse integration


## License

MIT License




