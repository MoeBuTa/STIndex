# STIndex - Spatiotemporal Information Extraction

[![PyPI version](https://img.shields.io/pypi/v/stindex.svg)](https://pypi.org/project/stindex/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Home](https://img.shields.io/badge/Home-green.svg)](https://stindex.ai4wa.com/)
[![Demo Dashboard](https://img.shields.io/badge/Demo-Dashboard-green.svg)](https://stindex.ai4wa.com/dashboard)

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
)
results = pipeline.run_pipeline(docs)
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



## Configuration

STIndex uses a hierarchical configuration structure organized by module:

### Preprocessing Configs (`cfg/preprocess/`)

- **`chunking.yml`**: Document chunking strategies
  - `strategy`: "sliding_window", "paragraph", "element_based", "semantic"
  - `max_chunk_size`: Maximum tokens per chunk (default: 1500)
  - `overlap`: Token overlap between chunks (default: 150)

- **`parsing.yml`**: Document parsing settings
  - `parsing_method`: "unstructured" (recommended) or "simple"
  - Format-specific settings for PDF, HTML, DOCX
  - `max_file_size_mb`: Maximum file size (default: 50MB)

- **`scraping.yml`**: Web scraping configuration
  - `rate_limit`: Seconds between requests (default: 2.0)
  - `timeout`: Request timeout (default: 30s)
  - `cache.enabled`: Enable response caching
  - `robots.respect_robots_txt`: Respect robots.txt rules

### Extraction Configs (`cfg/extraction/`)

#### Inference Configs (`cfg/extraction/inference/`)

- **`extract.yml`**: Main extraction configuration
  - `llm.llm_provider`: "hf", "openai", or "anthropic"
  - `extraction.enable_cache`: Cache extraction results
  - `extraction.auto_save`: Auto-save to `data/output/yyyy-mm-dd/hh-mm-ss.json`
  - `extraction.min_confidence`: Minimum confidence threshold (0.0-1.0)
  - Context-aware extraction settings
  - Post-processing toggles (reflection, OSM context, relative temporal resolution)

- **`dimensions.yml`**: Multi-dimensional extraction definitions
  - **temporal**: ISO 8601 normalized dates (enabled by default)
  - **spatial**: Geocoded locations with parent regions (enabled by default)
  - **event**: Optional categorical dimension for event types (disabled by default)
  - **entity**: Optional categorical dimension for named entities (disabled by default)
  - Each dimension defines: `enabled`, `extraction_type`, `schema_type`, `fields`, `examples`

- **`reflection.yml`**: Two-pass reflection settings
  - `enabled`: Enable LLM-based quality filtering (default: false)
  - `thresholds`: Relevance, accuracy, completeness, consistency scores
  - Context-aware reasoning for temporal/spatial consistency checks
  - Quality scoring with configurable weights

- **`openai.yml`**: OpenAI API settings
  - `model_name`: "gpt-4o-mini", "gpt-4o", "gpt-4.1", etc.
  - `temperature`: Generation temperature (default: 0.0)
  - `max_tokens`: Maximum output tokens (default: 2048)
  - Requires: `OPENAI_API_KEY` environment variable

- **`anthropic.yml`**: Anthropic Claude API settings
  - `model_name`: "claude-sonnet-4-5-20250929" (latest)
  - `temperature`: Generation temperature (default: 0.0)
  - `max_tokens`: Maximum output tokens (default: 2048)
  - Requires: `ANTHROPIC_API_KEY` environment variable

- **`hf.yml`**: HuggingFace/MS-SWIFT server settings
  - **Client config** (`llm`): API endpoint and generation parameters
    - `model_name`: Model name as reported by server (e.g., "Qwen3-8B")
    - `base_url`: Server endpoint (e.g., "http://localhost:8001")
    - `max_tokens`: Maximum tokens per request (default: 32768)
  - **Server config** (`deployment`): Model deployment settings
    - `model`: HuggingFace model ID (e.g., "Qwen/Qwen3-8B")
    - `port`: Server port (default: 8001)
    - `result_path`: Inference log directory (null to disable)
    - `vllm.tensor_parallel_size`: GPU configuration (`auto` or number)
    - `vllm.gpu_memory_utilization`: GPU memory fraction (default: 0.7)
    - `vllm.max_model_len`: Maximum sequence length (default: 32768)

#### Post-Processing Configs (`cfg/extraction/postprocess/`)

- **`spatial.yml`**: Geocoding and spatial validation
  - `geocoder`: "nominatim" (free, OSM) or "google" (requires API key)
  - `nominatim.rate_limit`: Rate limiting (minimum 1.0 seconds for OSM)
  - `cache.enabled`: Cache geocoding results
  - `disambiguation`: Context-aware disambiguation settings
  - `validation`: Geocoding validation (min_confidence, max_distance_km)

- **`temporal.yml`**: Temporal normalization
  - `format`: "iso8601" (default)
  - `timezone.default`: Default timezone (default: "UTC")
  - `relative.handle_relative`: Resolve relative dates (e.g., "Monday" → absolute date)
  - `ranges.expand_intervals`: Expand date ranges to start/end
  - `validation`: Year range validation (min_year: 1900, max_year: 2100)

#### Evaluation Config (`cfg/extraction/evaluation/`)

- **`evaluate.yml`**: Evaluation settings
  - `dataset.path`: Path to evaluation dataset
  - `dataset.sample_limit`: Limit number of chunks (null = all)
  - `llm.llm_provider`: LLM provider for evaluation
  - `context_aware.enabled`: Enable context-aware extraction
  - Post-processing settings for evaluation

### Switching LLM Providers

Edit `cfg/extraction/inference/extract.yml`:
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
