# STIndex - Spatiotemporal Information Extraction

STIndex is a multi-dimensional information extraction system that uses LLMs to extract temporal, spatial, and custom dimensional data from unstructured text. Features end-to-end pipeline with preprocessing, extraction, and visualization.

## Quick Start

### Installation

```bash
pip install -e .
```

### Basic Extraction

```bash
# Extract spatiotemporal entities
stindex extract "On March 15, 2022, a cyclone hit Broome, Western Australia."

# Use specific LLM provider
stindex extract "Text here..." --config openai  # or anthropic, hf
```

### End-to-End Pipeline (NEW in v0.4.0)

```python
from stindex import InputDocument, STIndexPipeline

# Create input documents (URL, file, or text)
docs = [
    InputDocument.from_url("https://example.com/article"),
    InputDocument.from_file("/path/to/document.pdf"),
    InputDocument.from_text("Your text here")
]

# Run full pipeline: preprocessing → extraction → visualization (default)
pipeline = STIndexPipeline(
    dimension_config="dimensions",
    output_dir="data/output"
)
results = pipeline.run_pipeline(docs)
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

## Project Structure

```
stindex/
├── core/                   # Core extraction logic
│   ├── dimensional_extraction.py  # DimensionalExtractor (multi-dimensional)
│   └── utils.py           # JSON extraction utilities
├── preprocessing/          # Preprocessing module (NEW v0.4.0)
│   ├── input_models.py    # InputDocument, DocumentChunk models
│   ├── scraping.py        # WebScraper (rate-limited web scraping)
│   ├── parsing.py         # DocumentParser (HTML/PDF/DOCX/TXT)
│   ├── chunking.py        # DocumentChunker (sliding window, paragraph, semantic)
│   └── processor.py       # Preprocessor (main orchestrator)
├── pipeline/               # Pipeline orchestration (NEW v0.4.0)
│   └── pipeline.py        # STIndexPipeline (end-to-end orchestrator)
├── visualization/          # Visualization module (NEW v0.4.0)
│   ├── visualizer.py      # STIndexVisualizer (main orchestrator)
│   ├── map_generator.py   # MapGenerator (interactive Folium maps)
│   ├── plot_generator.py  # PlotGenerator (statistical plots)
│   ├── statistical_summary.py  # StatisticalSummary
│   └── html_report.py     # HTMLReportGenerator
├── llm/                    # LLM provider implementations
│   ├── manager.py          # LLM factory
│   ├── openai.py          # OpenAI provider
│   ├── anthropic.py       # Anthropic provider
│   ├── hf.py              # HuggingFace client (multi-GPU load balancing)
│   ├── prompts/           # Prompt templates
│   └── response/          # Pydantic models
├── server/                 # Server implementations
│   ├── hf_server.py       # HuggingFace FastAPI server
│   └── mcp_server.py      # MCP server for Claude Desktop
├── spatial/                # Spatial processing & geocoding
├── temporal/               # Temporal processing
├── exe/                    # CLI execution logic
│   └── evaluate.py        # Evaluation system
└── cli.py                  # Typer CLI interface

case_studies/               # Example applications
└── public_health/         # Health surveillance case study
    └── scripts/
        └── run_case_study.py  # Uses generic pipeline

eval/                       # Evaluation scripts
├── evaluate.py            # Main evaluation (sequential/distributed)
└── generate_dataset.py    # Dataset generation

scripts/                    # Helper scripts
├── start_servers.sh       # Multi-GPU server startup
├── check_servers.sh       # Server health monitoring
└── eval_distributed.sh    # Distributed evaluation wrapper
```

## Recent Updates

### v0.4.0 (November 2025): Complete Pipeline & Visualization
- **Generic Preprocessing Module**: Web scraping, document parsing, intelligent chunking
- **End-to-End Pipeline**: Full workflow from URLs/files/text to visualizations
- **Comprehensive Visualization**: Interactive maps, statistical plots, HTML reports
- **4 Execution Modes**: Full pipeline, preprocessing only, extraction only, visualization only
- **Unified Input Model**: Support for URLs, files, and text with single API
- **Case Study Simplification**: Generic modules replace case-specific code

### v0.3.0 (October 2024): Multi-Dimensional Extraction
- **Dimensional Framework**: Extract custom domain-specific dimensions
- **YAML Configuration**: Define dimensions via configuration files
- **Flexible Schema**: Support for temporal, spatial, categorical, and custom dimensions

### v0.2.0 (October 2024): Architecture Refactor
- **Smart JSON Extraction**: Handles thinking models that generate reasoning before/after JSON
- **Raw Output Recording**: Always captures LLM output for debugging failed extractions
- **Evaluation Fixes**: Proper temporal/spatial matching with configurable modes
- **Model Name Display**: Fixed health endpoint to show correct model names
- **Batch Processing**: Preserved true batch mode (no sequential retries)

## Evaluation

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


## Slurm

```bash
sinfo -o "%20N %10P %10T %15G"
salloc -p gpu -n 16 --mem=128G --gres=gpu:v100:1
salloc -p data-inst -n 24 --mem=128G --gres=gpu:h100:1
salloc -p data-inst -n 48 --mem=256G --gres=gpu:h100:2
```


```bash
squeue -u $USER
watch -n 1 "srun --jobid=43462 -n1 bash -lc 'nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits'"

```

## License

MIT License
