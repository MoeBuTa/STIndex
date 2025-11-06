# STIndex - Spatiotemporal Information Extraction

STIndex is a spatiotemporal information extraction system that uses LLMs to extract and normalize temporal expressions (dates, times, durations) and spatial entities (locations) from unstructured text, with geocoding support.

## Quick Start

### Installation

```bash
pip install -e .
```

### Basic Usage

```bash
# Extract spatiotemporal entities
stindex extract "On March 15, 2022, a cyclone hit Broome, Western Australia."

# Run evaluation on a dataset
stindex evaluate

# Use specific LLM provider
stindex extract "Text here..." --config openai  # or anthropic, hf
```

### Python API

```python
from stindex import STIndexExtractor

# Initialize with default config (cfg/extract.yml)
extractor = STIndexExtractor()

# Or specify a config
extractor = STIndexExtractor(config_path="openai")

# Extract entities
result = extractor.extract("March 15, 2022 in Broome, Australia")

# Access results
print(f"Temporal: {len(result.temporal_entities)} entities")
print(f"Spatial: {len(result.spatial_entities)} entities")

# Raw LLM output available for debugging
if result.extraction_config:
    print(f"Raw output: {result.extraction_config.raw_llm_output}")
```

## Features

### Core Capabilities
- **Single LLM Call**: Unified extraction of both temporal and spatial information
- **Multiple Providers**: OpenAI (GPT-4), Anthropic (Claude), and HuggingFace (Qwen, Llama, etc.)
- **Multi-GPU Support**: Automatic load balancing across multiple GPU servers for high throughput
- **Batch Processing**: Efficient batch API for evaluations and bulk processing
- **Structured Outputs**: Pydantic models with automatic validation
- **ISO 8601 Normalization**: Standardized temporal representations

### Advanced Features
- **Thinking Model Support**: Handles reasoning models (Qwen3-4B-Thinking) with `<think>` tags
- **Smart JSON Extraction**: Finds and parses the last valid JSON object from LLM output
- **Context-Aware Geocoding**: Intelligent location disambiguation using parent regions
- **Raw Output Recording**: Always captures LLM raw output for debugging, even on failures
- **Checkpoint & Resume**: Evaluation supports resuming from interruptions

### Evaluation System
- **Comprehensive Metrics**: Following CoNLL-2003 NER and TempEval-3 standards
- **Temporal Metrics**: Precision, Recall, F1, normalization accuracy, type matching
- **Spatial Metrics**: Precision, Recall, F1, geocoding success, distance error, accuracy@25km
- **Distributed Evaluation**: Multi-GPU evaluation with Accelerate/DeepSpeed support

## Documentation

- **[CLAUDE.md](CLAUDE.md)**: Comprehensive developer guide and architecture documentation
- **[docs/HF_SERVER_CLIENT.md](docs/HF_SERVER_CLIENT.md)**: Multi-GPU server deployment guide
- **[docs/MCP_INTEGRATION.md](docs/MCP_INTEGRATION.md)**: Claude Desktop MCP server integration

## Server Deployment

### Single Server (Uses all available GPUs via model sharding)

```bash
./scripts/start_server.sh
```

### Multi-GPU (One server per GPU for parallel processing)

```bash
# Auto-detect GPUs
./scripts/start_servers.sh

# Or specify number
STINDEX_NUM_GPUS=4 ./scripts/start_servers.sh
```

### Server Management

```bash
# Check server status (health, GPU usage, processes)
./scripts/check_servers.sh

# Stop servers
./scripts/stop_servers.sh

# Restart servers
./scripts/restart_servers.sh
```

See [docs/HF_SERVER_CLIENT.md](docs/HF_SERVER_CLIENT.md) for detailed deployment instructions.

## Configuration

Configuration files in `cfg/`:
- `extract.yml`: Main configuration (sets LLM provider)
- `evaluate.yml`: Evaluation settings
- `openai.yml`: OpenAI API settings (GPT-4)
- `anthropic.yml`: Anthropic API settings (Claude)
- `hf.yml`: HuggingFace server settings (multi-GPU auto-detection)

### Switching Providers

Edit `cfg/extract.yml`:
```yaml
llm:
  llm_provider: hf  # or openai, anthropic
```

Or specify at runtime:
```python
extractor = STIndexExtractor(config_path="openai")
```

## Project Structure

```
stindex/
├── core/                   # Core extraction logic
│   ├── extraction.py       # STIndexExtractor (main API)
│   └── utils.py           # JSON extraction utilities
├── llm/                    # LLM provider implementations
│   ├── manager.py          # LLM factory
│   ├── openai.py          # OpenAI provider
│   ├── anthropic.py       # Anthropic provider
│   ├── hf.py              # HuggingFace client
│   ├── prompts/           # Prompt templates
│   └── response/          # Pydantic models
├── server/                 # Server implementations
│   ├── hf_server.py       # HuggingFace FastAPI server
│   └── mcp_server.py      # MCP server for Claude Desktop
├── spatio/                 # Spatial processing & geocoding
├── temporal/               # Temporal processing
├── exe/                    # CLI execution logic
│   └── evaluate.py        # Evaluation system
└── cli.py                  # Typer CLI interface

eval/                       # Evaluation scripts
├── evaluate.py            # Main evaluation (sequential/distributed)
└── generate_dataset.py    # Dataset generation

scripts/                    # Helper scripts
├── start_servers.sh       # Multi-GPU server startup
├── check_servers.sh       # Server health monitoring
└── eval_distributed.sh    # Distributed evaluation wrapper
```

## Recent Updates

### Latest Improvements (v0.1.0)
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

### Distributed Evaluation (Multi-GPU)

```bash
# Using convenience script
bash scripts/eval_distributed.sh

# Or directly with Accelerate
accelerate launch --config cfg/deepspeed_zero2.yaml eval/evaluate.py --distributed
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
accelerate launch --config_file configs/deepspeed_zero2.yaml -m mcrag train_reasoner_sft
accelerate launch --config_file configs/deepspeed_zero2.yaml -m mcrag train_reasoner
accelerate launch --config_file configs/deepspeed_zero2.yaml -m mcrag evaluate_reasoner
```

```bash
squeue -u $USER
watch -n 1 "srun --jobid=43462 -n1 bash -lc 'nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits'"

```

## License

MIT License
