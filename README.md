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
```

### Python API

```python
from stindex import STIndexExtractor

extractor = STIndexExtractor()
result = extractor.extract("March 15, 2022 in Broome, Australia")

print(f"Temporal: {len(result.temporal_entities)} entities")
print(f"Spatial: {len(result.spatial_entities)} entities")
```

## Features

- **Single LLM Call**: Unified extraction of both temporal and spatial information
- **Multiple Providers**: OpenAI, Anthropic, and HuggingFace support
- **Multi-GPU Support**: Load balance across multiple GPU servers for high throughput
- **Context-Aware Geocoding**: Intelligent location disambiguation
- **Structured Outputs**: Pydantic models with validation
- **ISO 8601 Normalization**: Standardized temporal representations

## Documentation

- **[CLAUDE.md](CLAUDE.md)**: Comprehensive developer guide and architecture documentation
- **[docs/MULTI_GPU_DEPLOYMENT.md](docs/MULTI_GPU_DEPLOYMENT.md)**: Multi-GPU server deployment guide

## Server Deployment

### Single Server

```bash
./scripts/start_server.sh
```

### Multi-GPU

```bash
./scripts/start_multi_gpu_servers.sh
```

See [docs/MULTI_GPU_DEPLOYMENT.md](docs/MULTI_GPU_DEPLOYMENT.md) for detailed deployment instructions.

## Configuration

Configuration files in `cfg/`:
- `extract.yml`: Main configuration
- `openai.yml`: OpenAI settings
- `anthropic.yml`: Anthropic settings
- `hf.yml`: HuggingFace single-server
- `hf_multi_gpu.yml`: HuggingFace multi-GPU

## License

[Your License Here]
