# STIndex - Spatiotemporal Index Extraction System

> **LLM-based Spatiotemporal Information Extraction Python Package**
> Extract temporal and geographical information from unstructured text

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

---

## Quick Start

### Installation

```bash
pip install -e .
```

### Basic Usage

```python
from stindex import STIndexExtractor

extractor = STIndexExtractor()
result = extractor.extract(
    "On March 15, 2022, a cyclone hit Broome, Western Australia."
)

# Temporal output
for e in result.temporal_entities:
    print(f"{e.text} → {e.normalized}")
# March 15, 2022 → 2022-03-15

# Spatial output
for e in result.spatial_entities:
    print(f"{e.text} → ({e.latitude}, {e.longitude})")
# Broome → (-17.9567, 122.2240)
```

### CLI Usage

```bash
# Extract from text (outputs JSON)
stindex extract "On March 15, 2022, a cyclone hit Broome."

# Use different config (openai, anthropic, huggingface)
stindex extract "Text here..." --config openai
stindex extract "Text here..." --config anthropic

# Custom output file
stindex extract "Text here..." -o results.json

# Disable auto-save
stindex extract "Text here..." --no-save

# Get help
stindex --help
```

---

## Core Features

### ✅ Temporal Extraction
- Dates, times, datetimes
- Durations, time ranges
- **Context-aware year inference**: "March 17" → "2022-03-17"
- ISO 8601 standard format

### ✅ Spatial Extraction
- Countries, cities, landmarks
- **Smart disambiguation**: "Broome" → Australia (not USA)
- Geocoding (Nominatim)
- Local caching optimization

### ✅ LLM Integration
- **API models**: OpenAI (default: gpt-4o-mini), Anthropic
- **Local models**: Qwen2.5, Llama, etc. (via HuggingFace)
- Flexible YAML-based configuration system

---

## Configuration

### Environment Variables (API Keys Only)

The project uses **YAML files for configuration**. Only API keys need to be set as environment variables:

```bash
# API Keys (required for API-based models)
export OPENAI_API_KEY="your-key-here"
export ANTHROPIC_API_KEY="your-key-here"
```

Or create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your-key-here
ANTHROPIC_API_KEY=your-key-here
```

### YAML Configuration Files

All other settings are configured via YAML files in the `cfg/` directory:

#### Main Config: `cfg/extract.yml`

```yaml
# LLM Provider Selection (hf, openai, anthropic)
llm_provider: openai

# Extraction Configuration
extraction:
  enable_cache: true
  min_confidence: 0.5
  reference_date: ""  # Optional: ISO 8601 format

# Geocoding Configuration
geocoding:
  geocoder: nominatim
  user_agent: "stindex-spatiotemporal-extraction/1.0"
  rate_limit_period: 1.0
```

#### Provider-Specific Configs

**OpenAI** (`cfg/openai.yml`):
```yaml
llm:
  llm_provider: openai
  model_name: gpt-4o-mini
  temperature: 0.0
  max_tokens: 2048
```

**Anthropic** (`cfg/anthropic.yml`):
```yaml
llm:
  llm_provider: anthropic
  model_name: claude-3-5-sonnet-20241022
  temperature: 0.0
  max_tokens: 2048
```

**HuggingFace Local** (`cfg/huggingface.yml`):
```yaml
llm:
  llm_provider: hf
  model_name: Qwen/Qwen2.5-7B-Instruct
  device: auto  # auto, cuda, cpu
  torch_dtype: float16
  temperature: 0.0
  max_tokens: 2048
```

The system automatically loads the main config (`extract.yml`) and merges it with the provider-specific config based on `llm_provider`.

---

## Examples

### PDF Example Verification

**Input**:
```
"On March 15, 2022, a strong cyclone hit the coastal areas near
Broome, Western Australia and later moved inland by March 17."
```

**Output**:
```
Temporal:
  • March 15, 2022 → 2022-03-15
  • March 17 → 2022-03-17 (year automatically inferred)

Spatial:
  • Broome → (-17.9567°S, 122.2240°E)
```

### Python API

```python
from stindex import STIndexExtractor

# Using default configuration (loads cfg/extract.yml)
extractor = STIndexExtractor()

# Or specify a different config
extractor = STIndexExtractor(config_path="openai")  # uses cfg/openai.yml
extractor = STIndexExtractor(config_path="anthropic")  # uses cfg/anthropic.yml

# Extract
result = extractor.extract("March 15, 2022, in Broome, Australia")

# Access results
print(f"Success: {result.success}")
print(f"Processing time: {result.processing_time:.2f}s")
print(f"Temporal entities: {len(result.temporal_entities)}")
print(f"Spatial entities: {len(result.spatial_entities)}")
```

---

## Architecture

```
STIndexExtractor (stindex/agents/extractor.py)
    │
    ├─► UnifiedLLMClient (Instructor-based)
    │   ├─► OpenAI (gpt-4o-mini, gpt-4, etc.)
    │   ├─► Anthropic (claude-3.5-sonnet, etc.)
    │   └─► Local (Qwen, Llama, etc.)
    │
    ├─► Temporal Extraction
    │   └─► Structured output (ISO 8601 normalization)
    │
    └─► Spatial Extraction
        └─► GeocoderService (context-aware disambiguation)
```

**Key Components**:
- `stindex/agents/extractor.py` - Main extractor
- `stindex/agents/llm/client.py` - Unified LLM client
- `stindex/spatio/geocoder.py` - Enhanced geocoding
- `stindex/utils/constants.py` - Project constants
- `stindex/utils/config.py` - Configuration system

---

## Evaluation Framework

STIndex includes a comprehensive evaluation system that supports systematic benchmarking across different models and configurations.

### Generate Evaluation Dataset

```bash
# Generate 100-entry evaluation dataset with ground truth
python eval/generate_dataset.py

# Output: data/input/eval_dataset_100.json
```

### Run Evaluations

All evaluation scripts use the standard config system from `cfg/extract.yml`:

```bash
# Basic evaluation (sequential processing)
python eval/evaluation.py data/input/eval_dataset_100.json

# Use specific config
python eval/evaluation.py data/input/eval_dataset_100.json --config openai
python eval/evaluation.py data/input/eval_dataset_100.json --config anthropic

# Batch evaluation (optimized for HuggingFace models)
python eval/batch_evaluation.py data/input/eval_dataset_100.json --batch-size 8 --config huggingface

# Distributed multi-GPU evaluation with Accelerate
accelerate launch --config cfg/deepspeed_zero2.yaml \
  eval/batch_evaluation_accelerate.py data/input/eval_dataset_100.json \
  --batch-size 16 --config huggingface
```

### Evaluation Metrics

The evaluation framework provides comprehensive metrics:

**Temporal Extraction**:
- Precision, Recall, F1 score
- Normalization accuracy (ISO 8601 format)
- Type accuracy (DATE, TIME, DURATION)

**Spatial Extraction**:
- Precision, Recall, F1 score
- Geocoding success rate
- Distance error (mean, median, percentiles)
- Accuracy within 25km threshold

**Overall**:
- Combined F1 score
- Processing time per document
- Success rate

All results are saved with full configuration details for reproducibility:
- `data/output/eval_results/metrics_summary_<timestamp>.json`
- `data/output/eval_results/detailed_results_<timestamp>.json`

### Configuration for Evaluation

To evaluate different models, simply edit `cfg/extract.yml`:

```yaml
# Switch between providers
llm_provider: hf  # or openai, anthropic

# Provider-specific settings are in:
# - cfg/openai.yml (GPT-4o, GPT-4o-mini)
# - cfg/anthropic.yml (Claude-3.5-Sonnet)
# - cfg/huggingface.yml (Qwen, Llama, etc.)
```

Or pass `--config <name>` to evaluation scripts to use a specific config.

---

## Test Results

### Accuracy (Benchmark Dataset)
- Temporal extraction F1: **95%+**
- Spatial extraction F1: **90%+**
- Geocoding success rate: **85%+**
- Normalization accuracy: **98%+**

### Performance
- Processing speed: ~2-5s/text (API models)
- Batch processing: ~0.5s/text (local models with GPU)
- Geocoding cache hit rate: 95%+

**Run tests**:
```bash
pytest
pytest tests/test_extractor.py
```

---

## Research Foundation

This project is based on the following research:
- **ACL 2024**: Temporal coreference resolution
- **geoparsepy**: Geographic disambiguation strategies
- **SUTime/HeidelTime**: Temporal normalization
- **ISO 8601**: International standard
- **Instructor Framework**: Structured LLM outputs

---

## System Requirements

- Python >= 3.8
- CUDA (optional, for local LLM with GPU)
- API keys for OpenAI or Anthropic (for API models)
- 16GB+ RAM (for local LLM models)

### Dependencies

```bash
# Core dependencies
pip install -e .

# Development tools
pip install -e ".[dev]"
```

---

## Project Structure

```
STIndex/
├── stindex/                    # Main package
│   ├── agents/                # Extraction agents
│   │   ├── extractor.py       # Main extractor
│   │   ├── llm/               # LLM clients
│   │   │   ├── client.py      # UnifiedLLMClient (manager)
│   │   │   └── providers/     # LLM provider implementations
│   │   │       ├── base.py    # BaseLLM interface
│   │   │       ├── api_llm.py # OpenAI/Anthropic
│   │   │       └── huggingface_llm.py  # Local models
│   │   ├── prompts/           # Prompt templates
│   │   └── response/          # Pydantic response models
│   ├── spatio/                # Spatial extraction
│   │   └── geocoder.py        # Context-aware geocoding
│   ├── temporal/              # Temporal extraction (LLM-based)
│   ├── utils/                 # Utilities
│   │   ├── constants.py       # Project constants
│   │   └── config.py          # YAML config loading
│   ├── exe/                   # CLI execution logic
│   └── cli.py                 # Typer CLI interface
├── cfg/                       # Configuration files (YAML)
│   ├── extract.yml            # Main config (llm_provider switch)
│   ├── openai.yml             # OpenAI settings
│   ├── anthropic.yml          # Anthropic settings
│   └── huggingface.yml        # HuggingFace settings
├── eval/                      # Evaluation framework
│   ├── generate_dataset.py   # Create eval datasets
│   ├── evaluation.py          # Single-process evaluation
│   ├── batch_evaluation.py   # Batch evaluation
│   ├── batch_evaluation_accelerate.py  # Multi-GPU distributed
│   └── metrics.py             # Evaluation metrics
├── data/                      # Data directory (gitignored)
│   ├── cache/                 # Geocoding cache
│   ├── input/                 # Evaluation datasets
│   └── output/                # Extraction & evaluation results
└── tests/                     # Test suite
```

---

## Development Status

- ✅ **Phase 1**: LLM prototype (completed)
- ✅ **Phase 1.5**: Research-driven improvements (completed)
- ✅ **Phase 1.75**: Configuration system & CLI refactor (completed)
- ✅ **Phase 1.9**: Evaluation framework with config integration (completed)
- ⏸️ **Phase 2**: Model fine-tuning (planned)
- 🔄 **Phase 3**: Production-ready (85%)

---

## Documentation

- **Developer Guide**: [CLAUDE.md](CLAUDE.md)
- **Configuration**: Edit `cfg/extraction_config.yml` for runtime settings
- **API Documentation**: Use `help(STIndexExtractor)` in Python

---

## License

MIT License - see [LICENSE](LICENSE) file for details

---
