# STIndex - Spatiotemporal Information Extraction

[![PyPI version](https://img.shields.io/pypi/v/stindex.svg)](https://pypi.org/project/stindex/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Home Homepage](https://img.shields.io/badge/Home-Homepage-green.svg)](https://stindex.ai4wa.com/)
[![Demo Dashboard](https://img.shields.io/badge/Demo-Dashboard-green.svg)](https://stindex.ai4wa.com/dashboard)

STIndex is a multi-dimensional information extraction system that uses LLMs to extract temporal, spatial, and custom dimensional data from unstructured text. Features an end-to-end pipeline with preprocessing, extraction, auto schema discovery, and evaluation.

**[Try the Demo Dashboard](https://stindex.ai4wa.com/)**

## Quick Start

### Installation

**Option 1: macOS / Apple Silicon (CPU-only)**

```bash
conda env create -f environment-mac.yml
conda activate stindex
pip install -e . --no-deps
python -m spacy download en_core_web_sm
```

This installs all API-based providers (OpenAI, Anthropic, Gemini, DeepSeek) without GPU dependencies. The `--no-deps` flag skips pulling GPU-only packages (vLLM, flashinfer, deepspeed) that are absent from `environment-mac.yml`.

**Option 2: Linux / GPU (CUDA 12.4)**

```bash
conda env create -f environment.yml
conda activate stindex
pip install -e .
python -m spacy download en_core_web_sm
```

The conda environment includes PyTorch 2.6.0 with CUDA 12.4, vLLM, and all dependencies.

**Option 3: pip only**

```bash
pip install stindex
python -m spacy download en_core_web_sm
```

### API Keys

Copy the environment template and add your API keys:

```bash
cp .env.example .env
```

Edit `.env`:

```bash
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
DEEPSEEK_API_KEY=sk-...        # DeepSeek — https://platform.deepseek.com
# No key needed for HuggingFace (local server)
```

### Basic Extraction

```bash
# Extract spatiotemporal entities (uses default config + PROVIDER_DEFAULTS)
stindex extract "On March 15, 2022, a cyclone hit Broome, Western Australia."

# Use a specific provider
stindex extract "Text here..." --config openai      # or anthropic, gemini, deepseek
stindex extract "Text here..." --config deepseek    # DeepSeek (great for macOS)

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

---

## Multi-Dimensional Extraction

STIndex supports extracting any combination of dimensions beyond temporal and spatial. Define custom schemas in YAML:

```python
from stindex import DimensionalExtractor

# Use a custom dimension schema (health surveillance example)
extractor = DimensionalExtractor(
    config_path="hf",
    dimension_config_path="case_studies/public_health/config/health_dimensions",
)
result = extractor.extract(
    "WA Health confirmed a measles outbreak at Port Hedland Hospital on Feb 12, 2024."
)

# Access all extracted dimensions
for dim_name, entities in result.entities.items():
    print(f"{dim_name}: {len(entities)} entities")
    for e in entities:
        print(f"  - {e.get('text')}: {e.get('category', e.get('normalized', ''))}")
```

**Included schemas:**
- `case_studies/public_health/config/health_dimensions.yml` — temporal, spatial, event_type, venue_type, disease
- `case_studies/wa_news/config/wa_dimensions.yml` — temporal, spatial, person, organization, event_type

### Custom Dimension Schema

```yaml
dimensions:
  temporal:
    enabled: true
  spatial:
    enabled: true
  disease:
    enabled: true
    description: Disease or health condition mentioned
    extraction_type: categorical
    hierarchy:
      - level: category
        values: [measles, influenza, covid19, pertussis, other]
      - level: disease_code
        description: ICD-10 code if applicable
```

---

## Auto Schema Discovery

Automatically discover dimensional schemas from Q&A datasets using LLM-powered clustering and extraction:

```python
from stindex.pipeline.discovery_pipeline import SchemaDiscoveryPipeline

llm_config = {
    "llm_provider": "hf",
    "model_name": "Qwen3-4B-Instruct-2507",
    "temperature": 0.0,
    "max_tokens": 4096,
    "base_url": "http://localhost:8001",
}

pipeline = SchemaDiscoveryPipeline(
    llm_config=llm_config,
    n_clusters=10,
    batch_size=30,
)

schema = pipeline.discover_schema(
    questions_file="data/questions.jsonl",
    output_dir="data/schema_discovery_output",
)
```

**CLI:**
```bash
python -m stindex.pipeline.discovery_pipeline \
    --questions data/questions.jsonl \
    --output-dir data/schema_discovery \
    --n-clusters 10 --llm-provider hf --batch-size 30
```

**Output:**
```
data/schema_discovery/
├── cluster_assignments.csv          # Question → cluster mapping
├── cluster_samples.json             # Sample questions per cluster
├── cluster_*_result.json            # Per-cluster results
├── final_schema.json                # Full discovered schema
├── final_schema.yml                 # YAML version
├── extraction_schema.yml            # Slim schema for extraction
└── cot/                             # Chain-of-thought reasoning logs
```

**Supported datasets:** MIRAGE, MedCorp, HotpotQA, 2WikiMQA, MuSiQue, or any JSONL with `{"question": "..."}` format.

---

## LLM Providers

Provider defaults are defined in `stindex/utils/config.py` and can be overridden at runtime via CLI flags or Python API.

| Provider | Default Model | Config File |
|----------|--------------|-------------|
| `openai` | `gpt-4o-mini` | `cfg/extraction/inference/openai.yml` |
| `anthropic` | `claude-sonnet-4-5-20250929` | `cfg/extraction/inference/anthropic.yml` |
| `gemini` | `gemini-2.0-flash` | `cfg/extraction/inference/gemini.yml` |
| `deepseek` | `deepseek-chat` | `cfg/extraction/inference/deepseek.yml` |
| `hf` | `Qwen3-4B-Instruct-2507` | `cfg/extraction/inference/hf.yml` |

Select provider in `cfg/extraction/inference/extract.yml`:
```yaml
llm:
  llm_provider: deepseek   # or openai, anthropic, gemini, hf
```

Or override everything at runtime:
```bash
stindex extract "Text..." --config openai --model gpt-4o --temperature 0.0

stindex extract "Text..." --config deepseek   # uses deepseek-chat by default

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
    gpu_memory_utilization: 0.85       # Adjust for your GPU VRAM
    max_model_len: 8192                # Max sequence length
```

```bash
# Start server (reads cfg/extraction/inference/hf.yml)
./scripts/server/deploy_ms_swift.sh

# Stop server
./scripts/server/stop_ms_swift.sh

# Monitor
tail -f logs/hf_server.log
```

**Tested configurations:**
- RTX 5080 (16GB): `gpu_memory_utilization: 0.85`, `max_model_len: 8192`
- H100 (80GB): `gpu_memory_utilization: 0.9`, `max_model_len: 16384`

---

## MCP Server

STIndex exposes its extraction pipeline as a **Model Context Protocol (MCP)** server over SSE/HTTP,
so that Claude Desktop, Cursor, docs2synth, and any other MCP-compatible client can send documents
and receive structured JSON back.

### Tools

| Tool | Description |
|------|-------------|
| `extract_text` | Extract entities from a plain-text or HTML string |
| `extract_file` | Extract from a local file path (PDF, DOCX, TXT, HTML) |
| `extract_url` | Scrape a web URL and extract entities |
| `extract_content` | Extract from base64-encoded file bytes (for remote clients) |
| `analyze` | Cluster and analyse a prior extraction result |

### Quick Start (local)

```bash
pip install -e .
stindex-mcp --port 8008          # SSE transport, all interfaces
# or
python -m stindex.mcp_server --port 8008 --transport sse
```

Test with MCP Inspector:
```bash
npx @modelcontextprotocol/inspector http://localhost:8008/sse
```

### Mac Mini / Server Deployment

The recommended setup runs `stindex-mcp` as a persistent **launchd** service behind an **Nginx**
reverse proxy so clients can reach it at a clean HTTPS URL like
`https://mcp.yourdomain.com/sse`.

#### 1 — Create a launchd service

Create `/Library/LaunchDaemons/com.stindex.mcp.plist` (runs as root, survives reboots):

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.stindex.mcp</string>

  <key>ProgramArguments</key>
  <array>
    <!-- adjust to your venv / conda env path -->
    <string>/opt/miniconda3/envs/stindex/bin/stindex-mcp</string>
    <string>--host</string>
    <string>127.0.0.1</string>
    <string>--port</string>
    <string>8008</string>
    <string>--transport</string>
    <string>sse</string>
  </array>

  <!-- working directory must contain cfg/ -->
  <key>WorkingDirectory</key>
  <string>/path/to/STIndex</string>

  <!-- API keys -->
  <key>EnvironmentVariables</key>
  <dict>
    <key>OPENAI_API_KEY</key>
    <string>sk-...</string>
    <key>ANTHROPIC_API_KEY</key>
    <string>sk-ant-...</string>
    <key>DEEPSEEK_API_KEY</key>
    <string>sk-...</string>
  </dict>

  <key>StandardOutPath</key>
  <string>/var/log/stindex-mcp.log</string>
  <key>StandardErrorPath</key>
  <string>/var/log/stindex-mcp.err</string>

  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
</dict>
</plist>
```

Load and start:
```bash
sudo launchctl load /Library/LaunchDaemons/com.stindex.mcp.plist
sudo launchctl start com.stindex.mcp

# Check status
sudo launchctl list | grep stindex
tail -f /var/log/stindex-mcp.log
```

#### 2 — Nginx reverse proxy with HTTPS

Install Nginx and [Certbot](https://certbot.eff.org/) (e.g. via Homebrew):

```bash
brew install nginx certbot
```

Create `/opt/homebrew/etc/nginx/servers/stindex-mcp.conf`:

```nginx
server {
    listen 80;
    server_name mcp.yourdomain.com;

    # Redirect HTTP → HTTPS
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl;
    server_name mcp.yourdomain.com;

    ssl_certificate     /etc/letsencrypt/live/mcp.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/mcp.yourdomain.com/privkey.pem;

    # SSE requires long-lived connections — disable buffering
    proxy_buffering off;
    proxy_cache off;
    proxy_read_timeout 3600s;
    proxy_send_timeout 3600s;
    keepalive_timeout  3600s;

    location / {
        proxy_pass         http://127.0.0.1:8008;
        proxy_http_version 1.1;
        proxy_set_header   Host              $host;
        proxy_set_header   X-Real-IP         $remote_addr;
        proxy_set_header   X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header   X-Forwarded-Proto $scheme;

        # Required for SSE
        proxy_set_header   Connection        '';
        chunked_transfer_encoding on;
    }
}
```

Issue a certificate and reload:
```bash
sudo certbot certonly --standalone -d mcp.yourdomain.com
sudo nginx -t && sudo nginx -s reload
```

The MCP server is now reachable at `https://mcp.yourdomain.com/sse`.

#### 3 — Connect clients

**Claude Desktop** (`~/.config/claude/claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "stindex": {
      "url": "https://mcp.yourdomain.com/sse"
    }
  }
}
```

**Cursor** — add to MCP settings:
```
https://mcp.yourdomain.com/sse
```

**docs2synth / any HTTP MCP client** — point it at:
```
https://mcp.yourdomain.com/sse
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
│   │   ├── anthropic.yml      # Provider selector (llm_provider: anthropic)
│   │   ├── gemini.yml         # Provider selector (llm_provider: gemini)
│   │   └── deepseek.yml       # Provider selector (llm_provider: deepseek)
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
    --dataset eval_data/doc_500.json \
    --config hf --model Qwen3-4B-Instruct-2507 --base-url http://localhost:8001

# Limit to 20 samples for quick testing
stindex evaluate --dataset eval_data/doc_500.json \
    --config openai --sample-limit 20
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
