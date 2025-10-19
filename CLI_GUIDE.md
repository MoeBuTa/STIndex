# STIndex CLI Quick Reference

## Installation

After installing the package, the `stindex` command will be available:

```bash
pip install -e .
```

## Commands

### 1. version - Show version information

```bash
stindex version
```

**Output:**
```
STIndex version 0.1.0
```

---

### 2. extract - Extract from text string

Extract spatiotemporal entities from text provided as a command-line argument.

```bash
stindex extract "TEXT"
```

**Arguments:**
- `TEXT` - Text to extract from (required)

**Options:**
- `-o, --output PATH` - Save to file instead of stdout
- `-f, --format [json|table]` - Output format (default: json)
- `-m, --model TEXT` - LLM model (default: Qwen/Qwen3-8B)
- `--temporal-only` - Extract only temporal entities
- `--spatial-only` - Extract only spatial entities
- `--no-geocoding` - Disable geocoding

**Examples:**

```bash
# Basic extraction (JSON to console)
stindex extract "On March 15, 2022, a cyclone hit Broome, Western Australia."

# Save to file
stindex extract "The conference is in Paris on June 5, 2024." -o results.json

# Table format (human-readable)
stindex extract "World War II ended on September 2, 1945 in Tokyo." -f table

# Extract only dates
stindex extract "The event is on March 15, 2022." --temporal-only

# Extract only locations
stindex extract "The tour stops in Paris, Tokyo, and New York." --spatial-only

# Use OpenAI model
stindex extract "Your text here..." -m gpt-4o-mini
```

**Output (JSON format):**
```json
{
  "temporal_entities": [
    {
      "text": "March 15, 2022",
      "normalized": "2022-03-15",
      "temporal_type": "date",
      "confidence": 0.95
    }
  ],
  "spatial_entities": [
    {
      "text": "Broome",
      "latitude": -17.9614,
      "longitude": 122.2359,
      "locality": "Broome",
      "admin_area": "Western Australia",
      "country": "Australia",
      "confidence": 0.90
    }
  ],
  "temporal_count": 1,
  "spatial_count": 1,
  "processing_time": 2.34
}
```

---

### 3. extract-file - Extract from file

Extract spatiotemporal entities from a text file.

```bash
stindex extract-file FILE_PATH
```

**Arguments:**
- `FILE_PATH` - Path to input text file (required)

**Options:**
- `-o, --output PATH` - Output file for results
- `-f, --format [json|table]` - Output format (default: json)
- `-m, --model TEXT` - LLM model (default: Qwen/Qwen3-8B)

**Examples:**

```bash
# Extract from file
stindex extract-file document.txt

# Save results
stindex extract-file document.txt -o results.json

# Use table format
stindex extract-file article.txt -f table

# Use specific model
stindex extract-file text.txt -m gpt-4o-mini
```

---

### 4. batch - Process multiple files

Process all text files in a directory and save results.

```bash
stindex batch INPUT_DIR OUTPUT_DIR
```

**Arguments:**
- `INPUT_DIR` - Directory containing input files (required)
- `OUTPUT_DIR` - Directory for output files (required)

**Options:**
- `-p, --pattern TEXT` - File pattern to match (default: *.txt)
- `-m, --model TEXT` - LLM model (default: Qwen/Qwen3-8B)

**Examples:**

```bash
# Process all .txt files
stindex batch ./documents/ ./results/

# Process markdown files
stindex batch ./docs/ ./output/ -p "*.md"

# Process with specific model
stindex batch ./texts/ ./output/ -m gpt-4o-mini

# Process specific pattern
stindex batch ./data/ ./output/ -p "report_*.txt"
```

**Output:**
- Creates one JSON file per input file
- Filenames: `{original_name}_stindex.json`
- Shows progress for each file processed

---

## Environment Configuration

Set defaults via environment variables (create a `.env` file):

```bash
# LLM Configuration
STINDEX_LLM_PROVIDER=local              # local/openai/anthropic
STINDEX_MODEL_NAME=Qwen/Qwen3-8B        # Model identifier
STINDEX_TEMPERATURE=0.0

# API Keys (for API-based models)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Feature Toggles
STINDEX_ENABLE_TEMPORAL=true
STINDEX_ENABLE_SPATIAL=true

# Geocoding
STINDEX_GEOCODER=nominatim
STINDEX_USER_AGENT=stindex
```

---

## Common Workflows

### Quick Test
```bash
stindex extract "The meeting is in Tokyo on June 15, 2024." -f table
```

### Process Single Document
```bash
stindex extract-file article.txt -o article_results.json
```

### Batch Processing
```bash
# Create output directory
mkdir -p results/

# Process all documents
stindex batch documents/ results/

# Check results
ls -lh results/
```

### Extract Only Dates
```bash
stindex extract "Born on January 1, 2000. Graduated May 2022." --temporal-only -f table
```

### Extract Only Locations
```bash
stindex extract "Visited Paris, Rome, and Barcelona." --spatial-only -f table
```

### Using OpenAI Models
```bash
export OPENAI_API_KEY="sk-..."
stindex extract "Your text..." -m gpt-4o-mini
```

---

## Troubleshooting

### Command not found
If `stindex` command is not available:
```bash
# Reinstall in editable mode
pip install -e .

# Or use as Python module
python -m stindex.cli --help
```

### Check installation
```bash
which stindex
stindex version
```

### Model loading issues
```bash
# For local models, ensure sufficient memory
# For API models, check API keys
echo $OPENAI_API_KEY
```

---

## Getting Help

```bash
# General help
stindex --help

# Command-specific help
stindex extract --help
stindex extract-file --help
stindex batch --help
```

---

## Output Formats

### JSON (default)
- Structured data with full entity details
- Suitable for programmatic processing
- Can be saved to file with `-o`

### Table
- Human-readable formatted tables
- Best for terminal viewing
- Shows entities with key attributes

---

## Performance Tips

1. **Batch Processing**: Use `stindex batch` for multiple files to reuse model loading
2. **Geocoding Cache**: Locations are cached to speed up repeated queries
3. **Local Models**: First run downloads model (~5GB for Qwen3-8B)
4. **API Models**: Faster but require API keys and have usage costs

---

## See Also

- Full documentation: `CLAUDE.md`
- Test examples: `tests/README.md`
- Python API: Import `from stindex import STIndexExtractor`
