# STIndex Pipeline

End-to-end pipeline orchestrator for STIndex with multiple execution modes.

## Overview

The STIndex pipeline provides a unified interface for:
1. **Full pipeline**: preprocessing → extraction → visualization
2. **Preprocessing only**: scraping → parsing → chunking
3. **Extraction only**: dimensional extraction from preprocessed chunks
4. **Visualization only**: generate visualizations from extraction results

## Quick Start

### Full Pipeline

```python
from stindex import InputDocument, STIndexPipeline

# Create input documents
docs = [
    InputDocument.from_url("https://example.com/article"),
    InputDocument.from_file("/path/to/document.pdf"),
    InputDocument.from_text("Your text here")
]

# Initialize pipeline
pipeline = STIndexPipeline(
    extractor_config="extract",
    dimension_config="dimensions",
    output_dir="data/output"
)

# Run full pipeline
results = pipeline.run_pipeline(docs, visualize=True)
```

## Execution Modes

### 1. Full Pipeline Mode

Runs the complete pipeline: preprocessing → extraction → visualization

```python
pipeline = STIndexPipeline(
    extractor_config="extract",
    dimension_config="dimensions",
    output_dir="data/output"
)

results = pipeline.run_pipeline(
    input_docs=docs,
    save_results=True,
    visualize=True
)
```

**Output:**
- `data/output/chunks/preprocessed_chunks.json` - Preprocessed chunks
- `data/output/results/extraction_results.json` - Extraction results
- `data/output/visualizations/` - Visualizations and summaries

### 2. Preprocessing Only

Run only preprocessing (scraping → parsing → chunking):

```python
pipeline = STIndexPipeline(
    max_chunk_size=2000,
    chunking_strategy="paragraph",
    output_dir="data/output"
)

all_chunks = pipeline.run_preprocessing(
    input_docs=docs,
    save_chunks=True
)
```

**Output:**
- `data/output/chunks/preprocessed_chunks.json`

### 3. Extraction Only

Run extraction on preprocessed chunks:

```python
pipeline = STIndexPipeline(
    extractor_config="extract",
    dimension_config="health_dimensions",
    output_dir="data/output"
)

# Load preprocessed chunks
chunks = pipeline.load_chunks_from_file("data/output/chunks/preprocessed_chunks.json")

# Run extraction
results = pipeline.run_extraction(
    chunks=chunks,
    save_results=True
)
```

**Output:**
- `data/output/results/extraction_results.json`

### 4. Visualization Only

Generate visualizations from extraction results:

```python
pipeline = STIndexPipeline(
    output_dir="data/output"
)

pipeline.run_visualization(
    results="data/output/results/extraction_results.json",
    output_dir="data/output/visualizations"
)
```

**Output:**
- `data/output/visualizations/extraction_summary.json`
- Additional visualizations (map, plots, etc.)

## Configuration

### Pipeline Parameters

```python
pipeline = STIndexPipeline(
    # Extraction config
    extractor_config="extract",              # Config path for DimensionalExtractor
    dimension_config="dimensions",           # Dimension config path

    # Preprocessing config
    max_chunk_size=2000,                     # Maximum chunk size in characters
    chunk_overlap=200,                       # Overlap between chunks
    chunking_strategy="sliding_window",      # "sliding_window", "paragraph", "semantic"
    parsing_method="unstructured",           # "unstructured" or "simple"

    # Output config
    output_dir="data/output",                # Output directory
    save_intermediate=True                   # Save intermediate results
)
```

### Dimension Configuration

The pipeline supports custom dimension configurations for domain-specific extraction:

```python
# Default dimensions (temporal + spatial)
pipeline = STIndexPipeline(
    dimension_config="dimensions"
)

# Health surveillance dimensions
pipeline = STIndexPipeline(
    dimension_config="case_studies/public_health/extraction/config/health_dimensions"
)

# Custom dimensions
pipeline = STIndexPipeline(
    dimension_config="path/to/custom_dimensions"
)
```

## Pipeline Workflow

```
Input Documents
    ↓
[Preprocessing Mode]
    ├─ URL → Web Scraping → HTML
    ├─ FILE → File Reading → Content
    └─ TEXT → Direct Use
    ↓
Document Parsing (HTML/PDF/DOCX → Structured Text)
    ↓
Document Chunking (Long Docs → Manageable Chunks)
    ↓
[Extraction Mode]
    ↓
Dimensional Extraction (LLM + Post-processing)
    ↓
Results (JSON with extracted entities)
    ↓
[Visualization Mode]
    ↓
Visualizations (Maps, Plots, Summaries)
```

## Data Flow

### Chunk Format

```json
{
  "chunk_id": "doc_0_chunk_0",
  "chunk_index": 0,
  "total_chunks": 3,
  "text": "Document text content...",
  "word_count": 150,
  "char_count": 850,
  "document_id": "doc_0",
  "document_title": "Health Alert",
  "document_metadata": {
    "publication_date": "2025-03-15",
    "region": "Australia",
    "source": "health_surveillance"
  },
  "start_char": 0,
  "end_char": 850
}
```

### Extraction Result Format

```json
{
  "chunk_id": "doc_0_chunk_0",
  "chunk_index": 0,
  "document_id": "doc_0",
  "document_title": "Health Alert",
  "extraction": {
    "input_text": "Document text...",
    "success": true,
    "entities": {
      "temporal": [
        {
          "text": "March 15, 2025",
          "normalized": "2025-03-15",
          "dimension_name": "temporal"
        }
      ],
      "spatial": [
        {
          "text": "Perth",
          "latitude": -31.9505,
          "longitude": 115.8605,
          "dimension_name": "spatial"
        }
      ],
      "disease": [
        {
          "text": "measles",
          "category": "infectious_disease",
          "dimension_name": "disease"
        }
      ]
    },
    "processing_time": 2.5,
    "extraction_config": {
      "llm_provider": "hf",
      "model_name": "Qwen/Qwen3-8B",
      "enabled_dimensions": ["temporal", "spatial", "disease"]
    }
  }
}
```

## Advanced Usage

### Resume from Checkpoint

The pipeline automatically saves intermediate results, allowing you to resume:

```python
# First run (interrupted)
pipeline = STIndexPipeline(output_dir="data/output")
try:
    results = pipeline.run_pipeline(docs)
except KeyboardInterrupt:
    print("Interrupted! Results saved to checkpoint.")

# Resume (chunks already saved)
chunks = pipeline.load_chunks_from_file("data/output/chunks/preprocessed_chunks.json")
results = pipeline.run_extraction(chunks)
```

### Custom Output Directories

```python
pipeline = STIndexPipeline(
    output_dir="custom/output/path"
)

# Custom structure:
# custom/output/path/
#   ├── chunks/
#   ├── results/
#   └── visualizations/
```

### Disable Intermediate Saving

```python
pipeline = STIndexPipeline(
    save_intermediate=False  # Don't save chunks to disk
)

# Chunks only in memory
all_chunks = pipeline.run_preprocessing(docs, save_chunks=False)
```

## Integration Examples

### Case Study Integration

See `case_studies/public_health/scripts/run_case_study.py` for a complete example:

```python
# Create input documents from URLs
docs = [
    InputDocument.from_url(
        url="https://health.wa.gov.au/news/2025/measles-alert",
        metadata={"source": "wa_health_au", "disease": "measles"}
    )
]

# Run pipeline with health dimensions
pipeline = STIndexPipeline(
    dimension_config="case_studies/public_health/extraction/config/health_dimensions",
    output_dir="case_studies/public_health/data"
)

results = pipeline.run_pipeline(docs, visualize=True)
```

### CLI Integration

```bash
# Full pipeline
python case_studies/public_health/scripts/run_case_study.py --mode pipeline --input-mode url

# Preprocessing only
python case_studies/public_health/scripts/run_case_study.py --mode preprocessing --input-mode file

# Extraction only
python case_studies/public_health/scripts/run_case_study.py --mode extraction --chunks-file data/chunks.json

# Visualization only
python case_studies/public_health/scripts/run_case_study.py --mode visualization --results-file data/results.json
```

## Error Handling

The pipeline handles errors gracefully:

```python
# Preprocessing errors: logged, pipeline continues with other documents
all_chunks = pipeline.run_preprocessing(docs)

# Extraction errors: stored in results with error field
results = pipeline.run_extraction(chunks)

for result in results:
    if result.get('error'):
        print(f"Failed: {result['chunk_id']}: {result['error']}")
    else:
        print(f"Success: {result['chunk_id']}")
```

## Performance Tips

1. **Batch processing**: Process multiple documents in one pipeline run
2. **Chunking strategy**: Use "paragraph" for better semantic coherence
3. **Intermediate saving**: Enable for large datasets (allows resume)
4. **Parallel extraction**: Use distributed mode for large-scale extraction (see `eval/evaluate.py`)

## Dependencies

Same as preprocessing module, plus:
- `stindex.core.dimensional_extraction` - DimensionalExtractor
- `stindex.preprocessing` - Preprocessor, InputDocument, DocumentChunk

## Next Steps

- See `stindex/preprocessing/README.md` for preprocessing details
- See `case_studies/public_health/` for complete example
- See `CLAUDE.md` for full architecture overview
