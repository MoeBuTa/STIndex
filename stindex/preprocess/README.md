# STIndex Preprocessing Module

Generic preprocessing pipeline for STIndex that handles web URLs, local files, and raw text.

## Overview

The preprocessing module provides a unified interface for:
- **Web scraping**: Fetch content from HTTP/HTTPS URLs
- **Document parsing**: Parse HTML, PDF, DOCX, TXT, and other formats
- **Document chunking**: Split long documents into manageable chunks

## Quick Start

### Basic Usage

```python
from stindex import InputDocument, Preprocessor

# Create input document
doc = InputDocument.from_url(
    url="https://example.com/article",
    metadata={"source": "example"}
)

# Initialize preprocessor
preprocessor = Preprocessor(
    max_chunk_size=2000,
    chunking_strategy="paragraph"
)

# Process document
chunks = preprocessor.process(doc)

print(f"Generated {len(chunks)} chunks")
for chunk in chunks:
    print(f"Chunk {chunk.chunk_index}: {chunk.word_count} words")
```

### Input Types

#### 1. Web URL

```python
doc = InputDocument.from_url(
    url="https://health.gov/alerts/measles-2025",
    document_id="measles_alert_1",
    title="Measles Alert 2025",
    metadata={
        "source": "health_surveillance",
        "publication_date": "2025-03-15",
        "region": "Australia"
    }
)
```

#### 2. Local File

```python
doc = InputDocument.from_file(
    file_path="/path/to/document.pdf",
    document_id="health_report_1",
    metadata={"source": "local_reports"}
)
```

#### 3. Raw Text

```python
doc = InputDocument.from_text(
    text="On March 15, 2022, a measles outbreak occurred in Perth, Australia.",
    title="Measles Alert",
    metadata={
        "publication_date": "2022-03-15",
        "region": "Australia"
    }
)
```

### Batch Processing

```python
docs = [
    InputDocument.from_url("https://example.com/article1"),
    InputDocument.from_file("/path/to/document.pdf"),
    InputDocument.from_text("Your text here")
]

preprocessor = Preprocessor()
all_chunks = preprocessor.process_batch(docs)

# all_chunks is a list of lists (one list per document)
for i, chunks in enumerate(all_chunks):
    print(f"Document {i}: {len(chunks)} chunks")
```

## Configuration

### Preprocessor Parameters

```python
preprocessor = Preprocessor(
    max_chunk_size=2000,         # Maximum chunk size in characters
    chunk_overlap=200,            # Overlap between chunks (context preservation)
    chunking_strategy="sliding_window",  # "sliding_window", "paragraph", or "semantic"
    parsing_method="unstructured",       # "unstructured" or "simple"
    user_agent="STIndex-Research/1.0",   # User agent for web scraping
    rate_limit=2.0                       # Seconds between web requests
)
```

### Chunking Strategies

1. **sliding_window** (default): Fixed-size chunks with overlap
   - Best for general use
   - Maintains context across chunks
   - Breaks at sentence boundaries when possible

2. **paragraph**: Chunk by paragraphs
   - Respects document structure
   - More semantic coherence
   - Variable chunk sizes (up to max_chunk_size)

3. **semantic**: Semantic chunking using embeddings
   - Not yet implemented (falls back to paragraph)
   - Future: Use sentence embeddings for better coherence

### Parsing Methods

1. **unstructured** (default): Uses the `unstructured` library
   - Supports HTML, PDF, DOCX, TXT, and more
   - Extracts tables and structured content
   - Better for complex documents
   - Requires: `pip install 'unstructured[local-inference]'`

2. **simple**: Uses BeautifulSoup for HTML, plain text reader for files
   - Faster and lighter
   - Good for simple HTML and text files
   - No PDF/DOCX support
   - Requires: `pip install beautifulsoup4`

## Components

### 1. InputDocument

Data model for input documents:

```python
@dataclass
class InputDocument:
    input_type: InputType  # URL, FILE, or TEXT
    content: str           # URL, file path, or raw text
    metadata: Dict[str, Any]
    document_id: Optional[str]
    title: Optional[str]
```

### 2. DocumentChunk

Data model for document chunks:

```python
@dataclass
class DocumentChunk:
    chunk_id: str          # e.g., "doc1_chunk_0"
    chunk_index: int
    total_chunks: int
    text: str
    word_count: int
    char_count: int
    document_id: str
    document_title: str
    document_metadata: Dict[str, Any]
    start_char: int        # Position in original document
    end_char: int
    previous_chunk_summary: Optional[str]
```

### 3. WebScraper

Web scraping with rate limiting:

```python
from stindex.preprocessing.scraping import WebScraper

scraper = WebScraper(
    user_agent="STIndex-Research/1.0",
    rate_limit=2.0,
    timeout=30
)

html, error = scraper.scrape("https://example.com")
if error:
    print(f"Scraping failed: {error}")
```

### 4. DocumentParser

Document parsing:

```python
from stindex.preprocessing.parsing import DocumentParser

parser = DocumentParser(parsing_method="unstructured")

# Parse HTML string
parsed_doc = parser.parse_html_string(
    html_content=html,
    document_id="doc1",
    title="My Document"
)

# Parse file
parsed_doc = parser.parse_file(
    file_path="/path/to/document.pdf",
    document_id="doc2"
)

# Parse raw text
parsed_doc = parser.parse_text(
    text="Your text here",
    document_id="doc3"
)
```

### 5. DocumentChunker

Document chunking:

```python
from stindex.preprocessing.chunking import DocumentChunker

chunker = DocumentChunker(
    max_chunk_size=2000,
    overlap=200,
    strategy="paragraph"
)

chunks = chunker.chunk_text(
    text="Long document text...",
    document_id="doc1",
    title="My Document"
)
```

## Advanced Usage

### Custom Metadata

```python
doc = InputDocument.from_url(
    url="https://health.gov/alerts/measles",
    metadata={
        "publication_date": "2025-03-15",
        "source_location": "Perth, Australia",
        "source": "health_surveillance",
        "disease": "measles",
        "severity": "high",
        # Any custom fields
        "custom_field": "value"
    }
)
```

Metadata is preserved through the pipeline and passed to extraction.

### Skip Chunking

```python
# Process without chunking (return single chunk with full content)
chunks = preprocessor.process(doc, skip_chunking=True)
assert len(chunks) == 1
```

### Save/Load Chunks

```python
import json

# Save chunks
chunks_data = [chunk.to_dict() for chunk in chunks]
with open("chunks.json", "w") as f:
    json.dump(chunks_data, f, indent=2)

# Load chunks
with open("chunks.json", "r") as f:
    chunks_data = json.load(f)
chunks = [DocumentChunk.from_dict(d) for d in chunks_data]
```

## Integration with Pipeline

The preprocessing module integrates with `STIndexPipeline` for end-to-end processing:

```python
from stindex import InputDocument, STIndexPipeline

docs = [
    InputDocument.from_url("https://example.com/article"),
    InputDocument.from_file("/path/to/document.pdf"),
]

pipeline = STIndexPipeline(
    extractor_config="extract",
    dimension_config="dimensions",
    max_chunk_size=2000,
    chunking_strategy="paragraph"
)

# Run full pipeline: preprocessing → extraction → visualization
results = pipeline.run_pipeline(docs)

# Or run preprocessing only
chunks = pipeline.run_preprocessing(docs)
```

See `stindex/pipeline/README.md` for more details on the full pipeline.

## Error Handling

```python
from stindex import InputDocument, Preprocessor

preprocessor = Preprocessor()

docs = [
    InputDocument.from_url("https://example.com/valid"),
    InputDocument.from_url("https://invalid-url-404.com"),
    InputDocument.from_file("/path/to/missing-file.pdf"),
]

# Batch processing continues even if some documents fail
all_chunks = preprocessor.process_batch(docs)

# Check results
print(f"Successfully processed {len(all_chunks)}/{len(docs)} documents")
```

## Dependencies

### Required
- `requests` - Web scraping
- `beautifulsoup4` - HTML parsing
- `loguru` - Logging

### Optional
- `unstructured[local-inference]` - Advanced document parsing (recommended)
- `nltk` - Required by unstructured (auto-downloaded)

Install all dependencies:
```bash
pip install requests beautifulsoup4 loguru
pip install 'unstructured[local-inference]'  # Optional but recommended
```

## Examples

See:
- `case_studies/public_health/scripts/run_case_study.py` - Full example
- `stindex/pipeline/pipeline.py` - Pipeline integration
- Tests (coming soon)
