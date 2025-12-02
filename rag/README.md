# STIndex RAG Pipeline Implementation

This document provides a detailed overview of the RAG (Retrieval-Augmented Generation) pipeline implementation for STIndex, designed for multi-hop question answering using the GRPO training dataset.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        STIndex RAG Pipeline                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐               │
│  │   Data       │───▶│   Vector     │───▶│   RAG        │               │
│  │   Processing │    │   Ingestion  │    │   Retriever  │               │
│  └──────────────┘    └──────────────┘    └──────────────┘               │
│         │                   │                   │                        │
│         ▼                   ▼                   ▼                        │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐               │
│  │ GRPO Dataset │    │ FAISS Index  │    │   Answer     │               │
│  │ (916K docs)  │    │ + Metadata   │    │   Generator  │               │
│  └──────────────┘    └──────────────┘    └──────────────┘               │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## Data Flow

### Stage 1: Data Preprocessing (pikerag)
Downloads and formats multi-hop QA datasets:
- **HotpotQA**: Multi-hop reasoning questions
- **2WikiMQA**: Two-document Wikipedia questions
- **MuSiQue**: Multi-step reasoning questions

```bash
# Run preprocessing
python rag/preprocess/pikerag/main.py data_process/config/datasets.yaml
```

### Stage 2: Training Data Generation
Generates GRPO and SFT training data with embedded documents:

```bash
python rag/preprocess/dataset/dataset_generator.py \
    --datasets hotpotqa two_wiki musique \
    --train-limits 10000 10000 5000 \
    --grpo-output-name grpo_25000
```

**Output**: `data/data_train/grpo/grpo_25000.jsonl`
- 275,827 samples (not just 25000 - naming convention)
- Each sample contains question + embedded reference documents

### Stage 3: Corpus Extraction from GRPO
Extracts unique documents from GRPO for RAG retrieval:

```bash
python rag/preprocess/extract_grpo_docs.py \
    --input data/data_train/grpo/grpo_25000.jsonl \
    --output data/corpus/grpo/chunks.jsonl
```

**Output**: `data/corpus/grpo/chunks.jsonl`
- **916,618 unique documents** extracted
- Documents are Wikipedia passages used in multi-hop QA

### Stage 4: Vector Ingestion
Encodes documents into FAISS vector index:

```bash
python -m rag.preprocess.ingestion.vector_ingest \
    --input data/corpus/grpo/chunks.jsonl \
    --output data/vector/rag \
    --model "sentence-transformers/all-MiniLM-L6-v2" \
    --batch-size 128 \
    --index-type flat
```

**Output**: `data/vector/rag/`
- `faiss_index.bin` - FAISS vector index
- `id_mapping.json` - Chunk ID to FAISS index mapping
- `chunks_metadata.jsonl` - Document metadata with text

---

## Module Implementation Details

### 1. Document Extraction (`rag/preprocess/extract_grpo_docs.py`)

Extracts unique documents embedded in GRPO training data.

**Key Functions:**
```python
def parse_documents_from_content(content: str) -> List[Tuple[int, str, str]]:
    """Parse documents from user message content in GRPO format.

    GRPO format: <documents>[N] Title: Content [N+1] Title: Content...</documents>

    Returns:
        List of (doc_num, title, text) tuples
    """
    pattern = r'\[(\d+)\]\s*([^:]+):\s*(.+?)(?=\[\d+\]|$)'
    matches = re.findall(pattern, docs_section, re.DOTALL)

def generate_doc_id(title: str, text: str) -> str:
    """Generate unique document ID from title and text prefix.
    Uses MD5 hash of title + first 100 chars for deduplication.
    """
    key = f"{title}:{text[:100]}"
    return hashlib.md5(key.encode()).hexdigest()[:12]
```

**Output Schema:**
```json
{
    "chunk_id": "abc123def456_c0000",
    "doc_id": "abc123def456",
    "chunk_index": 0,
    "total_chunks": 1,
    "text": "Document content...",
    "doc_type": "wikipedia",
    "doc_title": "Document Title"
}
```

### 2. Vector Ingestion (`rag/preprocess/ingestion/vector_ingest.py`)

FAISS CPU vector ingestion with sentence transformers.

**Class: `VectorIngester`**

```python
class VectorIngester:
    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",        # Encoder model
        output_dir: str = "data/vector/rag",     # Output directory
        batch_size: int = 32,                    # Encoding batch size
        use_gpu: bool = False,                   # CPU-only by default
        index_type: str = "flat",                # flat, ivf, or hnsw
        normalize_embeddings: bool = True,       # L2 normalize
    ):
        ...
```

**Supported Index Types:**
| Type | Best For | Search Type |
|------|----------|-------------|
| `flat` | < 100K docs | Exact (IndexFlatIP) |
| `ivf` | 100K - 10M docs | Approximate (IVF) |
| `hnsw` | > 1M docs | Approximate (HNSW) |

**Key Methods:**
```python
def encode_texts(self, texts: List[str]) -> np.ndarray:
    """Encode texts to embeddings using sentence transformer."""

def ingest(self, input_path: str, limit: Optional[int] = None) -> Dict:
    """Full ingestion pipeline with checkpointing."""
```

### 3. Hybrid Retriever (`stindex/retrieval/hybrid_retriever.py`)

Combines FAISS vector search with optional dimensional filtering.

**Class: `HybridRetriever`**

```python
class HybridRetriever:
    def __init__(
        self,
        vector_index_path: str,              # Path to FAISS index
        stindex_warehouse_path: str = None,  # Optional STIndex warehouse
        encoder_model: str = None,           # Override encoder model
        device: str = "cpu",                 # Encoding device
    ):
        ...
```

**Search Method:**
```python
def search(
    self,
    query: str,
    k: int = 10,
    temporal_filter: Optional[TemporalFilter] = None,  # Time filtering
    spatial_filter: Optional[SpatialFilter] = None,    # Location filtering
    filter_mode: str = "post",                         # pre or post filter
    expand_factor: int = 3,                            # Pre-filter expansion
) -> RetrievalResponse:
    """
    Search for relevant chunks.

    Filter modes:
    - 'post': Retrieve k*expand_factor, then filter (better recall)
    - 'pre': Filter first, then retrieve from filtered set (faster)
    """
```

**Response Schema:**
```python
@dataclass
class RetrievalResponse:
    query: str
    results: List[RetrievalResult]
    total_candidates: int
    filtered_candidates: Optional[int]
    filter_stats: Dict[str, Any]

@dataclass
class RetrievalResult:
    chunk_id: str
    text: str
    score: float
    metadata: Dict[str, Any]
```

### 4. RAG Retriever (`rag/retriever/retriever.py`)

High-level retriever with multi-hop support for complex questions.

**Class: `RAGRetriever`**

```python
@dataclass
class RAGRetrieverConfig:
    vector_index_path: str = "data/vector/rag"
    warehouse_path: Optional[str] = "data/warehouse/rag"
    default_k: int = 5
    max_context_length: int = 4000      # Max chars for LLM context
    max_hops: int = 2                   # Multi-hop iterations
    documents_per_hop: int = 3          # Docs per hop
    enable_dimensional_filter: bool = True
    filter_mode: str = "post"           # pre or post
```

**Multi-hop Retrieval:**
```python
def retrieve_multihop(
    self,
    question: str,
    max_hops: int = 2,
    documents_per_hop: int = 3,
) -> RetrievalContext:
    """
    Multi-hop retrieval for complex questions.

    Algorithm:
    1. Initial retrieval with question
    2. Extract document titles from results
    3. Use titles as follow-up queries
    4. Deduplicate and combine results
    """
```

**Context Formatting:**
```python
def get_formatted_context(
    self,
    include_titles: bool = True,
    include_scores: bool = False,
    max_length: Optional[int] = None,
) -> str:
    """
    Format retrieved chunks for LLM input.

    Output format:
    [Document 1] Title
    Content...

    [Document 2] Title
    Content...
    """
```

### 5. Answer Generator (`rag/generator/generator.py`)

LLM-based answer generation using retrieved context.

**Class: `RAGGenerator`**

```python
@dataclass
class GeneratorConfig:
    provider: str = "openai"           # openai or anthropic
    model: str = "gpt-4o-mini"         # Model name
    temperature: float = 0.3           # Generation temperature
    max_tokens: int = 500              # Max output tokens
    system_prompt: Optional[str] = None
```

**Supported Providers:**
| Provider | Models | API |
|----------|--------|-----|
| OpenAI | gpt-4o, gpt-4o-mini, gpt-4-turbo | OpenAI API |
| Anthropic | claude-3.5-sonnet, claude-3-opus | Anthropic API |

**Generation Method:**
```python
def generate(
    self,
    question: str,
    context: RetrievalContext,
    custom_prompt: Optional[str] = None,
) -> GenerationResult:
    """
    Generate answer using retrieved context.

    Returns:
        GenerationResult with answer, reasoning, sources
    """
```

**Response Schema:**
```python
@dataclass
class GenerationResult:
    answer: str                    # Final answer
    reasoning: Optional[str]       # Chain of thought
    sources: List[str]             # Source chunk IDs
    raw_response: str              # Full LLM response
    usage: Dict[str, int]          # Token usage
```

### 6. RAG Pipeline (`rag/pipeline.py`)

End-to-end orchestration of retrieval and generation.

**Class: `RAGPipeline`**

```python
class RAGPipeline:
    def __init__(
        self,
        retriever_config: Optional[RAGRetrieverConfig] = None,
        generator_config: Optional[GeneratorConfig] = None,
        vector_index_path: str = "data/vector/rag",
    ):
        self.retriever = RAGRetriever(config=retriever_config)
        self.generator = RAGGenerator(config=generator_config)
```

**Main Method:**
```python
def answer(
    self,
    question: str,
    k: int = 5,
    use_multihop: bool = False,
) -> Dict[str, Any]:
    """
    End-to-end question answering.

    Returns:
        {
            "question": str,
            "answer": str,
            "reasoning": str,
            "sources": List[str],
            "retrieved_chunks": List[Dict],
            "usage": Dict[str, int]
        }
    """
```

---

## Optional: STIndex Dimensional Extraction

For temporal/spatial filtering, run STIndex extraction on the corpus:

```bash
python -m rag.preprocess.ingestion.stindex_ingest \
    --input data/corpus/grpo/chunks.jsonl \
    --output data/warehouse/rag \
    --config extract \
    --parallel \
    --workers 100
```

This enables dimensional filters in retrieval:
```python
# Temporal filtering
retriever.retrieve(
    query="What happened in March 2022?",
    temporal_filter=TemporalFilter(year=2022, month=3)
)

# Spatial filtering
retriever.retrieve(
    query="Events in Australia",
    spatial_filter=SpatialFilter(region="Australia")
)
```

---

## File Structure

```
rag/
├── __init__.py                          # Package exports
├── pipeline.py                          # RAGPipeline orchestration
├── README.md                            # This file
│
├── preprocess/
│   ├── extract_grpo_docs.py            # Extract docs from GRPO
│   ├── dataset/
│   │   ├── dataset_generator.py        # GRPO data generation
│   │   └── dataset_generator_sft.py    # SFT data generation
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── vector_ingest.py            # FAISS vector ingestion
│   │   └── stindex_ingest.py           # STIndex extraction (optional)
│   └── pikerag/                         # Dataset downloading
│
├── retriever/
│   ├── __init__.py
│   ├── retriever.py                     # RAGRetriever
│   └── query_processor.py               # Query processing utilities
│
└── generator/
    ├── __init__.py
    ├── generator.py                     # RAGGenerator
    └── prompts.py                       # Prompt templates

stindex/retrieval/
├── __init__.py                          # Exports HybridRetriever, filters
├── hybrid_retriever.py                  # HybridRetriever implementation
└── dimensional_filter.py                # TemporalFilter, SpatialFilter

scripts/
└── process_data.sh                      # Full pipeline script
```

---

## Usage Examples

### Basic RAG Query
```python
from rag import RAGPipeline

pipeline = RAGPipeline(
    vector_index_path="data/vector/rag"
)

result = pipeline.answer(
    question="What nationality is the director of Hello Sister?",
    k=5
)
print(f"Answer: {result['answer']}")
```

### Custom Configuration
```python
from rag.retriever import RAGRetriever, RAGRetrieverConfig
from rag.generator import RAGGenerator, GeneratorConfig

retriever = RAGRetriever(config=RAGRetrieverConfig(
    vector_index_path="data/vector/rag",
    default_k=10,
    max_context_length=6000,
))

generator = RAGGenerator(config=GeneratorConfig(
    provider="openai",
    model="gpt-4o",
    temperature=0.2,
))

context = retriever.retrieve("Your question here")
result = generator.generate("Your question here", context)
```

### Multi-hop Retrieval
```python
from rag.retriever import RAGRetriever

retriever = RAGRetriever(vector_index_path="data/vector/rag")

# Complex question requiring multiple documents
context = retriever.retrieve_multihop(
    question="What is the birthplace of the director of the film that won Best Picture in 1994?",
    max_hops=3,
    documents_per_hop=3,
)

print(f"Retrieved {len(context.chunks)} chunks across {context.hops} hops")
```

---

## Performance Characteristics

| Metric | Value |
|--------|-------|
| Corpus Size | 916,618 documents |
| Index Type | FAISS Flat (exact search) |
| Embedding Model | all-MiniLM-L6-v2 |
| Embedding Dimension | 384 |
| Encoding Speed | ~200 docs/sec (CPU) |
| Search Latency | ~50ms (k=10) |
| Memory Usage | ~1.5GB (916K vectors) |

---

## Running the Full Pipeline

```bash
# Run all stages
./scripts/process_data.sh

# Skip stages if data exists
./scripts/process_data.sh --skip-download --skip-training

# With vector ingestion
./scripts/process_data.sh --with-vector

# Limit vector ingestion for testing
./scripts/process_data.sh --with-vector --vector-limit 10000
```

---

## Environment Variables

```bash
# Required for answer generation
export OPENAI_API_KEY="your-key"      # For OpenAI
export ANTHROPIC_API_KEY="your-key"   # For Anthropic

# Optional
export TOKENIZERS_PARALLELISM=false   # Suppress tokenizer warnings
```

---

## Dependencies

```
faiss-cpu>=1.7.0
sentence-transformers>=2.2.0
jsonlines>=3.1.0
openai>=1.0.0
anthropic>=0.18.0
loguru>=0.7.0
tqdm>=4.65.0
numpy>=1.24.0
```

have qa pairs + corpus

 - question extract entity types (llm, ner) -> schema
 - qa with domain -> ontology -> entity types
 - schema discovery & construction


schema discovery

key entity types in an article

auto define multi dim

pre generate a dimensional schema

progressively extract dimensions from unstructured text

question topic clustering

corpus ner based on 


dimensions

粗颗粒度 -> 细颗粒度

sparse dense retrieval



corpus




tableqa

text2sql

question generation based on original rows

dataset generated

question NER agent

