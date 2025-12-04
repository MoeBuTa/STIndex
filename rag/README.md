# STIndex RAG Module

RAG (Retrieval-Augmented Generation) pipeline with multi-hop QA and medical datasets.

## Quick Start

```bash
# Full pipeline: download + extract + vector index
./scripts/process_data.sh --with-vector

# Skip download if data exists
./scripts/process_data.sh --skip-download --with-vector

# Generate training data (GRPO, SFT)
./scripts/process_data.sh --with-training

# Process specific datasets only
./scripts/process_data.sh --datasets "hotpotqa two_wiki"

# Test with limited documents
./scripts/process_data.sh --with-vector --vector-limit 1000
```

## Datasets

| Dataset | Type | Questions | Documents | Purpose |
|---------|------|-----------|-----------|---------|
| HotpotQA | Multi-hop QA | 90,425 | ~60K | Training & Eval |
| 2WikiMQA | Multi-hop QA | 165,464 | ~100K | Training & Eval |
| MuSiQue | Multi-hop QA | 19,938 | ~20K | Training & Eval |
| MedCorp | Medical Corpus | 0 | 125,847 | Retrieval only |
| MIRAGE | Medical Benchmark | 6,545* | 0 | Evaluation only |

\* MIRAGE filtered to clinical exams (MedQA, MedMCQA, MMLU) by default. See [MIRAGE Configuration](#mirage-configuration).

## Data Pipeline

```
Stage 1: Download               Stage 2: Extract              Stage 3: Vector (optional)
──────────────────────          ─────────────────────         ──────────────────────────
data/original/                  data/corpus/                  data/vector/rag/
├── hotpotqa/train.jsonl        ├── documents.jsonl           ├── faiss_index.bin
├── two_wiki/train.jsonl        ├── {dataset}/train/          ├── id_mapping.json
├── musique/train.jsonl         │   └── documents.jsonl       ├── index_config.json
├── mirage/train.jsonl          └── stats.json                └── documents_metadata.jsonl
└── medcorp/train.jsonl         data/questions/
                                ├── questions.jsonl
                                ├── {dataset}/train/
                                │   └── questions.jsonl
                                └── stats.json
```

**Corpus Stats:**
- Documents: ~1.3M (deduplicated by content hash)
- Questions: ~282K (multi-hop + medical benchmark)

## Data Schemas

**Document** (`data/corpus/documents.jsonl`):
```json
{"doc_id": "abc123", "title": "Article Title", "contents": "..."}
```

**Question** (`data/questions/questions.jsonl`):
```json
{"question_id": "xyz", "question": "Who is the director of Hello Sister?"}
```

## Vector Ingestion

Creates FAISS index for dense retrieval:

```bash
python -m rag.preprocess.train.ingestion.vector_ingest \
    --input data/corpus/documents.jsonl \
    --output data/vector/rag \
    --model BAAI/bge-m3 \
    --index-type ivf
```

| Index Type | Best For | Description |
|------------|----------|-------------|
| `flat` | <100K docs | Exact search, slower |
| `ivf` | 100K-10M docs | Approximate with clustering |
| `hnsw` | >1M docs | Graph-based, balanced |

## Module Structure

```
rag/
├── cfg/ingestion.yaml              # Vector ingestion config
├── preprocess/
│   ├── corpus/                     # Document/question extraction
│   │   └── extract_documents.py
│   └── train/
│       ├── pikerag/                # Dataset download (from PikeRAG)
│       ├── ingestion/              # Vector & STIndex ingestion
│       └── dataset/                # Training data generation
├── retriever/                      # RAG retriever
└── generator/                      # Answer generation
```

## Usage

```python
# Vector search
from rag.preprocess.train.ingestion.vector_ingest import VectorIndexLoader

loader = VectorIndexLoader("data/vector/rag")
results = loader.search("When was Arthur's Magazine founded?", k=5)

for r in results:
    print(f"{r['chunk_id']}: {r['score']:.3f}")
```

## Command Reference

### Basic Commands

```bash
# Stage 1 only: Download and format datasets
./scripts/process_data.sh

# Stage 1 + 2: Download + extract corpus/questions
./scripts/process_data.sh

# Stage 1 + 2 + 4: Full pipeline with vector index
./scripts/process_data.sh --with-vector

# Stage 1 + 2 + 3 + 4: All stages including training data
./scripts/process_data.sh --with-training --with-vector
```

### Selective Processing

```bash
# Skip Stage 1 if data already downloaded
./scripts/process_data.sh --skip-download

# Skip Stage 2 if corpus already extracted
./scripts/process_data.sh --skip-download --skip-corpus --with-vector

# Process only specific datasets
./scripts/process_data.sh --datasets "hotpotqa mirage"
./scripts/process_data.sh --datasets "medcorp"

# Test with limited documents (vector stage)
./scripts/process_data.sh --with-vector --vector-limit 1000
```

### CLI Options

| Option | Description |
|--------|-------------|
| `--skip-download` | Skip Stage 1 (download and format) |
| `--skip-corpus` | Skip Stage 2 (corpus extraction) |
| `--with-training` | Enable Stage 3 (GRPO/SFT training data) |
| `--with-vector` | Enable Stage 4 (FAISS vector index) |
| `--vector-limit N` | Limit vector ingestion to N documents |
| `--datasets "..."` | Process only specified datasets |
| `--help` | Show help message |

## MIRAGE Configuration

MIRAGE benchmark has two modes:

1. **Clinical Exams Only** (default): 6,545 questions (MedQA, MedMCQA, MMLU)
   - Answerable by MedCorp corpus (Textbooks + StatPearls)

2. **All Questions**: 7,663 questions (adds PubMedQA, BioASQ)
   - Research QA requires PubMed corpus (not included)

**To enable all questions:**

Edit `rag/preprocess/train/pikerag/data_process/config/datasets.yaml`:
```yaml
mirage_config:
  filter_clinical_only: false  # Default: true
```

Then re-run Stage 1:
```bash
./scripts/process_data.sh --datasets "mirage"
```


       187 -  
       188 -  have qa pairs + corpus
       189 -  
       190 -   - question extract entity types (llm, ner) -> schema
       191 -   - qa with domain -> ontology -> entity types
       192 -   - schema discovery & construction
       193 -  
       194 -  
       195 -  schema discovery
       196 -  
       197 -  key entity types in an article
       198 -  
       199 -  auto define multi dim
       200 -  
       201 -  pre generate a dimensional schema
       202 -  
       203 -  progressively extract dimensions from unstructured text
       204 -  
       205 -  question topic clustering
       206 -  
       207 -  corpus ner based on 
       208 -  
       209 -  
       210 -  dimensions
       211 -  
       212 -  粗颗粒度 -> 细颗粒度
       213 -  
       214 -  sparse dense retrieval
       215 -  
       216 -  
       217 -  
       218 -  corpus
       219 -  
       220 -  
       221 -  
       222 -  
       223 -  tableqa
       224 -  
       225 -  text2sql
       226 -  
       227 -  question generation based on original rows
       228 -  
       229 -  dataset generated
       230 -  
       231 -  question NER agent
       232 -  
       233 -  ```mermaid
       234 -  graph TD
       235 -      subgraph "Phase 1: Demand Discovery"
       236 -      A[QA Dataset] --> B(Question Embedding)
       237 -      B --> C{Clustering Algorithm}
       238 -      C -->|Cluster 1| D[Intent Group A]
       239 -      C -->|Cluster 2| E[Intent Group B]
       240 -      D & E --> F[Entity & Keyword Extraction]
       241 -      end
       242 -  
       243 -      subgraph "Phase 2: Schema Construction"
       244 -      F --> G[LLM Dimension Proposer]
       245 -      G --> H[Candidate Schema]
       246 -      H --> I[Refinement Agent]
       247 -      I -->|Critique & Optimize| H
       248 -      I --> J[Final Dimensional Schema]
       249 -      end
       250 -  
       251 -      subgraph "Phase 3: Indexing & RAG"
       252 -      J --> K[Metadata Extractor Prompt]
       253 -      L[Original Document Corpus] --> K
       254 -      K --> M[Structured Chunks]
       255 -      M --> N[(Vector Store with Metadata)]
       256 -      
       257 -      UserQuery --> O[Query Analyzer]
       258 -      J --> O
       259 -      O -->|Filter: Region=US, Time=2024| N
       260 -      N --> P[Targeted Retrieval]
       261 -      end