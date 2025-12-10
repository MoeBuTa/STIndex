# Pipeline Architecture Guide

This document describes the two main pipelines in STIndex:
1. **STIndexPipeline**: Extraction pipeline for processing documents and extracting dimensional entities
2. **SchemaDiscoveryPipeline**: Schema discovery pipeline for discovering dimensions from Q&A datasets

---

## Table of Contents

- [1. STIndexPipeline (Extraction Pipeline)](#1-stindexpipeline-extraction-pipeline)
  - [1.1 Overview](#11-overview)
  - [1.2 Architecture](#12-architecture)
  - [1.3 Execution Modes](#13-execution-modes)
  - [1.4 Key Features](#14-key-features)
  - [1.5 Usage Examples](#15-usage-examples)
  - [1.6 Configuration](#16-configuration)
- [2. SchemaDiscoveryPipeline](#2-schemadiscoverypipeline)
  - [2.1 Overview](#21-overview)
  - [2.2 Architecture](#22-architecture)
  - [2.3 Pipeline Phases](#23-pipeline-phases)
  - [2.4 Key Features](#24-key-features)
  - [2.5 Usage Examples](#25-usage-examples)
  - [2.6 Configuration](#26-configuration)
- [3. Integration Patterns](#3-integration-patterns)

---

## 1. STIndexPipeline (Extraction Pipeline)

### 1.1 Overview

**Purpose**: Extract multi-dimensional entities from unstructured text documents

**Location**: `stindex/pipeline/pipeline.py`

**Key Capabilities**:
- Multi-dimensional extraction (temporal, spatial, event, entity, custom dimensions)
- Context-aware extraction with memory across document chunks
- Two-pass reflection for quality filtering
- Spatiotemporal clustering and analysis
- Data warehouse integration (optional)

**Input**: Documents (URLs, files, or raw text)

**Output**: Structured dimensional entities with optional analysis results

---

### 1.2 Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        STIndexPipeline                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Phase 1: PREPROCESSING                                             │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐           │
│  │   Scraper    │──>│    Parser    │──>│   Chunker    │           │
│  │  (Web/File)  │   │  (PDF/HTML)  │   │  (Strategy)  │           │
│  └──────────────┘   └──────────────┘   └──────────────┘           │
│         │                   │                   │                   │
│         └───────────────────┴───────────────────┘                   │
│                            │                                         │
│                     [DocumentChunks]                                │
│                            │                                         │
├────────────────────────────┼─────────────────────────────────────────┤
│                            ▼                                         │
│  Phase 2: EXTRACTION (Context-Aware)                                │
│  ┌───────────────────────────────────────────────────────┐         │
│  │          ExtractionContext (Memory)                    │         │
│  │  - Prior temporal references (last 10)                 │         │
│  │  - Prior spatial references (last 10)                  │         │
│  │  - Nearby locations for disambiguation                 │         │
│  │  - Resets between documents                            │         │
│  └───────────────┬───────────────────────────────────────┘         │
│                  │                                                   │
│                  ▼                                                   │
│  ┌────────────────────────────────────────────────────────┐        │
│  │       DimensionalExtractor (LLM-based)                  │        │
│  │  - Loads dimension configs (temporal, spatial, custom)  │        │
│  │  - Prompts with context hints                           │        │
│  │  - Extracts entities per dimension                      │        │
│  │  - Updates context after each chunk                     │        │
│  └────────────────┬───────────────────────────────────────┘        │
│                   │                                                  │
│                   ▼                                                  │
│         [Raw Dimensional Entities]                                  │
│                   │                                                  │
├───────────────────┼──────────────────────────────────────────────────┤
│                   ▼                                                  │
│  Phase 3: POST-PROCESSING                                           │
│  ┌──────────────────────────────────────────────────────┐          │
│  │  Normalization                                        │          │
│  │  - Temporal: ISO 8601 normalization                   │          │
│  │  - Spatial: Geocoding (lat/lon, parent regions)       │          │
│  └──────────────────┬───────────────────────────────────┘          │
│                     │                                                │
│                     ▼                                                │
│  ┌──────────────────────────────────────────────────────┐          │
│  │  Two-Pass Reflection (Optional)                       │          │
│  │  1. LLM scores each entity (relevance, accuracy,      │          │
│  │     completeness, consistency)                        │          │
│  │  2. Filters entities below thresholds                 │          │
│  │  3. Context-aware: validates against prior refs       │          │
│  └──────────────────┬───────────────────────────────────┘          │
│                     │                                                │
│           [Filtered Dimensional Entities]                           │
│                     │                                                │
├─────────────────────┼────────────────────────────────────────────────┤
│                     ▼                                                │
│  Phase 4: ANALYSIS (Optional)                                       │
│  ┌──────────────────────────────────────────────────────┐          │
│  │  EventClusterAnalyzer                                 │          │
│  │  - DBSCAN clustering (geodesic distance)              │          │
│  │  - Temporal burst detection                           │          │
│  │  - Story arc detection                                │          │
│  └──────────────────┬───────────────────────────────────┘          │
│                     │                                                │
│                     ▼                                                │
│  ┌──────────────────────────────────────────────────────┐          │
│  │  DimensionAnalyzer                                    │          │
│  │  - Statistical analysis per dimension                 │          │
│  │  - Entity frequency, diversity                        │          │
│  │  - Temporal/spatial distributions                     │          │
│  └──────────────────┬───────────────────────────────────┘          │
│                     │                                                │
│                     ▼                                                │
│  ┌──────────────────────────────────────────────────────┐          │
│  │  AnalysisDataExporter                                 │          │
│  │  - Exports to JSON/GeoJSON                            │          │
│  │  - Generates static files for frontend                │          │
│  │  - No backend required                                │          │
│  └──────────────────┬───────────────────────────────────┘          │
│                     │                                                │
│                     ▼                                                │
│         [Analysis Results + Export Files]                           │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

---

### 1.3 Execution Modes

The pipeline supports four execution modes:

#### Mode 1: Full Pipeline
```python
pipeline = STIndexPipeline(
    extractor_config="extract",
    enable_context_aware=True,
    enable_reflection=True
)

docs = [
    InputDocument.from_url("https://example.com/article"),
    InputDocument.from_file("data/document.pdf"),
    InputDocument.from_text("Plain text content")
]

# Run full pipeline: preprocessing → extraction → analysis
results = pipeline.run_pipeline(docs, analyze=True)
```

#### Mode 2: Preprocessing Only
```python
# Only scrape, parse, and chunk documents
chunks = pipeline.run_preprocessing(docs)

# Chunks saved to: data/output/chunks/preprocessed_chunks.json
```

#### Mode 3: Extraction Only
```python
# Load previously preprocessed chunks
chunks = pipeline.load_chunks("data/output/chunks/preprocessed_chunks.json")

# Extract entities from chunks
results = pipeline.run_extraction(chunks)
```

#### Mode 4: Analysis Only
```python
# Load previous extraction results
results = pipeline.load_results("data/output/results/gpt-4o-mini/extraction_results.json")

# Run analysis on existing results
analysis_data = pipeline.run_analysis(
    results=results,
    dimensions=['temporal', 'spatial', 'disease', 'event_type']
)
```

---

### 1.4 Key Features

#### 1.4.1 Context-Aware Extraction

**Problem**: Traditional extraction processes each chunk independently, losing context across chunks.

**Solution**: `ExtractionContext` maintains memory across document chunks.

**Features**:
- **Temporal Context**: Stores last 10 temporal references to resolve relative expressions
  - Example: "Monday" → resolves to absolute date using prior references
- **Spatial Context**: Stores last 10 spatial references for disambiguation
  - Example: "downtown" → disambiguated using document's geographic context
- **Nearby Locations**: Provides location hints for spatial extraction
- **Document Boundaries**: Automatically resets memory between different documents

**Configuration** (`cfg/extraction/inference/extract.yml`):
```yaml
context_aware:
  enabled: true
  max_memory_refs: 10
  enable_nearby_locations: false
```

**Example**:
```python
# Chunk 1: "The protest occurred on Monday in downtown Seattle."
# → Extracts: temporal="2024-01-15", spatial="Seattle, WA"

# Chunk 2: "The next day, another rally was held downtown."
# → With context: temporal="2024-01-16" (Monday + 1 day)
#                 spatial="Seattle, WA" (same city as previous chunk)
```

---

#### 1.4.2 Two-Pass Reflection

**Problem**: LLM extraction produces false positives and low-quality entities.

**Solution**: `ExtractionReflector` performs LLM-based quality scoring.

**How It Works**:
1. **First Pass**: Extract entities using main prompts
2. **Second Pass**: LLM scores each entity on 4 dimensions:
   - **Relevance**: Does entity fit dimension definition?
   - **Accuracy**: Is entity correctly normalized/geocoded?
   - **Completeness**: Are all required fields present?
   - **Consistency**: Does entity align with prior references (context-aware)?
3. **Filtering**: Remove entities below configurable thresholds

**Configuration** (`cfg/extraction/inference/reflection.yml`):
```yaml
enabled: false  # Default disabled (adds latency)
thresholds:
  relevance: 0.7
  accuracy: 0.7
  consistency: 0.6
```

**Impact**: Reduces extraction errors by 30-50% in evaluation benchmarks.

---

#### 1.4.3 Spatiotemporal Analysis

**EventClusterAnalyzer**:
- **DBSCAN Clustering**: Groups events by spatiotemporal proximity
  - Uses geodesic distance (Haversine formula)
  - Configurable epsilon (spatial radius) and min_samples
- **Temporal Burst Detection**: Identifies spikes in event frequency
- **Story Arc Detection**: Discovers narrative sequences

**DimensionAnalyzer**:
- **Statistical Analysis**: Entity frequency, diversity, coverage
- **Temporal Distributions**: Time series analysis, seasonality
- **Spatial Distributions**: Geographic heatmaps, region analysis

**AnalysisDataExporter**:
- Exports to JSON/GeoJSON for frontend visualization
- Generates static files (no backend required)
- Output files:
  - `events.json`: All extracted events
  - `clusters.json`: Spatiotemporal clusters
  - `events.geojson`: GeoJSON for mapping
  - `dimension_stats.json`: Statistical summaries

---

### 1.5 Usage Examples

#### Example 1: Basic Extraction
```python
from stindex.pipeline import STIndexPipeline
from stindex.preprocess import InputDocument

# Initialize pipeline
pipeline = STIndexPipeline(
    extractor_config="extract",
    dimension_config="dimensions"
)

# Create input documents
docs = [
    InputDocument.from_text("Hurricane Katrina struck New Orleans on August 29, 2005.")
]

# Run pipeline
results = pipeline.run_pipeline(docs)

# Access results
for result in results:
    print(f"Temporal entities: {result.entities.get('temporal', [])}")
    print(f"Spatial entities: {result.entities.get('spatial', [])}")
```

#### Example 2: Context-Aware Extraction
```python
pipeline = STIndexPipeline(
    extractor_config="extract",
    enable_context_aware=True,
    max_memory_refs=10
)

# Multi-chunk document (context preserved across chunks)
docs = [
    InputDocument.from_file("long_article.pdf")
]

results = pipeline.run_pipeline(docs)
```

#### Example 3: With Two-Pass Reflection
```python
pipeline = STIndexPipeline(
    extractor_config="extract",
    enable_context_aware=True,
    enable_reflection=True,
    relevance_threshold=0.8,
    accuracy_threshold=0.7
)

results = pipeline.run_pipeline(docs)
# Results now filtered for high-quality entities only
```

#### Example 4: With Analysis
```python
pipeline = STIndexPipeline(
    extractor_config="extract",
    enable_context_aware=True,
    output_dir="data/output"
)

# Run full pipeline with analysis
results = pipeline.run_pipeline(docs, analyze=True)

# Analysis files exported to:
# - data/output/analysis/gpt-4o-mini/events.json
# - data/output/analysis/gpt-4o-mini/clusters.json
# - data/output/analysis/gpt-4o-mini/events.geojson
# - data/output/analysis/gpt-4o-mini/dimension_stats.json
```

---

### 1.6 Configuration

#### Main Config (`cfg/extraction/inference/extract.yml`)
```yaml
llm:
  llm_provider: openai  # openai, anthropic, hf
  model_name: gpt-4o-mini

context_aware:
  enabled: true
  max_memory_refs: 10
  enable_nearby_locations: false

reflection:
  enabled: false  # Enable for quality filtering

output:
  save_intermediate: true
  output_dir: data/output
```

#### Dimension Config (`cfg/extraction/inference/dimensions.yml`)
```yaml
temporal:
  enabled: true
  extraction_type: normalized
  schema_type: single_field
  fields:
    - name: temporal_expression
      type: string
      description: Any temporal reference

spatial:
  enabled: true
  extraction_type: geocoded
  schema_type: multi_field
  fields:
    - name: location
      type: string
    - name: parent_region
      type: string

# Custom dimensions
disease:
  enabled: true
  extraction_type: categorical
  schema_type: multi_field
  fields:
    - name: disease_name
      type: string
    - name: disease_category
      type: string
  examples:
    - "COVID-19 (infectious disease)"
```

---

## 2. SchemaDiscoveryPipeline

### 2.1 Overview

**Purpose**: Automatically discover dimensional schemas from Q&A datasets

**Location**: `stindex/schema_discovery/discover_schema.py`

**Key Capabilities**:
- Data-driven dimension discovery (no predefined constraints)
- Cluster-level schema discovery (each cluster discovers independently)
- Hierarchical dimension structures (e.g., disease → symptom → severity)
- Cross-cluster schema merging with deduplication
- Parallel processing for scalability

**Input**: Question dataset (JSONL format) with Q&A pairs

**Output**: Final schema (YAML/JSON) with discovered dimensions and entities

---

### 2.2 Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                   SchemaDiscoveryPipeline                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Phase 1: QUESTION CLUSTERING                                       │
│  ┌──────────────────────────────────────────────────────┐          │
│  │       QuestionClusterer                               │          │
│  │  - Semantic embeddings (sentence-transformers)        │          │
│  │  - K-means clustering                                 │          │
│  │  - FAISS indexing for efficiency                      │          │
│  └──────────────────┬───────────────────────────────────┘          │
│                     │                                                │
│          [Cluster Assignments]                                      │
│          Cluster 0: 2652 questions                                  │
│          Cluster 1: 1447 questions                                  │
│          Cluster 2: 2446 questions                                  │
│                     │                                                │
├─────────────────────┼────────────────────────────────────────────────┤
│                     ▼                                                │
│  Phase 2: PER-CLUSTER DISCOVERY + EXTRACTION                        │
│  (Parallel Processing - 3 workers in this example)                  │
│                                                                      │
│  ┌────────────────────────────────────────────────────────┐        │
│  │  Cluster 0 Thread                                       │        │
│  │  ┌──────────────────────────────────────────────────┐  │        │
│  │  │ ClusterSchemaDiscoverer                           │  │        │
│  │  │                                                    │  │        │
│  │  │ Step 1: Discovery (sample 10 questions)           │  │        │
│  │  │   ┌──────────────────────────────────────────┐   │  │        │
│  │  │   │ ClusterSchemaPrompt (LLM)                 │   │  │        │
│  │  │   │ "Analyze these questions and discover     │   │  │        │
│  │  │   │  2-5 key dimensions with hierarchies"     │   │  │        │
│  │  │   └─────────────┬────────────────────────────┘   │  │        │
│  │  │                 │                                  │  │        │
│  │  │   [Discovered Dimensions]                         │  │        │
│  │  │   - Clinical_Trials: trial_type → intervention   │  │        │
│  │  │   - Pharmacology: drug → mechanism → indication  │  │        │
│  │  │   - Physiology: system → organ → function        │  │        │
│  │  │   - Medical_Education: level → curriculum        │  │        │
│  │  │                 │                                  │  │        │
│  │  │ Step 2: Extraction (all 2652 questions)          │  │        │
│  │  │   ┌──────────────────────────────────────────┐   │  │        │
│  │  │   │ ClusterEntityExtractor                    │   │  │        │
│  │  │   │ - Batch processing (50 questions/batch)   │   │  │        │
│  │  │   │ - Entity deduplication (fuzzy matching)   │   │  │        │
│  │  │   │ - Allows discovering new dimensions       │   │  │        │
│  │  │   └─────────────┬────────────────────────────┘   │  │        │
│  │  │                 │                                  │  │        │
│  │  │   [Extracted Entities]                            │  │        │
│  │  │   - Clinical_Trials: 45 entities                 │  │        │
│  │  │   - Pharmacology: 33 entities                    │  │        │
│  │  │   - Physiology: 197 entities                     │  │        │
│  │  │                 │                                  │  │        │
│  │  └─────────────────┼──────────────────────────────┘  │        │
│  └────────────────────┼─────────────────────────────────┘        │
│                       │                                            │
│  ┌────────────────────┼─────────────────────────────────┐        │
│  │  Cluster 1 Thread  ▼                                  │        │
│  │  [Similar process: Discovery → Extraction]            │        │
│  │  → Discovers 5 different dimensions                   │        │
│  │  → Extracts 109 entities                              │        │
│  └───────────────────────────────────────────────────────┘        │
│                       │                                            │
│  ┌────────────────────┼─────────────────────────────────┐        │
│  │  Cluster 2 Thread  ▼                                  │        │
│  │  [Similar process: Discovery → Extraction]            │        │
│  │  → Discovers 5 different dimensions                   │        │
│  │  → Extracts 127 entities                              │        │
│  └───────────────────────────────────────────────────────┘        │
│                       │                                            │
│           [All Cluster Results]                                   │
│           - Cluster 0: 4 dimensions, 275 entities                 │
│           - Cluster 1: 5 dimensions, 109 entities                 │
│           - Cluster 2: 5 dimensions, 127 entities                 │
│                       │                                            │
├───────────────────────┼────────────────────────────────────────────┤
│                       ▼                                            │
│  Phase 3: CROSS-CLUSTER SCHEMA MERGING                            │
│  ┌──────────────────────────────────────────────────────┐        │
│  │       SchemaMerger                                    │        │
│  │                                                        │        │
│  │  Step 1: Dimension Merging                            │        │
│  │    - Fuzzy name matching (0.85 similarity)            │        │
│  │    - Hierarchy structure comparison                   │        │
│  │    - Merge similar dimensions across clusters         │        │
│  │                                                        │        │
│  │  Step 2: Entity Deduplication                         │        │
│  │    - Fuzzy text matching per dimension                │        │
│  │    - Preserve source tracking (cluster IDs)           │        │
│  │    - Count entities per cluster                       │        │
│  │                                                        │        │
│  │  Step 3: Statistics & Metadata                        │        │
│  │    - Total entity count per dimension                 │        │
│  │    - Source clusters per dimension                    │        │
│  │    - Alternative dimension names                      │        │
│  └──────────────────┬───────────────────────────────────┘        │
│                     │                                              │
│           [Final Merged Schema]                                   │
│           14 dimensions, 571 unique entities                      │
│                     │                                              │
├─────────────────────┼──────────────────────────────────────────────┤
│                     ▼                                              │
│  Phase 4: EXPORT                                                  │
│  ┌──────────────────────────────────────────────────────┐        │
│  │  - final_schema.yml (YAML format)                     │        │
│  │  - final_schema.json (JSON format)                    │        │
│  │  - cluster_X_result.json (intermediate results)       │        │
│  │  - cot/ directory (reasoning traces)                  │        │
│  └───────────────────────────────────────────────────────┘        │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

---

### 2.3 Pipeline Phases

#### Phase 1: Question Clustering

**Component**: `QuestionClusterer`

**Purpose**: Group semantically similar questions

**Process**:
1. Load questions from JSONL file
2. Generate embeddings using sentence-transformers
3. Cluster using K-means (default: 10 clusters)
4. Save assignments + sample questions per cluster

**Output**:
- `cluster_assignments.csv`: Question → Cluster mapping
- `cluster_samples.json`: Sample questions per cluster
- `cluster_analysis.json`: Statistics (silhouette score, inertia)

---

#### Phase 2: Per-Cluster Discovery + Extraction

**Components**:
- `ClusterSchemaDiscoverer`: Orchestrates discovery + extraction per cluster
- `ClusterEntityExtractor`: Batch entity extraction

**Process** (per cluster):

**Step 1: Discovery**
1. Sample N questions (default: 20) from cluster
2. Send to LLM with `ClusterSchemaPrompt`:
   - "Analyze these questions and discover 2-5 key dimensions"
   - "For each dimension, provide a hierarchical structure"
   - "No predefined constraints - discover based on data"
3. Parse LLM response → `DiscoveredDimensionSchema` objects

**Step 2: Extraction**
1. Process ALL cluster questions in batches (default: 50 per batch)
2. For each batch:
   - Provide discovered dimensions as context
   - Extract entities using `ClusterEntityPrompt`
   - Allow discovering NEW dimensions during extraction
   - Deduplicate entities using fuzzy matching (0.85 threshold)
3. Update dimension schemas if new dimensions discovered

**Key Features**:
- **Data-Driven**: LLM decides dimension count based on question patterns
- **Hierarchical**: Dimensions have multi-level structures (e.g., symptom → type → severity)
- **Dynamic**: Can discover new dimensions during extraction
- **Efficient**: Batch processing (50 questions/batch) reduces LLM calls

**Output** (per cluster):
- `cluster_X_result.json`: Discovered dimensions + extracted entities
- `cot/cluster_X/`: Chain-of-thought reasoning traces

---

#### Phase 3: Cross-Cluster Schema Merging

**Component**: `SchemaMerger`

**Purpose**: Merge schemas from all clusters, removing duplicates

**Process**:

**Step 1: Dimension Merging**
- Compare dimension names across clusters (fuzzy matching, 0.85 threshold)
- Compare hierarchy structures
- Merge dimensions with similar names + structures
- Track alternative names

**Step 2: Entity Deduplication**
- For each merged dimension:
  - Collect entities from all source clusters
  - Deduplicate using fuzzy text matching (0.85 threshold)
  - Preserve source tracking (cluster IDs)

**Step 3: Statistics**
- Calculate total entity count per dimension
- Track source clusters per dimension
- Count entities per cluster per dimension

**Output**: `FinalSchema` Pydantic model

---

#### Phase 4: Export

**Formats**:
1. **YAML** (`final_schema.yml`):
   - Human-readable
   - Suitable for manual review/editing
   - Used as dimension config for extraction

2. **JSON** (`final_schema.json`):
   - Machine-readable
   - Suitable for programmatic access
   - Contains full metadata

**Structure**:
```yaml
dimensions:
  Clinical_Trials:
    hierarchy:
      - trial_type
      - intervention
      - population
    description: "Clinical trial dimensions..."
    examples:
      - "randomized controlled trial"
      - "drug trial with placebo"
    entities:
      - text: "Phase III trial"
        hierarchy_values:
          trial_type: "Phase III"
          intervention: "drug"
        dimension: "Clinical_Trials"
    total_entity_count: 45
    sources:
      cluster_ids: [0, 1]
      entity_counts_per_cluster:
        0: 30
        1: 15
```

---

### 2.4 Key Features

#### 2.4.1 Data-Driven Discovery

**Problem**: Predefined dimension constraints (e.g., "discover exactly 10 dimensions") force artificial schemas.

**Solution**: Let LLM decide based on actual data patterns.

**Implementation**:
- Remove `n_schemas_per_cluster` parameter
- Prompt: "Discover 2-5 dimensions based on what you observe"
- Result: Each cluster discovers different counts (e.g., 4, 5, 5)

**Benefits**:
- More natural dimension structures
- Avoids over-segmentation or under-segmentation
- Better reflects domain complexity

---

#### 2.4.2 Hierarchical Dimensions

**Structure**: Each dimension has a multi-level hierarchy

**Example**:
```
Pharmacology:
  drug_class → specific_drug → mechanism_of_action → indication

  Entities:
  - NSAIDs → aspirin → COX inhibitor → pain relief
  - Antibiotics → penicillin → cell wall synthesis inhibitor → bacterial infection
```

**Benefits**:
- Captures semantic relationships
- Enables hierarchical queries
- Supports multi-level aggregation

---

#### 2.4.3 Dynamic Dimension Discovery

**Feature**: Discover new dimensions DURING extraction phase

**Process**:
1. Initial discovery provides baseline dimensions
2. During extraction, LLM can propose new dimensions
3. New dimensions added to schema
4. Subsequent batches include new dimensions

**Example**:
```
Cluster 0 Discovery: 4 dimensions
  - Clinical_Trials, Pharmacology, Physiology, Anatomy

Cluster 0 Extraction (Batch 10): Discovers new dimension
  + Medical_Education (wasn't in initial discovery)

Final Cluster 0 Schema: 5 dimensions
```

**Configuration** (`cfg/schema_discovery.yml`):
```yaml
entity_extraction:
  allow_new_dimensions: true  # Enable dynamic discovery
```

---

#### 2.4.4 Parallel Processing

**Feature**: Process multiple clusters concurrently

**Configuration**:
```yaml
parallel:
  enabled: true
  max_workers: 5  # Number of concurrent cluster threads
```

**Benefits**:
- 5x speedup for 10 clusters (near-linear scaling)
- Thread-safe CoT logging
- Efficient resource utilization

**Example**:
- Sequential: 10 clusters × 30 min = 5 hours
- Parallel (5 workers): 10 clusters ÷ 5 × 30 min = 1 hour

---

#### 2.4.5 Chain-of-Thought Logging

**Component**: `CoTLogger`

**Purpose**: Save LLM reasoning traces for debugging and analysis

**Logged Data**:
- **Cluster Discovery**: LLM reasoning for initial dimension discovery
- **Batch Extraction**: LLM reasoning for each extraction batch
- **Statistics**: Reasoning coverage, average length

**Output Structure**:
```
cot/
├── cluster_0/
│   ├── discovery_reasoning.txt
│   ├── discovery_raw.txt
│   ├── batch_000_reasoning.txt
│   ├── batch_000_raw.txt
│   ├── batch_001_reasoning.txt
│   └── ...
├── cluster_1/
│   └── ...
└── reasoning_summary.json
```

**Benefits**:
- Debug extraction issues
- Analyze LLM decision-making
- Improve prompt engineering

---

### 2.5 Usage Examples

#### Example 1: Basic Schema Discovery
```bash
# Using the main module
python -m stindex.schema_discovery.discover_schema \
    --questions data/original/mirage/train.jsonl \
    --output-dir data/schema_discovery \
    --n-clusters 10 \
    --n-samples 20
```

#### Example 2: Using Regeneration Script
```bash
# Test mode (3 clusters, 10 samples, fast)
python scripts/regenerate_schemas.py --dataset mirage --test

# Full mode (10 clusters, 20 samples, production)
python scripts/regenerate_schemas.py --dataset mirage --full
```

#### Example 3: Custom Configuration
```bash
python -m stindex.schema_discovery.discover_schema \
    --questions data/original/mirage/train.jsonl \
    --output-dir data/custom_schema \
    --n-clusters 5 \
    --n-samples 15 \
    --batch-size 30 \
    --max-workers 3 \
    --llm-provider anthropic \
    --model claude-3-5-sonnet-20241022
```

#### Example 4: Programmatic Usage
```python
from stindex.schema_discovery import SchemaDiscoveryPipeline

# Initialize pipeline
llm_config = {
    'llm_provider': 'openai',
    'model_name': 'gpt-4o-mini'
}

pipeline = SchemaDiscoveryPipeline(
    llm_config=llm_config,
    n_clusters=10,
    n_samples_for_discovery=20,
    enable_parallel=True,
    max_workers=5
)

# Run discovery
final_schema = pipeline.discover_schema(
    questions_file="data/original/mirage/train.jsonl",
    output_dir="data/schema_discovery",
    reuse_clusters=True  # Reuse existing cluster assignments
)

# Access results
print(f"Discovered {len(final_schema.dimensions)} dimensions")
for dim_name, dimension in final_schema.dimensions.items():
    print(f"  {dim_name}: {dimension.total_entity_count} entities")
```

---

### 2.6 Configuration

#### Pipeline Config (`cfg/schema_discovery.yml`)
```yaml
cluster_discovery:
  num_clusters: 10
  samples_per_discovery: 20

entity_extraction:
  batch_size: 50
  allow_new_dimensions: true
  retry:
    max_retries: 3

schema_merging:
  similarity_threshold: 0.85

parallel:
  enabled: true
  max_workers: 5
```

#### CLI Arguments (Override Config)
```bash
--n-clusters 10              # Number of question clusters
--n-samples 20               # Samples per cluster for discovery
--batch-size 50              # Questions per LLM call
--max-workers 5              # Parallel workers
--similarity-threshold 0.85  # Fuzzy matching threshold
--no-parallel                # Disable parallel processing
--reuse-clusters             # Reuse existing cluster assignments
```

---

## 3. Integration Patterns

### 3.1 Schema Discovery → Extraction Pipeline

**Use Case**: Discover dimensions from Q&A dataset, then use for extraction

**Workflow**:
```bash
# Step 1: Discover schema from Q&A dataset
python scripts/regenerate_schemas.py --dataset mirage --full

# Step 2: Copy discovered schema to extraction config
cp data/schema_discovery_mirage_v2/final_schema.yml \
   cfg/extraction/inference/discovered_dimensions.yml

# Step 3: Use discovered dimensions for extraction
stindex extract "Your text here" --config discovered_dimensions
```

**Programmatic**:
```python
# Step 1: Discover schema
from stindex.schema_discovery import SchemaDiscoveryPipeline

discovery_pipeline = SchemaDiscoveryPipeline(
    llm_config={'llm_provider': 'openai', 'model_name': 'gpt-4o-mini'},
    n_clusters=10
)

final_schema = discovery_pipeline.discover_schema(
    questions_file="data/original/mirage/train.jsonl",
    output_dir="data/schema_discovery"
)

# Step 2: Convert schema to dimension config
from stindex.extraction.dimension_loader import DimensionLoader

dimension_configs = {}
for dim_name, dim_schema in final_schema.dimensions.items():
    dimension_configs[dim_name] = {
        'enabled': True,
        'extraction_type': 'categorical',
        'schema_type': 'hierarchical_categorical',
        'hierarchy': dim_schema.hierarchy,
        'description': dim_schema.description,
        'examples': [e.text for e in dim_schema.entities[:5]]
    }

# Step 3: Use for extraction
from stindex.pipeline import STIndexPipeline
from stindex.preprocess import InputDocument

extraction_pipeline = STIndexPipeline(
    extractor_config="extract",
    dimension_config=dimension_configs  # Use discovered dimensions
)

docs = [InputDocument.from_text("Your text here")]
results = extraction_pipeline.run_pipeline(docs)
```

---

### 3.2 Iterative Schema Refinement

**Use Case**: Discover schema → Extract → Evaluate → Refine schema

**Workflow**:
```bash
# Iteration 1: Initial discovery
python scripts/regenerate_schemas.py --dataset mirage --full

# Iteration 2: Extract on test corpus
stindex extract "test_corpus.txt" --config discovered_dimensions

# Iteration 3: Evaluate extraction quality
stindex evaluate --config evaluate --sample-limit 100

# Iteration 4: Refine schema based on evaluation
# - Remove low-performing dimensions
# - Merge similar dimensions
# - Adjust hierarchies

# Iteration 5: Re-run discovery with refined prompts
python scripts/regenerate_schemas.py --dataset mirage --full
```

---

### 3.3 Multi-Dataset Schema Discovery

**Use Case**: Discover schemas from multiple datasets, merge across datasets

**Workflow**:
```bash
# Discover from MIRAGE (medical)
python scripts/regenerate_schemas.py --dataset mirage --full

# Discover from HotpotQA (multi-hop QA)
python scripts/regenerate_schemas.py --dataset hotpotqa --full

# Merge schemas programmatically
python scripts/merge_schemas.py \
    --schema1 data/schema_discovery_mirage_v2/final_schema.yml \
    --schema2 data/schema_discovery_hotpotqa_v2/final_schema.yml \
    --output data/merged_schema.yml
```

---

## 4. Summary

### STIndexPipeline (Extraction)
- **Purpose**: Extract dimensional entities from documents
- **Input**: Documents (URL/file/text)
- **Output**: Dimensional entities + analysis
- **Key Features**: Context-aware, two-pass reflection, spatiotemporal analysis

### SchemaDiscoveryPipeline (Discovery)
- **Purpose**: Discover dimensional schemas from Q&A datasets
- **Input**: Question dataset (JSONL)
- **Output**: Final schema (YAML/JSON)
- **Key Features**: Data-driven, hierarchical, parallel processing

### Integration
- Discover schema → Use for extraction
- Iterative refinement
- Multi-dataset merging

---

## 5. References

- **STIndexPipeline**: `stindex/pipeline/pipeline.py`
- **SchemaDiscoveryPipeline**: `stindex/schema_discovery/discover_schema.py`
- **Configuration Files**:
  - Extraction: `cfg/extraction/inference/extract.yml`
  - Dimensions: `cfg/extraction/inference/dimensions.yml`
  - Schema Discovery: `cfg/schema_discovery.yml`
- **Migration Guides**:
  - `docs/MIGRATION_GUIDE.md`
  - `docs/QUICKSTART_MIGRATION.md`
