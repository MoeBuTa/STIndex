# Schema Discovery Module

Domain-agnostic schema discovery for question-answering datasets.

## Overview

This module discovers dimensional schemas from question-answering datasets without hardcoded domain knowledge. It works for any domain: medical, financial, scientific, etc.

### Datasets

**Currently Working On:**
- **MIRAGE** (medical benchmark): 6,545 questions, 0 documents
  - Filtered clinical-only from original 7,663 questions
  - MedQA-US: 1,273 questions (US Medical Licensing Exam)
  - MedMCQA: 4,183 questions (Indian medical entrance exams)
  - MMLU-Med: 1,089 questions (6 biomedical tasks from MMLU)
  - **Filtered out**: PubMedQA (500), BioASQ (618) - requires PubMed corpus
  - **Purpose**: Evaluation only (benchmark)
- **MedCorp** (medical corpus): 0 questions, 125,847 textbook documents
  - **Purpose**: Retrieval corpus only

**Planned Datasets:**
- **HotpotQA** (multi-hop QA): 90,425 questions, ~60K documents
  - **Purpose**: Training & Evaluation
- **2WikiMQA** (multi-hop QA): 165,464 questions, ~100K documents
  - **Purpose**: Training & Evaluation
- **MuSiQue** (multi-hop QA): 19,938 questions, ~20K documents
  - **Purpose**: Training & Evaluation

**Total Across All Datasets**: ~282K questions, ~1.3M documents

### Pure Cluster-Level Discovery (v3.0)

**Architecture**: Unified single-phase approach where discovery and extraction happen together

**Unified Batch Processing**:

**First Batch: Discovery + Extraction (Adaptive Size)**
- Adaptive batch size: max(50, min(150, cluster_size * 0.10))
- LLM discovers dimensions AND extracts entities simultaneously
- No separate discovery phase - saves 10% LLM calls
- Confidence threshold: 0.3 (easy to propose new dimensions)

**Subsequent Batches: Refinement + Extraction (With Decay)**
- Standard batch size: 50 questions
- Extract entities using discovered dimensions
- Can propose new dimensions with increasing confidence requirements:
  - Batches 1-2: threshold 0.3 (early - easy to propose)
  - Batches 3-5: threshold 0.6 (medium - moderate difficulty)
  - Batches 6+: threshold 0.9 (late - rare proposals only)
- Prevents schema instability in later batches

**Cross-Cluster Merging**
- Align dimensions across clusters using fuzzy matching
- Deduplicate entities with similarity threshold (0.85)
- No global baseline - pure cluster-to-cluster merging

## Core Components

### 1. ClusterSchemaDiscoverer

Per-cluster discovery and extraction with Pydantic models

```python
from stindex.discovery import ClusterSchemaDiscoverer
from stindex.discovery.models import ClusterSchemaDiscoveryResult

discoverer = ClusterSchemaDiscoverer(
    llm_config={'llm_provider': 'openai', 'model_name': 'gpt-4o-mini'},
    batch_size=50
)

result: ClusterSchemaDiscoveryResult = discoverer.discover_and_extract(
    cluster_id=0,
    cluster_questions=questions,
    adaptive_first_batch=True,  # Use adaptive sizing
    first_batch_min=50,
    first_batch_max=150,
    first_batch_ratio=0.10,     # First batch = 10% of cluster
    allow_new_dimensions=True,
    decay_config={               # Progressive thresholds
        'early': (1, 2, 0.3),
        'medium': (3, 5, 0.6),
        'late': (6, 999999, 0.9)
    }
)

# Access discovered dimensions
for dim_name, dim_schema in result.discovered_dimensions.items():
    print(f"{dim_name}: {dim_schema.hierarchy}")

# Access extracted entities
for dim_name, entities in result.entities.items():
    for entity in entities:
        print(f"{entity.text}: {entity.hierarchy_values}")
```

### 2. SchemaMerger

Cross-cluster schema merging without global baseline

```python
from stindex.schema_discovery import SchemaMerger
from stindex.schema_discovery.models import FinalSchema

merger = SchemaMerger(similarity_threshold=0.85)
final_schema: FinalSchema = merger.merge_clusters(cluster_results)

# Access merged dimensions
for dim_name in final_schema.get_dimension_names():
    dimension = final_schema.dimensions[dim_name]
    print(f"{dim_name}: {dimension.total_entity_count} entities")
    print(f"  From {len(dimension.sources.cluster_ids)} clusters")
```

## Example Usage

### CLI

```bash
# Discover schema from any QA dataset (unified approach)
python -m stindex.pipeline.discovery_pipeline \
    --questions data/original/mirage/train.jsonl \
    --output-dir data/schema_discovery_mirage \
    --n-clusters 10

# Test with specific clusters
python -m stindex.pipeline.discovery_pipeline \
    --questions data/original/mirage/train.jsonl \
    --output-dir data/schema_discovery_test \
    --n-clusters 3 \
    --test-clusters "0,1"

# Enable parallel processing (default)
python -m stindex.pipeline.discovery_pipeline \
    --questions data/original/mirage/train.jsonl \
    --output-dir data/schema_discovery_mirage \
    --max-workers 5  # Process 5 clusters in parallel
```

### Python API

```python
from stindex.pipeline.discovery_pipeline import SchemaDiscoveryPipeline
from stindex.discovery.models import FinalSchema

# Initialize pipeline
pipeline = SchemaDiscoveryPipeline(
    llm_config={'llm_provider': 'openai', 'model_name': 'gpt-4o-mini'},
    n_clusters=10,
    enable_parallel=True,
    max_workers=5
)

# Run discovery
final_schema: FinalSchema = pipeline.discover_schema(
    questions_file='data/original/mirage/train.jsonl',
    output_dir='data/schema_discovery_mirage'
)

# Access results
print(f"Discovered {len(final_schema.dimensions)} dimensions")
for dim_name in final_schema.get_dimension_names():
    dimension = final_schema.dimensions[dim_name]
    print(f"  • {dim_name}: {dimension.total_entity_count} entities")

# Export to YAML (done automatically by pipeline)
# Output saved to: data/schema_discovery_mirage/final_schema.yml
```

## Output Format (v3.0)

### Directory Structure

```
data/schema_discovery_mirage/
├── cluster_assignments.csv        # Question → cluster mapping
├── cluster_samples.json           # Sample questions per cluster
├── cluster_0_result.json          # Cluster 0 result (Pydantic format)
├── cluster_1_result.json          # Cluster 1 result
├── ...
├── final_schema.json              # Final merged schema (Pydantic)
├── final_schema.yml               # Final merged schema (YAML)
└── cot/                           # Chain-of-thought reasoning logs
    ├── cluster_0/
    │   ├── discovery_reasoning.txt
    │   ├── batch_000_reasoning.txt
    │   └── ...
    └── reasoning_summary.json
```

### Final Schema Format (JSON)

```json
{
  "dimensions": {
    "symptom": {
      "hierarchy": ["specific_symptom", "symptom_category"],
      "description": "Medical symptoms",
      "examples": ["fever", "cough"],
      "entities": [
        {
          "text": "fever",
          "hierarchy_values": {
            "specific_symptom": "fever",
            "symptom_category": "systemic"
          },
          "dimension": "symptom",
          "confidence": 1.0
        }
      ],
      "total_entity_count": 127,
      "sources": {
        "cluster_ids": [0, 1, 2],
        "entity_counts_per_cluster": {
          "0": 45,
          "1": 38,
          "2": 44
        }
      },
      "alternative_names": ["symptoms", "clinical_signs"]
    }
  },
  "n_clusters_processed": 10,
  "total_questions_processed": 6545,
  "pipeline_time": 1234.56,
  "timestamp": "2025-01-15T10:30:00"
}
```

### Key Differences from v2.0

| Feature | v2.0 (Two-Phase) | v3.0 (Unified) |
|---------|------------------|----------------|
| Architecture | 2 steps (discovery + extraction) | 1 step (unified) |
| Discovery | Separate 20-sample phase | First batch (adaptive size) |
| LLM Calls | 1 (discovery) + N (extraction) | N (unified) |
| First Batch | Standard size (50) | Adaptive (10% of cluster, 50-150) |
| Schema Refinement | Constant threshold | Progressive decay (0.3 → 0.6 → 0.9) |
| Efficiency | Baseline | 10% fewer LLM calls |
| Schema Stability | Risk of late changes | Decay prevents instability |

## Implementation Steps

1. **Question Clustering** (`question_clusterer.py`) - Semantic clustering with FAISS
2. **Unified Discovery + Extraction** (`cluster_schema_discoverer.py`, `cluster_entity_extractor.py`) - First batch discovers schema + extracts entities, subsequent batches refine with decay
3. **Schema Merging** (`schema_merger.py`) - Cross-cluster dimension alignment and deduplication
4. **End-to-End Pipeline** (`stindex/pipeline/discovery_pipeline.py`) - CLI interface with parallel processing

## Pydantic Models

All data structures use type-safe Pydantic models (`models.py`):

- `DiscoveredDimensionSchema` - Dimension with hierarchy, description, examples
- `HierarchicalEntity` - Entity with text, dimension, hierarchy_values, confidence
- `ClusterSchemaDiscoveryResult` - Per-cluster discovery + extraction result
- `MergedDimensionSchema` - Merged dimension with deduplicated entities
- `FinalSchema` - Complete final schema with all dimensions
- `DimensionSource` - Tracks which clusters contributed to each dimension

## Current Status

**Version 3.0: COMPLETE ✅**
- Unified single-phase discovery + extraction
- Adaptive first batch sizing (10% of cluster, min 50, max 150)
- Schema refinement decay (0.3 → 0.6 → 0.9)
- 10% fewer LLM calls vs v2.0
- Pydantic models throughout
- Fuzzy entity deduplication (0.85 threshold)
- Parallel cluster processing (configurable workers)
- Comprehensive unit tests (44 tests, 100% pass rate)

**Current Datasets:**
- **MIRAGE** (medical): 6,545 questions (filtered clinical-only from original 7,663)
  - MedQA-US: 1,273 questions
  - MedMCQA: 4,183 questions
  - MMLU-Med: 1,089 questions
  - Filtered out: PubMedQA (500), BioASQ (618) - requires PubMed corpus
- **MedCorp** corpus: 125,847 medical textbook snippets

**Planned Datasets:**
- **HotpotQA**: 90,425 multi-hop questions
- **2WikiMQA**: 165,464 multi-hop questions
- **MuSiQue**: 19,938 multi-hop questions

## Migration from v1.0

See `MIGRATION_GUIDE.md` in project root for:
- Config file updates
- Python API changes
- Schema regeneration instructions
- Troubleshooting

## Testing

Run unit tests:
```bash
# Run all discovery module tests
python -m pytest tests/discovery/ -v

# Run specific test file
python -m pytest tests/discovery/test_models.py -v
python -m pytest tests/discovery/test_schema_merger.py -v
```

Current test coverage: 44 tests, 100% pass rate

## Domain Examples

**Medical**: symptom → disease → disease_category → body_system
**Financial**: company → industry → sector → economy
**Scientific**: observation → hypothesis → theory → field

Discovery is data-driven, not hardcoded.
