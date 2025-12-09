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

### Pure Cluster-Level Discovery (v2.0)

**Architecture**: Each cluster discovers dimensions independently, then schemas are merged

**Per-Cluster Two-Phase Approach**:

**Phase 1: Cluster-Level Schema Discovery**
- Sample ~20 questions from within the cluster
- LLM proposes dimensions specific to this cluster's questions
- Get hierarchical structure + examples
- No dependency on global baseline

**Phase 2: Cluster-Level Entity Extraction**
- Extract entities from all questions in cluster
- Use dimensions discovered in Phase 1
- Can discover new dimensions during extraction
- Context-aware (maintain consistency within cluster)

**Phase 3: Cross-Cluster Merging**
- Align dimensions across clusters using fuzzy matching
- Deduplicate entities with similarity threshold (0.85)
- No global baseline - pure cluster-to-cluster merging

## Core Components

### 1. ClusterSchemaDiscoverer

Per-cluster discovery and extraction with Pydantic models

```python
from stindex.schema_discovery import ClusterSchemaDiscoverer
from stindex.schema_discovery.models import ClusterSchemaDiscoveryResult

discoverer = ClusterSchemaDiscoverer(
    llm_config={'llm_provider': 'openai', 'model_name': 'gpt-4o-mini'},
    n_schemas=10,
    batch_size=50
)

result: ClusterSchemaDiscoveryResult = discoverer.discover_and_extract(
    cluster_id=0,
    cluster_questions=questions,
    n_samples_for_discovery=20,
    allow_new_dimensions=True
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
# Discover schema from any QA dataset (pure cluster-level)
python -m stindex.schema_discovery.discover_schema \
    --questions data/original/mirage/train.jsonl \
    --output-dir data/schema_discovery_mirage \
    --n-clusters 10 \
    --n-samples 20 \
    --config cfg/schema_discovery.yml

# Test with specific clusters
python -m stindex.schema_discovery.discover_schema \
    --questions data/original/mirage/train.jsonl \
    --output-dir data/schema_discovery_test \
    --n-clusters 3 \
    --n-samples 10 \
    --test-clusters "0,1"

# Enable parallel processing (default)
python -m stindex.schema_discovery.discover_schema \
    --questions data/original/mirage/train.jsonl \
    --output-dir data/schema_discovery_mirage \
    --max-workers 5  # Process 5 clusters in parallel
```

### Python API

```python
from stindex.schema_discovery import SchemaDiscoveryPipeline
from stindex.schema_discovery.models import FinalSchema

# Initialize pipeline
pipeline = SchemaDiscoveryPipeline(
    llm_config={'llm_provider': 'openai', 'model_name': 'gpt-4o-mini'},
    n_clusters=10,
    n_samples_for_discovery=20,
    n_schemas_per_cluster=10,
    enable_parallel=True,
    max_workers=5
)

# Run discovery
final_schema: FinalSchema = pipeline.discover_schema(
    questions_file='data/original/mirage/train.jsonl',
    output_dir='data/schema_discovery_mirage',
    reuse_clusters=True
)

# Access results
print(f"Discovered {len(final_schema.dimensions)} dimensions")
for dim_name in final_schema.get_dimension_names():
    dimension = final_schema.dimensions[dim_name]
    print(f"  • {dim_name}: {dimension.total_entity_count} entities")

# Export to YAML
import yaml
yaml_dict = final_schema.to_yaml_dict()
with open('output_schema.yml', 'w') as f:
    yaml.dump(yaml_dict, f, sort_keys=False, indent=2)
```

## Output Format (v2.0)

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

### Key Differences from v1.0

| Feature | v1.0 (Global-Seeded) | v2.0 (Cluster-Level) |
|---------|---------------------|---------------------|
| Architecture | 4 steps with global baseline | 3 steps, no global baseline |
| Discovery | Global → Per-cluster | Per-cluster only |
| Data Format | Dict-based | Pydantic models |
| Entity Format | Mixed fields (`hierarchy_level_1`, `hierarchy`) | Consistent `hierarchy_values` dict |
| Output Files | `global_dimensions.json` + mixed formats | Pydantic JSON + YAML |
| Validation | Runtime errors possible | Type-safe with Pydantic |
| Deduplication | Exact match | Fuzzy matching (0.85 threshold) |

## Implementation Steps

1. **Question Clustering** (`question_clusterer.py`) - Semantic clustering with FAISS
2. **Cluster Schema Discovery** (`cluster_schema_discoverer.py`) - Per-cluster dimension discovery + entity extraction
3. **Schema Merging** (`schema_merger.py`) - Cross-cluster dimension alignment and deduplication
4. **End-to-End Pipeline** (`discover_schema.py`) - CLI interface with parallel processing

## Pydantic Models

All data structures use type-safe Pydantic models (`models.py`):

- `DiscoveredDimensionSchema` - Dimension with hierarchy, description, examples
- `HierarchicalEntity` - Entity with text, dimension, hierarchy_values, confidence
- `ClusterSchemaDiscoveryResult` - Per-cluster discovery + extraction result
- `MergedDimensionSchema` - Merged dimension with deduplicated entities
- `FinalSchema` - Complete final schema with all dimensions
- `DimensionSource` - Tracks which clusters contributed to each dimension

## Current Status

**Version 2.0: COMPLETE ✅**
- Pure cluster-level discovery architecture
- Pydantic models throughout
- Fuzzy entity deduplication (0.85 threshold)
- Parallel cluster processing (configurable workers)
- Comprehensive unit tests (44 tests, 100% pass rate)
- Migration guide provided

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
# Run all schema discovery tests
python -m pytest tests/schema_discovery/ -v

# Run specific test file
python -m pytest tests/schema_discovery/test_models.py -v
python -m pytest tests/schema_discovery/test_schema_merger.py -v
```

Current test coverage: 44 tests, 100% pass rate

## Domain Examples

**Medical**: symptom → disease → disease_category → body_system
**Financial**: company → industry → sector → economy
**Scientific**: observation → hypothesis → theory → field

Discovery is data-driven, not hardcoded.
