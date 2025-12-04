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

### Two-Phase Approach

**Phase 1: Initial Schema Discovery** (HIGH-LEVEL)
- Sample ~20 questions from each cluster
- LLM proposes 2-3 domain-specific dimensions
- Get hierarchical structure + examples

**Phase 2: Entity Extraction + Discovery** (DETAILED)
- Extract entities from all questions
- Context-aware (maintain consistency)
- Can refine/propose new dimensions

## Prompt Classes

Located in `stindex/llm/prompts/`:

### 1. GlobalSchemaPrompt

HIGH-LEVEL schema discovery from cluster samples across all clusters

```python
from stindex.llm.prompts.initial_schema_discovery import GlobalSchemaPrompt

prompt = GlobalSchemaPrompt(n_schemas=2)
messages = prompt.build_messages(sample_questions[:20])
schemas = prompt.parse_response(llm_response.content)
```

### 2. ClusterEntityPrompt

Cluster-level entity extraction with dimensional refinement

```python
from stindex.llm.prompts.entity_extraction_with_discovery import ClusterEntityPrompt

prompt = ClusterEntityPrompt(
    dimensions=dimension_configs,
    extraction_context=context,
    allow_new_dimensions=True
)

messages = prompt.build_messages_for_question(question, index, total)
result = prompt.parse_response_with_discovery(llm_response.content)
# Returns: {'entities': {...}, 'new_dimensions': {...}}
```

## Example Usage

```bash
# Discover schema from any QA dataset
python -m stindex.schema_discovery.discover_schema \
    --questions data/original/{dataset}/train.jsonl \
    --corpus data/original/{corpus}/train.jsonl \
    --output cfg/discovered_schema_{dataset}.yml \
    --n-clusters 10 \
    --n-initial-schemas 2
```

## Implementation Steps

1. **Question Clustering** (`question_clusterer.py`) - Semantic clustering
2. **Schema Discovery** (`llm_schema_discoverer.py`) - LLM-based discovery
3. **Schema Optimization** (`schema_optimizer.py`) - Deduplication/merging
4. **Corpus Labeling** (`corpus_labeler.py`) - Label corpus with schemas
5. **End-to-End Pipeline** (`discover_schema.py`) - CLI interface

See `/Users/wenxiao/.claude/plans/atomic-swimming-milner.md` for detailed implementation.

## Current Status

**Phase 1: COMPLETE ✅**
- Domain-agnostic prompt classes created (`GlobalSchemaPrompt`, `ClusterEntityPrompt`)
- Question clustering module implemented
- Global dimensional discovery implemented
- Per-cluster entity extraction implemented

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

**Phase 2: IN PROGRESS**
- Testing entity-first format with CoT reasoning
- Batch processing with parallel cluster extraction

## Domain Examples

**Medical**: symptom → disease → disease_category → body_system
**Financial**: company → industry → sector → economy
**Scientific**: observation → hypothesis → theory → field

Discovery is data-driven, not hardcoded.
