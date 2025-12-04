# Schema Discovery Pipeline Guide

## Overview

The simplified schema discovery pipeline discovers dimensional schemas from question-answering datasets using:
- **Global dimensional discovery** (`GlobalSchemaPrompt`) - Sample questions from all clusters to discover universal dimensions
- **Cluster-level entity extraction** (`ClusterEntityPrompt`) - Extract entities from each cluster in parallel
- **Simple entity tracking** - Track unique entities per dimension (no FAISS, no embeddings)
- **Fuzzy deduplication** - Merge similar entities across clusters

**Performance**: ~100 seconds faster than complex retrieval-based approach

---

## Datasets

### Currently Working On

**MIRAGE** (medical benchmark): 6,545 questions | ACL 2024
- Filtered clinical-only from original 7,663 questions
- **MedQA-US**: 1,273 questions (US Medical Licensing Exam)
- **MedMCQA**: 4,183 questions (Indian medical entrance exams)
- **MMLU-Med**: 1,089 questions (6 biomedical tasks from MMLU)
- **Filtered out**: PubMedQA (500), BioASQ (618) - requires PubMed corpus
- **Schema Potential**: ⭐⭐⭐⭐⭐ (temporal, spatial, medical entities, categorical, relational)
- **Source**: [GitHub](https://github.com/Teddy-XiongGZ/MIRAGE)

**MedCorp** (medical corpus): 125,847 textbook snippets

### Planned Datasets

**Multi-hop QA (Already Downloaded):**
- **HotpotQA**: 90,425 questions, ~60K documents
- **2WikiMQA**: 165,464 questions, ~100K documents
- **MuSiQue**: 19,938 questions, ~20K documents

**Financial Domain (Recommended Next - EMNLP/ACL 2021):**
- **FinQA**: 8,281 QA pairs from 10-K financial reports
  - Schema Potential: ⭐⭐⭐⭐☆ (fiscal years/quarters, companies/sectors, financial metrics)
  - Integration: Medium (table parsing, HuggingFace available)
- **TAT-QA**: ~17K hybrid tabular+textual financial QA
- **FinDER** (2025): RAG-specific with ambiguity handling

**Alternative Options:**
- **MedExQA** (BioNLP 2024): 2,400 healthcare topics, USMLE-based
- **Amazon Reviews 2023**: Millions (sample by category, ⭐⭐⭐☆☆ schema potential)

---

## Architecture

```
Step 1: Question Clustering (ALREADY COMPLETE ✅)
└─> 10 clusters, ~654 questions each
    └─> Output: cluster_samples.json, cluster_assignments.csv

Step 2: Global Dimensional Discovery
├─> Sample 200 questions (20 from each cluster)
├─> LLM discovers global dimensions
└─> Output: global_dimensions.json
    Example: {'symptom': {...}, 'disease': {...}, 'anatomy': {...}}

Step 3: Per-Cluster Entity Extraction (PARALLEL)
For each cluster (can run in parallel):
  ├─> Start with global dimensions + empty entity lists
  ├─> For each question (~654 per cluster):
  │   ├─> Show: dimension names + current entity lists
  │   ├─> LLM extracts entities
  │   └─> Update entity lists (simple set.add())
  └─> Output: cluster_X_entities.json

Step 4: Merge & Deduplicate
├─> Load all 10 cluster entity lists
├─> Concatenate per dimension
├─> Fuzzy deduplication (threshold=0.85)
└─> Output: final_schema.yml, final_schema.json
```

---

## Quick Start

### Prerequisites

```bash
# Ensure you're in the project directory
cd /Users/wenxiao/PycharmProjects/STIndex

# Set your OpenAI API key (or use other providers)
export OPENAI_API_KEY="your-key-here"

# Verify cluster results exist (already created)
ls data/schema_discovery/clusters/
# Should see: cluster_samples.json, cluster_assignments.csv
```

### Run the Pipeline

**Option 1: Command Line (Recommended)**

```bash
# Full pipeline (reuses existing clusters)
python -m stindex.schema_discovery.discover_schema \
    --questions data/original/mirage/train.jsonl \
    --output-dir data/schema_discovery \
    --reuse-clusters \
    --llm-provider openai \
    --model gpt-4o-mini

# Small test run (3 clusters, faster)
python -m stindex.schema_discovery.discover_schema \
    --questions data/original/mirage/train.jsonl \
    --output-dir data/schema_discovery_test \
    --n-clusters 3 \
    --n-samples 10 \
    --llm-provider openai \
    --model gpt-4o-mini
```

**Option 2: Python API**

```python
from stindex.schema_discovery import SchemaDiscoveryPipeline

# Configure LLM
llm_config = {
    'llm_provider': 'openai',  # or 'anthropic', 'hf'
    'model_name': 'gpt-4o-mini'
}

# Initialize pipeline
pipeline = SchemaDiscoveryPipeline(
    llm_config=llm_config,
    n_clusters=10,
    n_samples_per_cluster=20,
    similarity_threshold=0.85
)

# Run full pipeline
result = pipeline.discover_schema(
    questions_file='data/original/mirage/train.jsonl',
    output_dir='data/schema_discovery',
    reuse_clusters=True
)

# Access results
print(f"Discovered {len(result['final_schema'])} dimensions")
for dim_name, dim_info in result['final_schema'].items():
    print(f"  {dim_name}: {dim_info['count']} entities")
```

---

## Configuration Options

### LLM Providers

**OpenAI** (requires `OPENAI_API_KEY`):
```bash
--llm-provider openai --model gpt-4o-mini       # Cheap, fast
--llm-provider openai --model gpt-4o            # Better quality
--llm-provider openai --model gpt-3.5-turbo     # Cheapest
```

**Anthropic** (requires `ANTHROPIC_API_KEY`):
```bash
--llm-provider anthropic --model claude-3-haiku-20240307
--llm-provider anthropic --model claude-3-5-sonnet-20241022
```

**HuggingFace/MS-SWIFT** (local models):
```bash
--llm-provider hf --model Qwen/Qwen2.5-7B-Instruct
# Requires MS-SWIFT server running (see cfg/extraction/inference/hf.yml)
```

### Pipeline Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--n-clusters` | 10 | Number of question clusters |
| `--n-samples` | 20 | Samples per cluster for global discovery |
| `--similarity-threshold` | 0.85 | Fuzzy matching threshold (0.0-1.0) |
| `--reuse-clusters` | False | Reuse existing cluster results |
| `--output-dir` | data/schema_discovery | Output directory |

---

## Expected Output Files

```
data/schema_discovery/
├── cluster_assignments.csv          # Question → cluster mapping
├── cluster_analysis.json            # Cluster statistics
├── cluster_samples.json             # 20 samples per cluster
│
├── global_dimensions.json           # Discovered dimensions (Step 2)
│   Example:
│   {
│     "symptom": {
│       "hierarchy": ["specific_symptom", "symptom_category"],
│       "description": "Patient symptoms and clinical signs",
│       "examples": ["fever", "cough", "headache"]
│     },
│     "disease": {...},
│     "anatomy": {...}
│   }
│
├── cluster_0_entities.json          # Entities from cluster 0 (Step 3)
├── cluster_1_entities.json          # Entities from cluster 1
├── ...
├── cluster_9_entities.json          # Entities from cluster 9
│   Example:
│   {
│     "cluster_id": 0,
│     "n_questions": 687,
│     "entities": {
│       "symptom": ["fever", "cough", "dyspnea", ...],
│       "disease": ["pneumonia", "COVID-19", ...]
│     },
│     "entity_counts": {"symptom": 127, "disease": 89}
│   }
│
├── final_schema.yml                 # MAIN OUTPUT (Step 4)
└── final_schema.json                # JSON version
    Example:
    {
      "symptom": {
        "hierarchy": ["specific_symptom", "symptom_category"],
        "description": "...",
        "examples": ["fever", "cough"],
        "entities": ["fever", "cough", "dyspnea", ...],  # Deduplicated
        "count": 347,
        "sources": {"cluster_0": 45, "cluster_1": 38, ...}
      }
    }
```

---

## Step-by-Step Workflow

### Step 1: Question Clustering (Already Complete)

```bash
# This step is already done! Results in:
# - data/schema_discovery/clusters/cluster_samples.json
# - data/schema_discovery/clusters/cluster_assignments.csv

# If you need to re-cluster:
from stindex.schema_discovery import QuestionClusterer
clusterer = QuestionClusterer()
clusterer.cluster_questions_from_file(
    questions_file='data/original/mirage/train.jsonl',
    output_dir='data/schema_discovery/clusters',
    n_clusters=10
)
```

### Step 2: Global Dimensional Discovery

**What it does**: LLM analyzes 200 representative questions to discover ~5 universal dimensions

**LLM calls**: 1 (analyzes all 200 questions at once)

**Example output**:
```json
{
  "symptom": {
    "hierarchy": ["specific_symptom", "symptom_category"],
    "description": "Patient symptoms like fever, cough, pain"
  },
  "disease": {
    "hierarchy": ["disease_name", "disease_category", "body_system"],
    "description": "Medical conditions and diseases"
  }
}
```

### Step 3: Entity Extraction (Per Cluster)

**What it does**: For each cluster, extract entities from all questions using global dimensions

**LLM calls**: ~6,545 (one per question across all clusters)

**Context shown to LLM** (example after processing 100 questions):
```
# Discovered Dimensions
- symptom: fever, cough, headache, dyspnea, chest pain ... (53 total)
- disease: pneumonia, influenza, diabetes, hypertension ... (27 total)
- anatomy: heart, lung, liver, brain, kidney ... (18 total)
```

**How entities are tracked**:
```python
# Simple set.add() - no complex retrieval!
self.entity_lists['symptom'].add('fever')
self.entity_lists['symptom'].add('cough')
# Result: {'symptom': {'fever', 'cough', ...}}
```

### Step 4: Merge & Deduplicate

**What it does**: Combines entity lists from all 10 clusters with fuzzy matching

**Fuzzy matching examples**:
- "fever" + "Fever " → keep "fever" (exact after normalization)
- "myocardial infarction" + "myocardial infarction" → keep one (exact match)
- "influenza" + "flu" → keep both (similarity = 0.27 < 0.85)

**Algorithm**:
```python
similarity = difflib.SequenceMatcher(None, text1, text2).ratio()
if similarity >= 0.85:
    # Consider duplicate
```

---

## Performance & Cost Estimates

### Full Pipeline (10 clusters, 6,545 questions)

**Time**:
- Step 1 (Clustering): ~35 seconds (already done)
- Step 2 (Global discovery): ~10 seconds (1 LLM call)
- Step 3 (Entity extraction): ~30-60 minutes (6,545 LLM calls)
- Step 4 (Merge): ~5 seconds

**Total**: ~30-60 minutes (depending on LLM speed)

**Cost** (OpenAI gpt-4o-mini):
- Input: ~6,545 questions × 200 tokens = ~1.3M tokens
- Output: ~6,545 responses × 100 tokens = ~0.65M tokens
- **Total cost**: ~$0.02-0.05 USD

### Test Run (3 clusters, ~2,000 questions)

**Time**: ~10-15 minutes
**Cost**: ~$0.01 USD

---

## Monitoring Progress

The pipeline logs progress at multiple levels:

```bash
# Watch the pipeline run
python -m stindex.schema_discovery.discover_schema ... 2>&1 | tee pipeline.log

# Logs show:
# - Step 1: ✓ Reusing existing cluster samples
# - Step 2: ✓ Discovered 5 dimensions: symptom, disease, anatomy, treatment, diagnostic
# - Step 3: Processing Cluster 0... Progress: 50/687 questions, 245 unique entities
# - Step 4: Merging entity lists... symptom: 347 unique entities (from 1,234 total)
```

**Check intermediate results**:
```bash
# After Step 2
cat data/schema_discovery/global_dimensions.json

# After Step 3 (per cluster)
ls data/schema_discovery/cluster_*_entities.json
cat data/schema_discovery/cluster_0_entities.json

# After Step 4 (final)
cat data/schema_discovery/final_schema.yml
```

---

## Troubleshooting

### Issue: "No valid JSON found in LLM output"

**Cause**: LLM didn't return valid JSON

**Fix**:
1. Check your API key is set correctly
2. Try a different model (e.g., `gpt-4o` instead of `gpt-4o-mini`)
3. Reduce `n_samples` to simplify the discovery prompt

### Issue: "Failed to extract from question X"

**Cause**: LLM error or rate limit on specific question

**Fix**: Pipeline continues processing other questions automatically. Check logs for details.

### Issue: Pipeline is slow

**Solutions**:
1. Use faster model: `gpt-4o-mini` or `claude-3-haiku`
2. Run parallel cluster processing (future enhancement)
3. Reduce number of clusters: `--n-clusters 5`

### Issue: Too many/few dimensions discovered

**Solution**: Adjust `n_schemas` in `global_schema_discoverer.py:73`:
```python
prompt = GlobalSchemaPrompt(
    n_schemas=3,  # Change from 5 to 3 for fewer dimensions
    ...
)
```

---

## Next Steps After Schema Discovery

Once you have `final_schema.yml`, you can:

1. **Use for RAG retrieval filtering**:
   ```python
   # Filter corpus by discovered dimensions
   retriever.filter_by_dimension('disease', 'pneumonia')
   ```

2. **Generate training data for entity extraction**:
   ```python
   # Use discovered entities as weak labels
   entities = schema['symptom']['entities']
   ```

3. **Validate and refine schema**:
   ```python
   # Manual review and editing of final_schema.yml
   # Add/remove dimensions or entities as needed
   ```

4. **Export to STIndex dimension config**:
   ```yaml
   # Convert to cfg/extraction/inference/discovered_dimensions.yml
   dimensions:
     symptom:
       enabled: true
       hierarchy: [specific_symptom, symptom_category]
       ...
   ```

---

## Summary

**Simple Architecture Benefits**:
- ✅ ~800 lines of code (vs ~1500 in complex design)
- ✅ ~100 seconds faster (no embedding/retrieval overhead)
- ✅ Easy to understand and debug
- ✅ Achieves same consistency goals by showing entity lists

**Ready to Run**: Existing cluster results can be reused, pipeline can start immediately from Step 2.

**Expected Results**: ~5 dimensions with ~300-500 unique entities each for MIRAGE medical dataset.
