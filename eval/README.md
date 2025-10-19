# STIndex Evaluation Framework

Comprehensive evaluation tools for spatiotemporal information extraction.

## Overview

This evaluation framework assesses STIndex's performance on:
- **Temporal extraction**: Identifying and normalizing time expressions (dates, times, durations)
- **Spatial extraction**: Identifying locations and geocoding to coordinates
- **Overall system performance**: Success rates, processing time

## Evaluation Metrics

### Based on Research Standards

Our metrics follow established NLP/IE evaluation practices:

**Entity-Level Evaluation** (not token-level):
- Following CoNLL-2003 NER and SemEval'13 standards
- Entities must match on both type and boundaries

**Temporal Metrics** (TempEval-3, TIMEX3 standard):
- Precision, Recall, F1 for extraction
- Normalization accuracy (ISO 8601 compliance)
- Type classification accuracy (date/time/duration/set)

**Spatial Metrics** (Geoparsing standards):
- Precision, Recall, F1 for extraction
- Geocoding success rate
- Distance error (mean, median in km)
- Accuracy@k (percentage within k km of ground truth)
- Type classification accuracy (city/region/country)

**Overall Metrics**:
- Combined F1 (macro-average of temporal and spatial)
- Extraction success rate
- Average processing time

## Usage

### 1. Generate Evaluation Dataset

```bash
python eval/generate_dataset.py
```

This creates `data/input/eval_dataset_100.json` with 100 annotated test cases.

### 2. Run Evaluation

```python
from eval import run_evaluation

# Run evaluation on dataset
metrics = run_evaluation("data/input/eval_dataset_100.json")
```

Or via command line:

```bash
python eval/evaluation.py data/input/eval_dataset_100.json
```

### 3. Review Results

Results are saved to `data/output/eval_results/`:

**Detailed Results** (`detailed_results_TIMESTAMP.json`):
- For each entry:
  - Input text
  - Input prompt
  - Ground truth annotations
  - LLM raw output
  - Final output (after post-processing)
  - Per-entry evaluation metrics

**Metrics Summary** (`metrics_summary_TIMESTAMP.json`):
- Overall performance metrics
- Temporal extraction metrics
- Spatial extraction metrics

## Dataset Format

```json
{
  "id": "entry_001",
  "text": "On March 15, 2022, a cyclone hit Broome, Western Australia.",
  "prompt": "Extract all temporal and spatial information...",
  "ground_truth": {
    "temporal": [
      {
        "text": "March 15, 2022",
        "normalized": "2022-03-15",
        "temporal_type": "DATE"
      }
    ],
    "spatial": [
      {
        "text": "Broome",
        "location_type": "CITY",
        "latitude": -17.9614,
        "longitude": 122.2359
      }
    ]
  },
  "metadata": {
    "category": "weather",
    "complexity": "simple"
  }
}
```

## Output Format

Each evaluation entry produces:

```json
{
  "id": "entry_001",
  "text": "...",
  "input_prompt": "...",
  "ground_truth": {...},
  "llm_raw_output": {
    "temporal_mentions": [...],
    "spatial_mentions": [...]
  },
  "final_output": {
    "temporal_entities": [...],
    "spatial_entities": [...]
  },
  "evaluation": {
    "success": true,
    "processing_time": 2.35,
    "temporal_metrics": {...},
    "spatial_metrics": {...}
  }
}
```

## Metrics Explained

### Temporal Metrics

- **Precision**: % of extracted temporal expressions that are correct
- **Recall**: % of ground truth temporal expressions that were found
- **F1 Score**: Harmonic mean of precision and recall
- **Normalization Accuracy**: % of temporal expressions correctly normalized to ISO 8601
- **Type Accuracy**: % of temporal types correctly classified

### Spatial Metrics

- **Precision**: % of extracted locations that are correct
- **Recall**: % of ground truth locations that were found
- **F1 Score**: Harmonic mean of precision and recall
- **Geocoding Success Rate**: % of locations successfully geocoded
- **Mean Distance Error**: Average distance (km) between predicted and ground truth coordinates
- **Accuracy@25km**: % of predictions within 25km of ground truth (standard threshold)
- **Type Accuracy**: % of location types correctly classified

### Overall Metrics

- **Combined F1**: Macro-average F1 across temporal and spatial
- **Success Rate**: % of documents successfully processed
- **Avg Processing Time**: Average seconds per document

## Customization

### Adding Custom Metrics

Edit `eval/metrics.py` to add custom evaluation logic:

```python
@dataclass
class CustomMetrics:
    # Add your metrics here
    pass
```

### Custom Dataset

Create your own dataset following the format in `eval/generate_dataset.py`:

```python
dataset = [
    {
        "id": "custom_001",
        "text": "Your text here",
        "ground_truth": {
            "temporal": [...],
            "spatial": [...]
        }
    }
]
```

## Matching Strategies

### Temporal Matching

1. **Exact**: Exact string match
2. **Overlap**: Significant word overlap (IoU ≥ 0.5)
3. **Normalized**: Match on normalized values

### Spatial Matching

1. **Exact**: Exact string match
2. **Fuzzy**: Substring or word overlap

## References

- CoNLL-2003 NER Shared Task
- TempEval-3: TIMEX3 Evaluation
- SemEval'13: Entity-Level Metrics
- ACE TERN: Temporal Expression Recognition
- Geoparsing Evaluation (Language Resources and Evaluation, 2019)

## Citation

If you use this evaluation framework, please cite:

```
STIndex Evaluation Framework
Spatiotemporal Index Extraction
2025
```
