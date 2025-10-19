# STIndex Evaluation Framework - Summary

## Completed Deliverables

### 1. Evaluation Metrics (`eval/metrics.py`)

Comprehensive metrics based on NLP/IE research standards:

**Standards Followed:**
- CoNLL-2003 NER (entity-level evaluation)
- TempEval-3 / TIMEX3 (temporal expression evaluation)
- SemEval'13 (entity-level metrics)
- ACE TERN (Temporal Expression Recognition and Normalization)
- Geoparsing standards (distance-based accuracy)

**Metrics Implemented:**

*Temporal Metrics:*
- ✅ Precision, Recall, F1 Score (entity-level)
- ✅ Normalization Accuracy (ISO 8601 compliance)
- ✅ Type Classification Accuracy (date/time/duration/set)
- ✅ True Positives, False Positives, False Negatives tracking

*Spatial Metrics:*
- ✅ Precision, Recall, F1 Score (entity-level)
- ✅ Geocoding Success Rate
- ✅ Distance Error Metrics (mean, median in km)
- ✅ Accuracy@k (25km, 50km, 100km thresholds)
- ✅ Type Classification Accuracy (city/region/country)
- ✅ True Positives, False Positives, False Negatives tracking

*Overall Metrics:*
- ✅ Combined F1 (macro-average across temporal and spatial)
- ✅ Success Rate
- ✅ Average Processing Time

### 2. Evaluation Framework (`eval/evaluation.py`)

Complete evaluation runner with:

**Features:**
- ✅ Batch evaluation on datasets
- ✅ Per-entry evaluation with detailed results
- ✅ Automatic metrics calculation
- ✅ Result saving (detailed + summary)
- ✅ Rich console output with formatted tables
- ✅ Error handling and logging

**Output Format:**
Each entry includes:
- Input text
- Input prompt
- Ground truth annotations
- LLM raw output (temporal_mentions, spatial_mentions)
- Final output (after geocoding/post-processing)
- Per-entry evaluation metrics

**Matching Strategies:**
- Temporal: Exact match, overlap-based, normalized value match
- Spatial: Exact match, fuzzy match with coordinate validation

### 3. Evaluation Dataset (`data/input/eval_dataset_100.json`)

**Dataset Statistics:**
- Total entries: 100
- Temporal entities: 110 (avg 1.1 per entry)
- Spatial entities: 130 (avg 1.3 per entry)

**Categories:**
- Weather/Climate events: 20 entries
- Historical events: 10 entries
- General cases: 70 entries

**Complexity Distribution:**
- Simple cases: Single temporal/spatial entity
- Complex cases: Multiple entities, year inference, disambiguation

**Diversity Coverage:**
1. ✅ Simple extraction (single date + location)
2. ✅ Year inference (context-aware temporal resolution)
3. ✅ Multiple locations in single text
4. ✅ Multiple temporal expressions
5. ✅ Spatial disambiguation (same name, different places)
6. ✅ Natural disaster scenarios
7. ✅ Historical events
8. ✅ Time expressions (not just dates)
9. ✅ Location types (city, region, country)
10. ✅ Complex documents with mixed content

### 4. Supporting Files

**`eval/generate_dataset.py`:**
- Dataset generator with configurable size
- Diverse test case templates
- Automatic variation generation
- Statistics reporting

**`eval/quick_start.py`:**
- Quick evaluation script for testing
- Runs on sample subset (default 10 entries)
- User-friendly output

**`eval/README.md`:**
- Comprehensive documentation
- Usage examples
- Metric explanations
- Dataset format specification
- Customization guide

**`eval/__init__.py`:**
- Package initialization
- Clean API exports

## Usage Examples

### Quick Start (10 samples)
```bash
python eval/quick_start.py
```

### Full Evaluation (100 entries)
```bash
python eval/evaluation.py data/input/eval_dataset_100.json
```

### Custom Sample Size
```bash
python eval/quick_start.py 25
```

### Programmatic Usage
```python
from eval import run_evaluation

metrics = run_evaluation("data/input/eval_dataset_100.json")
print(f"Combined F1: {metrics['overall']['combined_f1']}")
```

## Output Structure

### Detailed Results (`data/output/eval_results/detailed_results_TIMESTAMP.json`)

Each entry contains:

```json
{
  "id": "entry_001",
  "text": "On March 15, 2022, a cyclone hit Broome, Western Australia.",
  "input_prompt": "Extract all temporal expressions...",
  "ground_truth": {
    "temporal": [...],
    "spatial": [...]
  },
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

### Metrics Summary (`data/output/eval_results/metrics_summary_TIMESTAMP.json`)

```json
{
  "temporal": {
    "precision": 0.95,
    "recall": 0.92,
    "f1_score": 0.935,
    "normalization_accuracy": 0.89,
    "type_accuracy": 0.91
  },
  "spatial": {
    "precision": 0.88,
    "recall": 0.85,
    "f1_score": 0.865,
    "geocoding_success_rate": 0.82,
    "mean_distance_error_km": 12.5,
    "accuracy_at_25km": 0.91
  },
  "overall": {
    "combined_f1": 0.90,
    "success_rate": 0.98,
    "average_processing_time_seconds": 2.8
  }
}
```

## Research Foundation

This evaluation framework is based on established research in:

1. **Named Entity Recognition (NER)**
   - CoNLL-2003 shared task
   - Entity-level (not token-level) evaluation

2. **Temporal Information Extraction**
   - TempEval-3 evaluation framework
   - TIMEX3 standard (ISO-TimeML)
   - ACE TERN evaluation

3. **Geoparsing and Geocoding**
   - Distance-based accuracy metrics
   - Standard 25km threshold
   - Type-aware evaluation

4. **Information Extraction**
   - SemEval'13 entity evaluation
   - Precision, Recall, F1 methodology

## Key Design Decisions

1. **Entity-Level Evaluation**: Following CoNLL-2003, we evaluate complete entities rather than individual tokens.

2. **Flexible Matching**: Support for exact, overlap-based, and normalized matching to handle LLM output variations.

3. **Distance-Based Spatial Accuracy**: Geographic accuracy measured in kilometers, following geoparsing standards.

4. **Comprehensive Output**: All inputs, outputs, and intermediate results saved for analysis.

5. **Standard Metrics**: Using established precision/recall/F1 rather than custom metrics for comparability.

## Integration with STIndex

The evaluation framework is fully integrated:

- ✅ Uses STIndexExtractor directly
- ✅ Respects all configuration options
- ✅ Compatible with all LLM providers
- ✅ Handles extraction errors gracefully
- ✅ Measures real-world performance

## Next Steps

To evaluate STIndex performance:

1. **Run Quick Test:**
   ```bash
   python eval/quick_start.py
   ```

2. **Review Sample Results:**
   - Check `data/output/eval_results/detailed_results_*.json`
   - Analyze which cases work well/poorly

3. **Run Full Evaluation:**
   ```bash
   python eval/evaluation.py data/input/eval_dataset_100.json
   ```

4. **Analyze Metrics:**
   - Review metrics summary
   - Identify areas for improvement
   - Compare across different LLM providers

5. **Iterate:**
   - Add more test cases to dataset
   - Fine-tune prompts based on results
   - Improve geocoding accuracy
   - Enhance normalization

## File Structure

```
STIndex/
├── eval/
│   ├── __init__.py              # Package initialization
│   ├── README.md                # Documentation
│   ├── metrics.py               # Evaluation metrics (350 lines)
│   ├── evaluation.py            # Evaluation runner (400 lines)
│   ├── generate_dataset.py      # Dataset generator (200 lines)
│   └── quick_start.py           # Quick evaluation script
├── data/
│   ├── input/
│   │   └── eval_dataset_100.json  # Evaluation dataset (100 entries)
│   └── output/
│       └── eval_results/          # Evaluation results (generated)
└── CLAUDE.md                      # Updated with evaluation docs
```

## Metrics Reference

### Temporal Metrics

| Metric | Description | Range |
|--------|-------------|-------|
| Precision | % of extracted temporal expressions that are correct | 0-1 |
| Recall | % of ground truth temporal expressions found | 0-1 |
| F1 Score | Harmonic mean of precision and recall | 0-1 |
| Normalization Accuracy | % correctly normalized to ISO 8601 | 0-1 |
| Type Accuracy | % correctly classified (date/time/duration/set) | 0-1 |

### Spatial Metrics

| Metric | Description | Range |
|--------|-------------|-------|
| Precision | % of extracted locations that are correct | 0-1 |
| Recall | % of ground truth locations found | 0-1 |
| F1 Score | Harmonic mean of precision and recall | 0-1 |
| Geocoding Success | % of locations successfully geocoded | 0-1 |
| Mean Distance Error | Average distance error in km | 0-∞ |
| Accuracy@25km | % of predictions within 25km of ground truth | 0-1 |

## Summary

✅ **Complete evaluation framework delivered**
✅ **Research-based metrics implemented**
✅ **100-entry dataset generated**
✅ **Comprehensive output format**
✅ **Fully documented**
✅ **Ready to use**

The evaluation framework provides all requested components:
- Input prompt ✓
- Text to be extracted ✓
- LLM raw output ✓
- Final output ✓
- Metrics results ✓
- Per-entry detailed results ✓
