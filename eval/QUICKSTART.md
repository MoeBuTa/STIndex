# Quick Evaluation Guide

## Status

✅ **Evaluation framework is fully functional**
✅ **Works with OpenAI and Anthropic providers**
⚠️ **HuggingFace provider has a known issue** (being fixed)

## Recommended Usage

### Use OpenAI Provider (Recommended)

```bash
# Make sure OPENAI_API_KEY is set
export OPENAI_API_KEY="your-key-here"

# Run evaluation (uses OpenAI by default)
python eval/quick_start.py 10
```

### Use Anthropic Provider

```bash
# Make sure ANTHROPIC_API_KEY is set
export ANTHROPIC_API_KEY="your-key-here"

# Edit cfg/extract.yml to set:
# llm_provider: anthropic

# Run evaluation
python eval/quick_start.py 10
```

## Quick Start Examples

### Test on 3 Samples (Fast)
```bash
python eval/quick_start.py 3
```

**Expected Output:**
- Processing time: ~10-15 seconds
- Success rate: ~100%
- Combined F1: ~0.95-1.00

### Test on 10 Samples (Default)
```bash
python eval/quick_start.py
```

**Expected Output:**
- Processing time: ~30-40 seconds
- Success rate: ~100%
- Detailed metrics for temporal and spatial extraction

### Full Evaluation (100 Samples)
```bash
python eval/evaluation.py data/input/eval_dataset_100.json
```

**Expected Output:**
- Processing time: ~5-8 minutes
- Comprehensive metrics
- Detailed results saved to `data/output/eval_results/`

## Successful Run Example

```
================================================================================
STIndex Evaluation - Quick Start
================================================================================

Loading dataset from data/input/eval_dataset_100.json...
Running evaluation on 3 samples...

Evaluating... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% 0:00:11

Evaluation Results

        Overall Metrics
┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃ Metric              ┃ Value  ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ Combined F1 Score   │ 1.0000 │
│ Success Rate        │ 1.0000 │
│ Avg Processing Time │ 3.80s  │
│ Total Documents     │ 3      │
└─────────────────────┴────────┘

    Temporal Extraction Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃ Metric                 ┃ Value  ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ Precision              │ 1.0000 │
│ Recall                 │ 1.0000 │
│ F1 Score               │ 1.0000 │
│ Normalization Accuracy │ 1.0000 │
│ Type Accuracy          │ 0.0000 │
└────────────────────────┴────────┘

    Spatial Extraction Metrics
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃ Metric                     ┃ Value  ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ Precision                  │ 1.0000 │
│ Recall                     │ 1.0000 │
│ F1 Score                   │ 1.0000 │
│ Geocoding Success Rate     │ 1.0000 │
│ Mean Distance Error        │ 0.00 km│
│ Median Distance Error      │ 0.00 km│
│ Accuracy @ 25km            │ 1.0000 │
│ Type Accuracy              │ 0.0000 │
└────────────────────────────┴────────┘
```

## Viewing Results

After evaluation completes, check:

```bash
# View latest detailed results
ls -lt data/output/eval_results/detailed_results_*.json | head -1

# View latest metrics summary
ls -lt data/output/eval_results/metrics_summary_*.json | head -1

# Pretty print summary
python -m json.tool data/output/eval_results/metrics_summary_*.json
```

## Troubleshooting

### Issue: HuggingFace Provider Fails

**Error:** `'function' object has no attribute 'create'`

**Solution:** Use OpenAI or Anthropic provider instead:
```bash
# Set provider in cfg/extract.yml
llm_provider: openai  # or anthropic
```

### Issue: API Key Not Set

**Error:** `AuthenticationError` or `API key not found`

**Solution:**
```bash
export OPENAI_API_KEY="your-key-here"
# or
export ANTHROPIC_API_KEY="your-key-here"
```

### Issue: Slow Evaluation

**Cause:** Geocoding API rate limits

**Solution:** The geocoder has built-in rate limiting (1 req/sec) and caching. Subsequent runs will be faster due to cache.

## Known Limitations

1. **Type Accuracy = 0.0000**: This is expected because the current implementation doesn't include temporal/spatial type in the ground truth matching. This will be fixed in a future update.

2. **HuggingFace Provider**: Currently not working due to a bug in the LLM client. Use OpenAI or Anthropic instead.

3. **Geocoding Rate Limits**: Nominatim has a 1 request/second rate limit. Large evaluations take time but results are cached.

## Next Steps

1. Run a small test: `python eval/quick_start.py 3`
2. Review the output in `data/output/eval_results/`
3. Run full evaluation when ready
4. Analyze metrics to identify improvements
