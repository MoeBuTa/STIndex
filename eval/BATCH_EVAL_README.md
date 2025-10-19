# Batch Evaluation with HuggingFace Models

This directory contains batch evaluation scripts optimized for HuggingFace models with multi-GPU support via Accelerate and DeepSpeed.

## Overview

**Features:**
- Batch processing for efficient GPU utilization
- Multi-GPU distributed evaluation with DeepSpeed ZeRO-2
- Resume capability with checkpointing
- Compatible with standard evaluation metrics

## Installation

Install with HuggingFace and Accelerate support:

```bash
pip install -e ".[transformers]"
```

This installs:
- `transformers>=4.30.0`
- `torch>=2.0.0`
- `accelerate>=0.25.0`
- `deepspeed>=0.14.0`
- `tqdm>=4.65.0`

## Usage

### Single GPU Batch Evaluation

For single GPU with batching:

```bash
python eval/batch_evaluation.py data/input/eval_dataset_100.json \
    --model Qwen/Qwen2.5-7B-Instruct \
    --batch-size 8 \
    --torch-dtype bfloat16
```

**Arguments:**
- `dataset_path`: Path to evaluation dataset JSON (required)
- `--model`: HuggingFace model ID (default: Qwen/Qwen2.5-7B-Instruct)
- `--batch-size`: Number of samples per batch (default: 8)
- `--output-dir`: Output directory (default: data/output/eval_results)
- `--device`: Device placement (default: auto)
- `--torch-dtype`: Torch dtype (default: bfloat16)

### Multi-GPU Distributed Evaluation

For multi-GPU with DeepSpeed ZeRO-2:

```bash
accelerate launch --config_file cfg/deepspeed_zero2.yaml \
    eval/batch_evaluation_accelerate.py data/input/eval_dataset_100.json \
    --model Qwen/Qwen2.5-7B-Instruct \
    --batch-size 8 \
    --torch-dtype bfloat16
```

**Arguments:** Same as single GPU, but `--batch-size` is per GPU.

**DeepSpeed Configuration** (`cfg/deepspeed_zero2.yaml`):
```yaml
compute_environment: LOCAL_MACHINE
distributed_type: DEEPSPEED
deepspeed_config:
  zero_stage: 2                    # ZeRO-2: optimizer + gradient partitioning
  offload_optimizer_device: none   # No CPU offloading
  offload_param_device: none
  zero3_init_flag: false
mixed_precision: bf16              # BFloat16 for efficiency
num_processes: 2                   # Number of GPUs
```

## Architecture

### Batch Processing Flow

```
batch_evaluation.py (Single GPU)
    │
    ├─► Load dataset
    ├─► Check for existing results (resume)
    ├─► Process in batches:
    │   ├─► Build message batch
    │   ├─► HuggingFaceLLM.generate_batch()
    │   │   ├─► Format prompts + schema
    │   │   ├─► Tokenize with padding
    │   │   ├─► model.generate() (batched)
    │   │   └─► Extract & validate JSON
    │   ├─► Post-process (char positions)
    │   └─► Geocode locations
    └─► Save results + metrics
```

### Distributed Flow (Accelerate)

```
batch_evaluation_accelerate.py (Multi-GPU)
    │
    ├─► Initialize Accelerator
    ├─► Load model on each GPU
    ├─► Create DataLoader
    ├─► Accelerator.prepare(dataloader)  # Distributes batches
    │
    ├─► Each GPU processes its batches:
    │   ├─► HuggingFaceLLM.generate_batch()
    │   ├─► Post-process + geocode
    │   └─► Save individual results
    │
    ├─► Gather results from all GPUs
    └─► Main process: calculate metrics + save
```

## Implementation Details

### HuggingFaceLLM.generate_batch()

Added to `stindex/agents/llm/providers/huggingface_llm.py`:

```python
def generate_batch(
    self,
    messages_batch: List[List[Dict[str, str]]],
    response_model: type[T],
    max_tokens: int = 2048,
    temperature: Optional[float] = None,
) -> List[T]:
    """Batch extraction with left padding for decoder-only models"""
```

**Key features:**
- Left padding for decoder-only models (GPT, Llama, Qwen)
- Single forward pass for entire batch
- Removes prompt tokens from each output
- JSON extraction + Pydantic validation per sample

### Resume Capability

**Single GPU (`batch_evaluation.py`):**
- Saves `checkpoint.json` after each batch
- On restart, loads checkpoint and skips completed samples

**Multi-GPU (`batch_evaluation_accelerate.py`):**
- Saves individual `result_{id}.json` files immediately
- On restart, loads all individual files and skips completed samples
- More fault-tolerant for long-running evaluations

## Performance Considerations

**Batch Size Recommendations:**

| GPU Memory | Batch Size | Model Size |
|------------|------------|------------|
| 24GB       | 4-8        | 7B params  |
| 40GB       | 8-16       | 7B params  |
| 80GB       | 16-32      | 7B params  |

**Multi-GPU Scaling:**
- Effective batch size = `batch_size × num_processes`
- Example: `--batch-size 8` on 2 GPUs = 16 samples per iteration
- DeepSpeed ZeRO-2 reduces memory per GPU (enables larger batches)

**DeepSpeed ZeRO Stages:**
- **ZeRO-2**: Recommended for inference (partitions optimizer state + gradients)
- **ZeRO-3**: For very large models (partitions all parameters, slower)

## Output Files

**Single GPU:**
- `checkpoint.json`: Resume checkpoint with all results
- `batch_detailed_results_{timestamp}.json`: Final detailed results
- `batch_metrics_summary_{timestamp}.json`: Metrics summary

**Multi-GPU:**
- `individual_results/result_{id}.json`: Per-sample results (for resume)
- `accelerate_detailed_results_{timestamp}.json`: Aggregated results
- `accelerate_metrics_summary_{timestamp}.json`: Metrics summary

## Metrics

Same metrics as standard evaluation:

**Temporal:**
- Precision, Recall, F1
- Normalization accuracy (ISO 8601)
- Type classification accuracy

**Spatial:**
- Precision, Recall, F1
- Geocoding success rate
- Distance error (mean/median)
- Accuracy@25km
- Type classification accuracy

**Overall:**
- Combined F1 (macro-average)
- Success rate
- Average processing time

## Troubleshooting

**Out of Memory (OOM):**
- Reduce `--batch-size`
- Use DeepSpeed ZeRO-2: `accelerate launch --config_file cfg/deepspeed_zero2.yaml`
- Use smaller model or quantization

**Slow Geocoding:**
- Geocoding is sequential (1 req/sec rate limit)
- Consider caching: `~/.stindex/geocode_cache/`
- Disable geocoding for speed tests (edit code)

**Resume Not Working:**
- Check `checkpoint.json` or `individual_results/` exists
- Ensure sample IDs are consistent
- Delete corrupted checkpoint files to restart

## Examples

**Quick test (10 samples, single GPU):**
```bash
python eval/quick_start.py
```

**Full evaluation (100 samples, single GPU):**
```bash
python eval/batch_evaluation.py data/input/eval_dataset_100.json --batch-size 8
```

**Multi-GPU evaluation (2 GPUs, batch_size=8 per GPU = 16 total):**
```bash
accelerate launch --config_file cfg/deepspeed_zero2.yaml \
    eval/batch_evaluation_accelerate.py data/input/eval_dataset_100.json \
    --batch-size 8
```

**Custom model:**
```bash
python eval/batch_evaluation.py data/input/eval_dataset_100.json \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --batch-size 4 \
    --torch-dtype float16
```
