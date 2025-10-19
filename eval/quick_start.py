#!/usr/bin/env python
"""
Quick start script for STIndex evaluation.

Runs evaluation on a small sample and displays results.
"""

import json
import sys
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.evaluation import run_evaluation, STIndexEvaluator
from stindex import STIndexExtractor


def run_sample_evaluation(num_samples: int = 10):
    """Run evaluation on a small sample"""
    print("=" * 80)
    print("STIndex Evaluation - Quick Start")
    print("=" * 80)
    print()

    # Load full dataset
    dataset_path = "data/input/eval_dataset_100.json"
    print(f"Loading dataset from {dataset_path}...")

    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    # Take sample
    sample_dataset = dataset[:num_samples]
    print(f"Running evaluation on {len(sample_dataset)} samples...")
    print()

    # Create temporary sample file
    sample_path = "data/input/eval_sample.json"
    with open(sample_path, 'w') as f:
        json.dump(sample_dataset, f, indent=2)

    # Run evaluation
    metrics = run_evaluation(sample_path, output_dir="data/output/eval_results")

    print()
    print("=" * 80)
    print("Evaluation Complete!")
    print("=" * 80)
    print()
    print("Results saved to: data/output/eval_results/")
    print()
    print("Next steps:")
    print("  1. Review detailed results in: data/output/eval_results/detailed_results_*.json")
    print("  2. Check metrics summary in: data/output/eval_results/metrics_summary_*.json")
    print("  3. Run full evaluation: python eval/evaluation.py data/input/eval_dataset_100.json")
    print()

    return metrics


if __name__ == "__main__":
    import sys

    num_samples = 10
    if len(sys.argv) > 1:
        num_samples = int(sys.argv[1])

    run_sample_evaluation(num_samples)
