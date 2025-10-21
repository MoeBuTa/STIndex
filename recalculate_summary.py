#!/usr/bin/env python3
"""
Recalculate evaluation summary metrics with the fixed calculation method.
"""

import json
from pathlib import Path

from stindex.exe.evaluate import calculate_cumulative_metrics

# Path to the CSV file
csv_path = Path("data/output/evaluations/eval_dataset_100-gpt-4o-mini/eval_20251021_121714_openai_default.csv")

print(f"Recalculating metrics from: {csv_path}")

# Calculate with correct matching (exact mode is case-insensitive)
cumulative_metrics = calculate_cumulative_metrics(
    csv_path,
    temporal_match_mode="exact",  # Case-insensitive text matching
    spatial_match_mode="exact"    # Case-insensitive text matching
)

# Save new summary
summary_path = csv_path.with_suffix(".summary.json")
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(cumulative_metrics.to_dict(), f, indent=2)

print(f"\nNew summary saved to: {summary_path}")
print("\nCorrected Metrics:")
print(json.dumps(cumulative_metrics.to_dict(), indent=2))
