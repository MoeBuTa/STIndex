"""
Efficiency evaluation pipeline for STIndex extraction.

Evaluates timing overhead, throughput, and latency from timing logs.
"""

from typing import Dict, List, Optional, Any
import json
import glob
import os
from pathlib import Path
from datetime import datetime
import pandas as pd
from loguru import logger

from stindex.eval.efficiency_metrics import (
    EfficiencyMetrics,
    load_timing_logs,
    compute_efficiency_metrics,
)


class EfficiencyEvaluator:
    """
    Evaluates efficiency metrics from timing logs.

    Loads timing logs from extraction results and computes:
    - Timing overhead
    - Throughput (chunks/sec, docs/sec)
    - Latency percentiles (P50, P95, P99)
    - GPU utilization (if available)
    """

    def __init__(
        self,
        timing_log_dir: str,
        output_dir: str = "data/output/evaluations/efficiency",
        baseline_log_dir: Optional[str] = None,
    ):
        """
        Initialize efficiency evaluator.

        Args:
            timing_log_dir: Directory containing timing logs (e.g., data/extraction_results_textbook_test)
            output_dir: Directory for evaluation outputs
            baseline_log_dir: Optional directory with baseline timing logs (without instrumentation)
        """
        self.timing_log_dir = Path(timing_log_dir)
        self.output_dir = Path(output_dir)
        self.baseline_log_dir = Path(baseline_log_dir) if baseline_log_dir else None

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def find_timing_files(self) -> Dict[str, Path]:
        """
        Find timing log files in directory.

        Returns:
            Dictionary with 'logs' and 'summary' file paths
        """
        # Find timing.jsonl files
        timing_files = list(self.timing_log_dir.glob("*_timing.jsonl"))

        # Find timing_summary.json files
        summary_files = list(self.timing_log_dir.glob("*_timing_summary.json"))

        if not timing_files:
            raise FileNotFoundError(
                f"No timing log files found in {self.timing_log_dir}"
            )

        # Use the most recent file
        timing_file = max(timing_files, key=os.path.getmtime)
        summary_file = None

        if summary_files:
            # Find corresponding summary file
            base_name = timing_file.stem.replace("_timing", "")
            for sf in summary_files:
                if base_name in sf.stem:
                    summary_file = sf
                    break

        logger.info(f"Found timing log: {timing_file}")
        if summary_file:
            logger.info(f"Found summary: {summary_file}")

        return {"logs": timing_file, "summary": summary_file}

    def load_timing_data(self, timing_file: Path) -> List[Dict[str, Any]]:
        """
        Load timing data from JSONL file.

        Args:
            timing_file: Path to timing log file

        Returns:
            List of timing records
        """
        timing_records = []

        with open(timing_file, "r") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    timing_records.append(record)

        logger.info(f"Loaded {len(timing_records)} timing records")
        return timing_records

    def load_summary_data(self, summary_file: Path) -> Dict[str, Any]:
        """
        Load timing summary data.

        Args:
            summary_file: Path to timing summary JSON

        Returns:
            Summary data dictionary
        """
        with open(summary_file, "r") as f:
            summary = json.load(f)

        logger.info(f"Loaded timing summary: {summary.get('name', 'unknown')}")
        return summary

    def compute_metrics(
        self,
        timing_records: List[Dict[str, Any]],
        summary_data: Optional[Dict[str, Any]] = None,
    ) -> EfficiencyMetrics:
        """
        Compute efficiency metrics from timing data.

        Args:
            timing_records: List of per-document timing records
            summary_data: Optional summary data with aggregate stats

        Returns:
            EfficiencyMetrics object
        """
        metrics = EfficiencyMetrics()

        # Extract per-document timing
        for record in timing_records:
            total_time = record.get("total_duration_seconds", 0)
            metrics.operation_times.append(total_time)

        # Use summary data for aggregate info
        if summary_data:
            timings = summary_data.get("timings", {})
            total = timings.get("total", {})

            metrics.total_chunks = total.get("count", len(timing_records))
            metrics.total_time = total.get("total_seconds", sum(metrics.operation_times))

        # If no summary, compute from records
        if not summary_data:
            metrics.total_chunks = len(timing_records)
            metrics.total_time = sum(metrics.operation_times)

        # Assume 1 document per timing record (could be multiple chunks per document)
        metrics.total_documents = len(timing_records)

        logger.info(f"Computed metrics for {metrics.total_chunks} chunks")
        logger.info(f"Total time: {metrics.total_time:.2f}s")

        return metrics

    def generate_detailed_report(
        self,
        timing_records: List[Dict[str, Any]],
        metrics: EfficiencyMetrics,
    ) -> pd.DataFrame:
        """
        Generate detailed per-document timing report.

        Args:
            timing_records: List of timing records
            metrics: Computed metrics

        Returns:
            DataFrame with per-document details
        """
        rows = []

        for i, record in enumerate(timing_records):
            doc_id = record.get("doc_id", f"doc_{i}")
            total_time = record.get("total_duration_seconds", 0)
            components = record.get("components", {})

            # Extract component timings
            extraction_time = components.get("extraction", {}).get("duration_seconds", 0)
            labeling_time = components.get("labeling", {}).get("duration_seconds", 0)

            # Postprocessing breakdown
            extraction_components = components.get("extraction", {}).get("components", {})
            postproc_time = extraction_components.get("postprocessing_seconds", 0)
            temporal_res_time = extraction_components.get("temporal_resolution_seconds", 0)

            rows.append({
                "doc_id": doc_id,
                "doc_index": record.get("doc_index", i),
                "timestamp": record.get("timestamp", ""),
                "total_duration_seconds": total_time,
                "extraction_seconds": extraction_time,
                "labeling_seconds": labeling_time,
                "postprocessing_seconds": postproc_time,
                "temporal_resolution_seconds": temporal_res_time,
            })

        df = pd.DataFrame(rows)
        logger.info(f"Generated detailed report with {len(df)} rows")

        return df

    def generate_summary_report(self, metrics: EfficiencyMetrics) -> Dict[str, Any]:
        """
        Generate summary report with aggregate metrics.

        Args:
            metrics: Computed efficiency metrics

        Returns:
            Dictionary with summary statistics
        """
        summary = {
            "evaluation_timestamp": datetime.now().isoformat(),
            "metrics": metrics.to_dict(),
            "targets": {
                "timing_overhead_percent": 0.5,
                "throughput_chunks_per_second": 10.0,
                "latency_p95_ms": 500.0,
            },
            "meets_targets": metrics.meets_targets(),
        }

        logger.info("Generated summary report")
        return summary

    def save_reports(
        self,
        detailed_df: pd.DataFrame,
        summary: Dict[str, Any],
        prefix: str = "efficiency_eval",
    ) -> Dict[str, Path]:
        """
        Save evaluation reports to disk.

        Args:
            detailed_df: Detailed per-document DataFrame
            summary: Summary dictionary
            prefix: Filename prefix

        Returns:
            Dictionary with output file paths
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save detailed CSV
        csv_path = self.output_dir / f"{prefix}_{timestamp}.csv"
        detailed_df.to_csv(csv_path, index=False)
        logger.info(f"Saved detailed report: {csv_path}")

        # Save summary JSON
        json_path = self.output_dir / f"{prefix}_{timestamp}_summary.json"
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved summary report: {json_path}")

        return {"csv": csv_path, "json": json_path}

    def evaluate(self) -> Dict[str, Any]:
        """
        Run full efficiency evaluation pipeline.

        Returns:
            Dictionary with evaluation results and file paths
        """
        logger.info("Starting efficiency evaluation")

        # Find timing files
        timing_files = self.find_timing_files()

        # Load timing data
        timing_records = self.load_timing_data(timing_files["logs"])

        summary_data = None
        if timing_files["summary"]:
            summary_data = self.load_summary_data(timing_files["summary"])

        # Compute metrics
        metrics = self.compute_metrics(timing_records, summary_data)

        # Generate reports
        detailed_df = self.generate_detailed_report(timing_records, metrics)
        summary = self.generate_summary_report(metrics)

        # Save reports
        output_files = self.save_reports(detailed_df, summary)

        logger.success("Efficiency evaluation complete!")

        # Print summary
        print("\n" + "=" * 60)
        print("EFFICIENCY EVALUATION SUMMARY")
        print("=" * 60)
        print(f"\nTiming Overhead: {metrics.timing_overhead():.4f}% (target: <0.5%)")
        print(f"Throughput: {metrics.throughput_chunks_per_second():.2f} chunks/sec (target: ≥10)")

        latency = metrics.latency_percentiles()
        print(f"\nLatency Percentiles:")
        print(f"  P50: {latency['p50']:.2f} ms")
        print(f"  P95: {latency['p95']:.2f} ms (target: <500ms)")
        print(f"  P99: {latency['p99']:.2f} ms")

        meets_targets = metrics.meets_targets()
        print(f"\nTargets Met:")
        print(f"  Timing Overhead: {'✓' if meets_targets['timing_overhead'] else '✗'}")
        print(f"  Throughput: {'✓' if meets_targets['throughput'] else '✗'}")
        print(f"  Latency P95: {'✓' if meets_targets['latency_p95'] else '✗'}")

        print(f"\nReports saved to:")
        print(f"  CSV: {output_files['csv']}")
        print(f"  JSON: {output_files['json']}")
        print("=" * 60 + "\n")

        return {
            "metrics": metrics.to_dict(),
            "meets_targets": meets_targets,
            "output_files": {k: str(v) for k, v in output_files.items()},
        }


def run_efficiency_evaluation(
    timing_log_dir: str = "data/extraction_results_textbook_test",
    output_dir: str = "data/output/evaluations/efficiency",
) -> Dict[str, Any]:
    """
    Run efficiency evaluation pipeline.

    Args:
        timing_log_dir: Directory with timing logs
        output_dir: Directory for outputs

    Returns:
        Dictionary with evaluation results
    """
    evaluator = EfficiencyEvaluator(
        timing_log_dir=timing_log_dir,
        output_dir=output_dir,
    )

    return evaluator.evaluate()


if __name__ == "__main__":
    # Run evaluation
    results = run_efficiency_evaluation()
