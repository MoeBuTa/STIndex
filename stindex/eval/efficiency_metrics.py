"""
Efficiency metrics for STIndex extraction pipeline.

Evaluates timing overhead, throughput, GPU utilization, and latency.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import numpy as np
from datetime import datetime
import json


@dataclass
class EfficiencyMetrics:
    """Metrics for efficiency evaluation of extraction pipeline"""

    # Timing data
    operation_times: List[float] = field(default_factory=list)
    baseline_times: List[float] = field(default_factory=list)

    # Throughput data
    total_chunks: int = 0
    total_documents: int = 0
    total_time: float = 0.0

    # GPU utilization (if available)
    gpu_memory_used: List[float] = field(default_factory=list)
    gpu_compute_util: List[float] = field(default_factory=list)

    def timing_overhead(self) -> float:
        """
        Calculate timing overhead as percentage.

        Overhead = (instrumented_time - baseline_time) / baseline_time * 100

        Target: <0.5%

        Returns:
            Overhead percentage
        """
        if not self.operation_times or not self.baseline_times:
            return 0.0

        instrumented = sum(self.operation_times)
        baseline = sum(self.baseline_times)

        if baseline == 0:
            return 0.0

        overhead = ((instrumented - baseline) / baseline) * 100
        return max(0.0, overhead)  # Ensure non-negative

    def throughput_chunks_per_second(self) -> float:
        """
        Calculate throughput in chunks per second.

        Target: ≥10 chunks/second

        Returns:
            Chunks processed per second
        """
        if self.total_time == 0:
            return 0.0
        return self.total_chunks / self.total_time

    def throughput_documents_per_second(self) -> float:
        """
        Calculate throughput in documents per second.

        Returns:
            Documents processed per second
        """
        if self.total_time == 0:
            return 0.0
        return self.total_documents / self.total_time

    def latency_percentiles(self) -> Dict[str, float]:
        """
        Calculate latency percentiles (P50, P95, P99).

        Target P95: <500ms per chunk

        Returns:
            Dictionary with percentile values in milliseconds
        """
        if not self.operation_times:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0}

        times_ms = [t * 1000 for t in self.operation_times]  # Convert to ms

        return {
            "p50": float(np.percentile(times_ms, 50)),
            "p95": float(np.percentile(times_ms, 95)),
            "p99": float(np.percentile(times_ms, 99)),
            "mean": float(np.mean(times_ms)),
            "std": float(np.std(times_ms)),
        }

    def gpu_utilization(self) -> Dict[str, float]:
        """
        Calculate GPU utilization statistics.

        Returns:
            Dictionary with GPU memory and compute utilization
        """
        if not self.gpu_memory_used and not self.gpu_compute_util:
            return {"memory_mean": 0.0, "compute_mean": 0.0}

        result = {}

        if self.gpu_memory_used:
            result.update({
                "memory_mean": float(np.mean(self.gpu_memory_used)),
                "memory_max": float(np.max(self.gpu_memory_used)),
                "memory_std": float(np.std(self.gpu_memory_used)),
            })

        if self.gpu_compute_util:
            result.update({
                "compute_mean": float(np.mean(self.gpu_compute_util)),
                "compute_max": float(np.max(self.gpu_compute_util)),
                "compute_std": float(np.std(self.gpu_compute_util)),
            })

        return result

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for reporting"""
        return {
            "timing_overhead_percent": round(self.timing_overhead(), 4),
            "throughput": {
                "chunks_per_second": round(self.throughput_chunks_per_second(), 2),
                "documents_per_second": round(self.throughput_documents_per_second(), 4),
            },
            "latency": self.latency_percentiles(),
            "gpu_utilization": self.gpu_utilization(),
            "totals": {
                "chunks": self.total_chunks,
                "documents": self.total_documents,
                "total_time_seconds": round(self.total_time, 2),
            },
        }

    def meets_targets(self) -> Dict[str, bool]:
        """
        Check if efficiency metrics meet target thresholds.

        Targets:
        - Timing overhead: <0.5%
        - Throughput: ≥10 chunks/second
        - Latency P95: <500ms

        Returns:
            Dictionary indicating which targets are met
        """
        latency = self.latency_percentiles()

        return {
            "timing_overhead": self.timing_overhead() < 0.5,
            "throughput": self.throughput_chunks_per_second() >= 10.0,
            "latency_p95": latency.get("p95", float("inf")) < 500.0,
        }


def load_timing_logs(timing_log_path: str) -> List[Dict[str, Any]]:
    """
    Load timing logs from JSONL file.

    Args:
        timing_log_path: Path to timing log file (*.jsonl)

    Returns:
        List of timing records
    """
    timing_records = []

    with open(timing_log_path, "r") as f:
        for line in f:
            if line.strip():
                timing_records.append(json.loads(line))

    return timing_records


def compute_efficiency_metrics(timing_records: List[Dict[str, Any]]) -> EfficiencyMetrics:
    """
    Compute efficiency metrics from timing logs.

    Args:
        timing_records: List of timing records from timing logs

    Returns:
        EfficiencyMetrics object with computed metrics
    """
    metrics = EfficiencyMetrics()

    # Extract timing data
    for record in timing_records:
        # Operation times (extraction, postprocessing, etc.)
        if "duration" in record:
            metrics.operation_times.append(record["duration"])

        # Count chunks and documents
        if "operation" in record:
            if "extraction" in record["operation"].lower():
                metrics.total_chunks += 1

    # Calculate total time
    if metrics.operation_times:
        metrics.total_time = sum(metrics.operation_times)

    # Extract document count from metadata (if available)
    if timing_records and "metadata" in timing_records[-1]:
        metadata = timing_records[-1]["metadata"]
        if "total_documents" in metadata:
            metrics.total_documents = metadata["total_documents"]
        if "total_chunks" in metadata:
            metrics.total_chunks = metadata["total_chunks"]

    return metrics


def compute_baseline_metrics(baseline_times: List[float], num_chunks: int) -> EfficiencyMetrics:
    """
    Compute baseline efficiency metrics (without timing instrumentation).

    Args:
        baseline_times: List of operation times without instrumentation
        num_chunks: Number of chunks processed

    Returns:
        EfficiencyMetrics object with baseline data
    """
    metrics = EfficiencyMetrics()
    metrics.baseline_times = baseline_times
    metrics.total_chunks = num_chunks
    metrics.total_time = sum(baseline_times) if baseline_times else 0.0

    return metrics
