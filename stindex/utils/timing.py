"""Timing utilities for performance measurement."""
import time
import json
from typing import Dict, Any, Optional
from pathlib import Path
from loguru import logger


class TimingContext:
    """Context manager for timing code blocks."""

    def __init__(self, name: str, parent: Optional['TimingStats'] = None):
        self.name = name
        self.parent = parent
        self.start_time = None
        self.duration = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.duration = time.time() - self.start_time
        if self.parent:
            self.parent.add_timing(self.name, self.duration)
        return False


class TimingStats:
    """Collects and manages timing statistics."""

    def __init__(self, name: str = "pipeline", num_gpus: int = 1):
        self.name = name
        self.num_gpus = num_gpus
        self.start_time = time.time()
        self.timings: Dict[str, Any] = {}
        self.counters: Dict[str, int] = {}

    def timer(self, name: str) -> TimingContext:
        """Create a timing context manager."""
        return TimingContext(name, parent=self)

    def add_timing(self, name: str, duration: float):
        """Add a timing measurement."""
        if name not in self.timings:
            self.timings[name] = []
        self.timings[name].append(duration)

    def add_counter(self, name: str, value: int = 1):
        """Add to a counter."""
        self.counters[name] = self.counters.get(name, 0) + value

    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive timing statistics including GPU-hours."""
        total_duration = time.time() - self.start_time
        total_gpu_hours = (total_duration / 3600) * self.num_gpus

        summary = {
            "name": self.name,
            "num_gpus": self.num_gpus,
            "total_duration_seconds": round(total_duration, 3),
            "total_gpu_hours": round(total_gpu_hours, 4),
            "timings": {},
            "counters": self.counters
        }

        for name, durations in self.timings.items():
            total_secs = sum(durations)
            mean_secs = total_secs / len(durations)
            gpu_hours = (total_secs / 3600) * self.num_gpus

            summary["timings"][name] = {
                "count": len(durations),
                "total_seconds": round(total_secs, 3),
                "mean_seconds": round(mean_secs, 3),
                "min_seconds": round(min(durations), 3),
                "max_seconds": round(max(durations), 3),
                "total_gpu_hours": round(gpu_hours, 4)
            }

        return summary

    def save_json(self, output_path: Path):
        """Save timing statistics to JSON file."""
        summary = self.get_summary()
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"✓ Timing statistics saved to: {output_path}")

