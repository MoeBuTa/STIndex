"""
Shared utility functions for CLI execution commands.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.json import JSON
from rich.panel import Panel

from stindex.utils.constants import OUTPUT_DIR, PROJECT_DIR


console = Console()


def get_output_dir() -> Path:
    """Get timestamped output directory in data/output/."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(OUTPUT_DIR) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_result(result, output_dir: Path, filename: str):
    """Save result to both JSON and TXT in the output directory."""
    # Save JSON
    json_file = output_dir / f"{filename}.json"
    result_dict = {
        "temporal_entities": [e.dict() for e in result.temporal_entities],
        "spatial_entities": [e.dict() for e in result.spatial_entities],
        "success": result.success,
        "error": result.error,
        "processing_time": result.processing_time,
    }
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=2, ensure_ascii=False)

    # Save TXT summary
    txt_file = output_dir / f"{filename}.txt"
    temporal_count = len(result.temporal_entities)
    spatial_count = len(result.spatial_entities)

    with open(txt_file, "w", encoding="utf-8") as f:
        f.write(f"STIndex Extraction Results\n")
        f.write(f"{'=' * 80}\n\n")
        f.write(f"Temporal Entities: {temporal_count}\n")
        f.write(f"Spatial Entities: {spatial_count}\n")
        f.write(f"Processing Time: {result.processing_time:.2f}s\n\n")

        if result.temporal_entities:
            f.write(f"Temporal Entities:\n")
            for entity in result.temporal_entities:
                f.write(f"  • '{entity.text}' → {entity.normalized} [{entity.temporal_type.value}]\n")
            f.write("\n")

        if result.spatial_entities:
            f.write(f"Spatial Entities:\n")
            for entity in result.spatial_entities:
                f.write(f"  • '{entity.text}' → ({entity.latitude:.4f}, {entity.longitude:.4f})\n")

    return json_file, txt_file


def display_json(result, output: Optional[Path] = None):
    """Display results as JSON."""
    result_dict = {
        "temporal_entities": [e.dict() for e in result.temporal_entities],
        "spatial_entities": [e.dict() for e in result.spatial_entities],
        "success": result.success,
        "error": result.error,
        "processing_time": result.processing_time,
    }
    json_str = json.dumps(result_dict, indent=2, ensure_ascii=False, default=str)

    if output:
        # Save to file
        with open(output, "w", encoding="utf-8") as f:
            f.write(json_str)
        console.print(f"[green]Results saved to:[/green] {output}")
    else:
        # Display to console
        console.print(Panel(JSON(json_str), title="Extraction Results", border_style="blue"))
