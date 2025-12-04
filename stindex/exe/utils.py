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
    """Get output directory in data/output/yyyy-mm-dd/."""
    date_str = datetime.now().strftime("%Y-%m-%d")
    output_dir = Path(OUTPUT_DIR) / date_str
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def get_output_filename() -> str:
    """Get timestamped filename hh-mm-ss.json."""
    time_str = datetime.now().strftime("%H-%M-%S")
    return f"{time_str}.json"


def save_result(result, output_dir: Path, filename: str):
    """Save result to JSON file with all extraction data.

    Args:
        result: SpatioTemporalResult object
        output_dir: Directory to save the file
        filename: Filename (should include .json extension)
    """
    # Save JSON (with extraction config and raw LLM output)
    json_file = output_dir / filename

    # Handle entities - they might be dicts or Pydantic models
    temporal_entities = []
    for e in result.temporal_entities:
        if isinstance(e, dict):
            temporal_entities.append(e)
        else:
            temporal_entities.append(e.model_dump() if hasattr(e, 'model_dump') else e.dict())

    spatial_entities = []
    for e in result.spatial_entities:
        if isinstance(e, dict):
            spatial_entities.append(e)
        else:
            spatial_entities.append(e.model_dump() if hasattr(e, 'model_dump') else e.dict())

    result_dict = {
        "input_text": result.input_text,
        "temporal_entities": temporal_entities,
        "spatial_entities": spatial_entities,
        "success": result.success,
        "error": result.error,
        "processing_time": result.processing_time,
    }

    # Add extraction config if available
    # extraction_config can be a dict or Pydantic model
    if result.extraction_config:
        if isinstance(result.extraction_config, dict):
            result_dict["extraction_config"] = result.extraction_config
        else:
            result_dict["extraction_config"] = {
                "llm_provider": result.extraction_config.llm_provider,
                "model_name": result.extraction_config.model_name,
                "temperature": result.extraction_config.temperature,
                "max_tokens": result.extraction_config.max_tokens,
                "raw_llm_output": result.extraction_config.raw_llm_output,
            }

    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=2, ensure_ascii=False)

    return json_file


def display_json(result, output: Optional[Path] = None):
    """Display results as JSON."""
    # Handle entities - they might be dicts or Pydantic models
    temporal_entities = []
    for e in result.temporal_entities:
        if isinstance(e, dict):
            temporal_entities.append(e)
        else:
            temporal_entities.append(e.model_dump() if hasattr(e, 'model_dump') else e.dict())

    spatial_entities = []
    for e in result.spatial_entities:
        if isinstance(e, dict):
            spatial_entities.append(e)
        else:
            spatial_entities.append(e.model_dump() if hasattr(e, 'model_dump') else e.dict())

    result_dict = {
        "input_text": result.input_text,
        "temporal_entities": temporal_entities,
        "spatial_entities": spatial_entities,
        "success": result.success,
        "error": result.error,
        "processing_time": result.processing_time,
    }

    # Add extraction config if available
    # extraction_config can be a dict or Pydantic model
    if result.extraction_config:
        if isinstance(result.extraction_config, dict):
            result_dict["extraction_config"] = result.extraction_config
        else:
            result_dict["extraction_config"] = {
                "llm_provider": result.extraction_config.llm_provider,
                "model_name": result.extraction_config.model_name,
                "temperature": result.extraction_config.temperature,
                "max_tokens": result.extraction_config.max_tokens,
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
