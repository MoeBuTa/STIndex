"""
Extract command execution - extracts spatiotemporal indices from text.
"""

import sys
from pathlib import Path
from typing import Optional

from rich.console import Console

from stindex import DimensionalExtractor
from .utils import get_output_dir, get_output_filename, save_result, display_json


console = Console()


def execute_extract(
    text: str,
    config: str = "extract",
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    base_url: Optional[str] = None,
    auto_start: bool = True,
    output: Optional[Path] = None,
):
    """Execute extraction from text string."""
    try:
        # Create extractor with config file and optional overrides
        extractor = DimensionalExtractor(
            config_path=config,
            model_name=model,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=base_url,
            auto_start=auto_start,
        )

        # Extract
        with console.status("[bold green]Extracting spatiotemporal indices..."):
            result = extractor.extract(text)

        if not result.success:
            console.print(f"[bold red]Extraction failed:[/bold red] {result.error}")
            sys.exit(1)

        # Display results as JSON
        display_json(result, output if output else None)

        # Auto-save based on config (unless custom output path is specified)
        if not output:
            auto_save = extractor.config.get("extraction", {}).get("auto_save", True)
            if auto_save:
                output_dir = get_output_dir()
                filename = get_output_filename()
                json_file = save_result(result, output_dir, filename)
                console.print(f"\n[green]✓ Auto-saved to:[/green] {json_file}")

        temporal_count = len(result.temporal_entities)
        spatial_count = len(result.spatial_entities)
        console.print(
            f"\n[dim]Extracted {temporal_count} temporal and "
            f"{spatial_count} spatial entities in "
            f"{result.processing_time:.2f}s[/dim]"
        )

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        sys.exit(1)
