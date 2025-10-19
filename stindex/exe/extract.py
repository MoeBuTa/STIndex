"""
Extract command execution - extracts spatiotemporal indices from text.
"""

import sys
from pathlib import Path
from typing import Optional

from rich.console import Console

from stindex import STIndexExtractor
from .utils import get_output_dir, save_result, display_json


console = Console()


def execute_extract(
    text: str,
    config: str = "extract",
    output: Optional[Path] = None,
):
    """Execute extraction from text string."""
    try:
        # Create extractor with config file
        extractor = STIndexExtractor(config_path=config)

        # Extract
        with console.status("[bold green]Extracting spatiotemporal indices..."):
            result = extractor.extract(text)

        if not result.success:
            console.print(f"[bold red]Extraction failed:[/bold red] {result.error}")
            sys.exit(1)

        # Display results as JSON
        display_json(result, output if output else None)

        # Auto-save to data/output/datetime/ unless custom output specified
        if not output:
            output_dir = get_output_dir()
            json_file, txt_file = save_result(result, output_dir, "extract_result")
            console.print(f"\n[green]✓ Auto-saved to:[/green] {output_dir}/")
            console.print(f"  • JSON: {json_file.name}")
            console.print(f"  • TXT:  {txt_file.name}")

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
