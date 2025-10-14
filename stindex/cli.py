"""
Command-line interface for STIndex.
"""

import json
import os
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.json import JSON
from rich.panel import Panel
from rich.table import Table

from stindex import __version__
from stindex.core.extractor import STIndexExtractor
from stindex.models.schemas import ExtractionConfig

app = typer.Typer(
    name="stindex",
    help="STIndex: Spatiotemporal Index Extraction from Unstructured Text",
    add_completion=False,
)
console = Console()

# Default model from environment variable or fallback to local Qwen3-8B
DEFAULT_MODEL = os.getenv("STINDEX_MODEL_NAME", "Qwen/Qwen3-8B")


@app.command()
def version():
    """Show version information."""
    console.print(f"[bold blue]STIndex[/bold blue] version {__version__}")


@app.command()
def extract(
    text: str = typer.Argument(..., help="Text to extract spatiotemporal indices from"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file path"),
    format: str = typer.Option("json", "--format", "-f", help="Output format (json/table)"),
    model: str = typer.Option(DEFAULT_MODEL, "--model", "-m", help="LLM model to use"),
    temporal_only: bool = typer.Option(False, "--temporal-only", help="Extract only temporal"),
    spatial_only: bool = typer.Option(False, "--spatial-only", help="Extract only spatial"),
    no_geocoding: bool = typer.Option(False, "--no-geocoding", help="Disable geocoding"),
):
    """Extract spatiotemporal indices from text."""
    try:
        # Create extractor
        config = ExtractionConfig(
            model_name=model,
            enable_temporal=not spatial_only,
            enable_spatial=not temporal_only,
        )

        if no_geocoding:
            config.enable_spatial = False

        extractor = STIndexExtractor(config=config)

        # Extract
        with console.status("[bold green]Extracting spatiotemporal indices..."):
            result = extractor.extract(text)

        # Display results
        if format == "table":
            _display_table(result)
        else:
            _display_json(result, output)

        console.print(
            f"\n[dim]Extracted {result.temporal_count} temporal and "
            f"{result.spatial_count} spatial entities in "
            f"{result.processing_time:.2f}s[/dim]"
        )

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        sys.exit(1)


@app.command()
def extract_file(
    file_path: Path = typer.Argument(..., help="Path to input text file"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file path"),
    format: str = typer.Option("json", "--format", "-f", help="Output format (json/table)"),
    model: str = typer.Option(DEFAULT_MODEL, "--model", "-m", help="LLM model to use"),
):
    """Extract spatiotemporal indices from a file."""
    if not file_path.exists():
        console.print(f"[bold red]Error:[/bold red] File not found: {file_path}")
        sys.exit(1)

    try:
        # Read file
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        console.print(f"[dim]Processing file: {file_path}[/dim]")

        # Create extractor
        config = ExtractionConfig(model_name=model)
        extractor = STIndexExtractor(config=config)

        # Extract
        with console.status("[bold green]Extracting spatiotemporal indices..."):
            result = extractor.extract(text)

        # Display results
        if format == "table":
            _display_table(result)
        else:
            _display_json(result, output)

        console.print(
            f"\n[dim]Extracted {result.temporal_count} temporal and "
            f"{result.spatial_count} spatial entities in "
            f"{result.processing_time:.2f}s[/dim]"
        )

    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        sys.exit(1)


@app.command()
def batch(
    input_dir: Path = typer.Argument(..., help="Input directory with text files"),
    output_dir: Path = typer.Argument(..., help="Output directory for results"),
    pattern: str = typer.Option("*.txt", "--pattern", "-p", help="File pattern to match"),
    model: str = typer.Option(DEFAULT_MODEL, "--model", "-m", help="LLM model to use"),
):
    """Process multiple files in batch."""
    if not input_dir.exists():
        console.print(f"[bold red]Error:[/bold red] Directory not found: {input_dir}")
        sys.exit(1)

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find files
    files = list(input_dir.glob(pattern))

    if not files:
        console.print(f"[yellow]Warning:[/yellow] No files found matching pattern: {pattern}")
        sys.exit(0)

    console.print(f"[dim]Found {len(files)} files to process[/dim]")

    # Create extractor
    config = ExtractionConfig(model_name=model)
    extractor = STIndexExtractor(config=config)

    # Process files
    for file_path in files:
        try:
            console.print(f"\n[cyan]Processing:[/cyan] {file_path.name}")

            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

            result = extractor.extract(text)

            # Save result
            output_file = output_dir / f"{file_path.stem}_stindex.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)

            console.print(
                f"  [green]✓[/green] {result.temporal_count} temporal, "
                f"{result.spatial_count} spatial → {output_file.name}"
            )

        except Exception as e:
            console.print(f"  [red]✗[/red] Error: {str(e)}")

    console.print(f"\n[bold green]Batch processing complete![/bold green]")


def _display_table(result):
    """Display results as a table."""
    # Temporal entities table
    if result.temporal_entities:
        temp_table = Table(title="Temporal Entities", show_header=True, header_style="bold cyan")
        temp_table.add_column("Text", style="yellow")
        temp_table.add_column("Normalized", style="green")
        temp_table.add_column("Type", style="magenta")
        temp_table.add_column("Confidence", justify="right")

        for entity in result.temporal_entities:
            temp_table.add_row(
                entity.text,
                entity.normalized,
                entity.temporal_type.value,
                f"{entity.confidence:.2f}",
            )

        console.print(temp_table)

    # Spatial entities table
    if result.spatial_entities:
        spatial_table = Table(title="Spatial Entities", show_header=True, header_style="bold cyan")
        spatial_table.add_column("Text", style="yellow")
        spatial_table.add_column("Coordinates", style="green")
        spatial_table.add_column("Location", style="blue")
        spatial_table.add_column("Confidence", justify="right")

        for entity in result.spatial_entities:
            coords = f"{entity.latitude:.4f}, {entity.longitude:.4f}"
            location = entity.locality or entity.admin_area or entity.country or "N/A"
            spatial_table.add_row(entity.text, coords, location, f"{entity.confidence:.2f}")

        console.print(spatial_table)


def _display_json(result, output: Optional[Path] = None):
    """Display results as JSON."""
    result_dict = result.to_dict()
    json_str = json.dumps(result_dict, indent=2, ensure_ascii=False)

    if output:
        # Save to file
        with open(output, "w", encoding="utf-8") as f:
            f.write(json_str)
        console.print(f"[green]Results saved to:[/green] {output}")
    else:
        # Display to console
        console.print(Panel(JSON(json_str), title="Extraction Results", border_style="blue"))


def main():
    """Main entry point for CLI."""
    app()


if __name__ == "__main__":
    main()
