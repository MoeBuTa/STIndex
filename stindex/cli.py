"""
Command-line interface for STIndex.

Uses the new agentic ExtractionPipeline with observe-reason-act pattern.
"""

import json
import os
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

import typer
from rich.console import Console
from rich.json import JSON
from rich.panel import Panel
from rich.table import Table

from stindex import __version__, ExtractionPipeline

app = typer.Typer(
    name="stindex",
    help="STIndex: Spatiotemporal Index Extraction from Unstructured Text",
    add_completion=False,
)
console = Console()

# Default model from environment variable or fallback to local Qwen3-8B
DEFAULT_MODEL = os.getenv("STINDEX_MODEL_NAME", "Qwen/Qwen3-8B")


def _get_output_dir() -> Path:
    """Get timestamped output directory in data/output/."""
    # Get project root (where stindex package is installed)
    # This works whether installed or running from source
    try:
        from stindex import __file__ as stindex_file
        project_root = Path(stindex_file).parent.parent
    except:
        project_root = Path.cwd()

    # Create timestamped directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = project_root / "data" / "output" / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir


def _save_result(result, output_dir: Path, filename: str):
    """Save result to both JSON and TXT in the output directory."""
    # Save JSON
    json_file = output_dir / f"{filename}.json"
    result_dict = {
        "text": result.text,
        "temporal_entities": result.temporal_entities,
        "spatial_entities": result.spatial_entities,
        "success": result.success,
        "error": result.error,
        "processing_time": result.processing_time,
        "metadata": result.metadata,
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
                f.write(f"  • '{entity['text']}' → {entity['normalized']} [{entity['type']}]\n")
            f.write("\n")

        if result.spatial_entities:
            f.write(f"Spatial Entities:\n")
            for entity in result.spatial_entities:
                f.write(f"  • '{entity['text']}' → ({entity['latitude']:.4f}, {entity['longitude']:.4f})\n")

    return json_file, txt_file


@app.command()
def version():
    """Show version information."""
    console.print(f"[bold blue]STIndex[/bold blue] version {__version__}")


@app.command()
def extract(
    text: str = typer.Argument(..., help="Text to extract spatiotemporal indices from"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Custom output file path (overrides auto-save)"),
    format: str = typer.Option("json", "--format", "-f", help="Output format (json/table)"),
    model: str = typer.Option(DEFAULT_MODEL, "--model", "-m", help="LLM model to use"),
    no_save: bool = typer.Option(False, "--no-save", help="Don't auto-save to data/output/"),
):
    """Extract spatiotemporal indices from text."""
    try:
        # Create pipeline
        config = {
            "model_name": model,
        }
        pipeline = ExtractionPipeline(config=config)

        # Extract
        with console.status("[bold green]Extracting spatiotemporal indices..."):
            result = pipeline.extract(text)

        if not result.success:
            console.print(f"[bold red]Extraction failed:[/bold red] {result.error}")
            sys.exit(1)

        # Display results
        if format == "table":
            _display_table(result)
        else:
            _display_json(result, output if output else None)

        # Auto-save to data/output/datetime/ unless disabled or custom output specified
        if not no_save and not output:
            output_dir = _get_output_dir()
            json_file, txt_file = _save_result(result, output_dir, "extract_result")
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


@app.command()
def extract_file(
    file_path: Path = typer.Argument(..., help="Path to input text file"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Custom output file path (overrides auto-save)"),
    format: str = typer.Option("json", "--format", "-f", help="Output format (json/table)"),
    model: str = typer.Option(DEFAULT_MODEL, "--model", "-m", help="LLM model to use"),
    no_save: bool = typer.Option(False, "--no-save", help="Don't auto-save to data/output/"),
):
    """Extract spatiotemporal indices from a file."""
    if not file_path.exists():
        console.print(f"[bold red]Error:[/bold red] File not found: {file_path}")
        sys.exit(1)

    try:
        console.print(f"[dim]Processing file: {file_path}[/dim]")

        # Create pipeline
        config = {"model_name": model}
        pipeline = ExtractionPipeline(config=config)

        # Extract from file
        with console.status("[bold green]Extracting spatiotemporal indices..."):
            result = pipeline.extract_from_file(str(file_path))

        if not result.success:
            console.print(f"[bold red]Extraction failed:[/bold red] {result.error}")
            sys.exit(1)

        # Display results
        if format == "table":
            _display_table(result)
        else:
            _display_json(result, output if output else None)

        # Auto-save to data/output/datetime/ unless disabled or custom output specified
        if not no_save and not output:
            output_dir = _get_output_dir()
            filename = f"{file_path.stem}_result"
            json_file, txt_file = _save_result(result, output_dir, filename)
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


@app.command()
def batch(
    input_dir: Path = typer.Argument(..., help="Input directory with text files"),
    output_dir: Optional[Path] = typer.Argument(None, help="Custom output directory (optional, uses data/output/datetime/ by default)"),
    pattern: str = typer.Option("*.txt", "--pattern", "-p", help="File pattern to match"),
    model: str = typer.Option(DEFAULT_MODEL, "--model", "-m", help="LLM model to use"),
):
    """Process multiple files in batch."""
    if not input_dir.exists():
        console.print(f"[bold red]Error:[/bold red] Directory not found: {input_dir}")
        sys.exit(1)

    # Use custom output dir or create timestamped one
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = _get_output_dir()
        console.print(f"[dim]Auto-saving to: {output_dir}[/dim]")

    # Find files
    files = list(input_dir.glob(pattern))

    if not files:
        console.print(f"[yellow]Warning:[/yellow] No files found matching pattern: {pattern}")
        sys.exit(0)

    console.print(f"[dim]Found {len(files)} files to process[/dim]\n")

    # Create pipeline
    config = {"model_name": model}
    pipeline = ExtractionPipeline(config=config)

    # Track statistics
    stats = {"success": 0, "error": 0, "total_temporal": 0, "total_spatial": 0}

    # Process files
    for idx, file_path in enumerate(files, 1):
        try:
            console.print(f"[cyan][{idx}/{len(files)}] Processing:[/cyan] {file_path.name}")

            result = pipeline.extract_from_file(str(file_path))

            if not result.success:
                raise Exception(result.error)

            # Save result
            filename = f"{file_path.stem}_result"
            json_file, txt_file = _save_result(result, output_dir, filename)

            temporal_count = len(result.temporal_entities)
            spatial_count = len(result.spatial_entities)

            console.print(
                f"  [green]✓[/green] {temporal_count} temporal, "
                f"{spatial_count} spatial → {json_file.name}"
            )

            stats["success"] += 1
            stats["total_temporal"] += temporal_count
            stats["total_spatial"] += spatial_count

        except Exception as e:
            console.print(f"  [red]✗[/red] Error: {str(e)}")
            stats["error"] += 1

    # Summary
    console.print(f"\n{'=' * 80}")
    console.print(f"[bold green]Batch processing complete![/bold green]")
    console.print(f"  Processed: {stats['success']}/{len(files)} files")
    console.print(f"  Errors: {stats['error']}")
    console.print(f"  Total Temporal Entities: {stats['total_temporal']}")
    console.print(f"  Total Spatial Entities: {stats['total_spatial']}")
    console.print(f"\n[green]Results saved to:[/green] {output_dir}/")
    console.print(f"{'=' * 80}")


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
                entity.get("text", ""),
                entity.get("normalized", ""),
                entity.get("type", ""),
                f"{entity.get('confidence', 0.0):.2f}",
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
            coords = f"{entity.get('latitude', 0.0):.4f}, {entity.get('longitude', 0.0):.4f}"
            location = entity.get("locality") or entity.get("admin_area") or entity.get("country") or "N/A"
            spatial_table.add_row(
                entity.get("text", ""),
                coords,
                location,
                f"{entity.get('confidence', 0.0):.2f}"
            )

        console.print(spatial_table)


def _display_json(result, output: Optional[Path] = None):
    """Display results as JSON."""
    result_dict = {
        "text": result.text,
        "temporal_entities": result.temporal_entities,
        "spatial_entities": result.spatial_entities,
        "success": result.success,
        "error": result.error,
        "processing_time": result.processing_time,
        "metadata": result.metadata,
    }
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
