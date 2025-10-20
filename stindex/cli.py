"""
Command-line interface for STIndex.

Uses configuration files to specify LLM provider and settings.
"""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from stindex import __version__
from stindex.exe import execute_extract, execute_evaluate

app = typer.Typer(
    name="stindex",
    help="STIndex: Spatiotemporal Index Extraction from Unstructured Text",
    add_completion=False,
)
console = Console()


@app.command()
def version():
    """Show version information."""
    console.print(f"[bold blue]STIndex[/bold blue] version {__version__}")


@app.command()
def extract(
    text: str = typer.Argument(..., help="Text to extract spatiotemporal indices from"),
    config: str = typer.Option("extract", "--config", "-c", help="Config file name (default: extract.yml)"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Custom output file path (overrides auto-save)"),
):
    """Extract spatiotemporal indices from text."""
    execute_extract(
        text=text,
        config=config,
        output=output,
    )


@app.command()
def evaluate(
    config: str = typer.Option("evaluate", "--config", "-c", help="Config file name (default: evaluate.yml)"),
    dataset: Optional[Path] = typer.Option(None, "--dataset", "-d", help="Override dataset path"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-o", help="Override output directory"),
    sample_limit: Optional[int] = typer.Option(None, "--sample-limit", "-n", help="Limit number of samples to evaluate"),
    resume: Optional[bool] = typer.Option(None, "--resume/--no-resume", help="Resume from checkpoint"),
):
    """Run evaluation on a dataset."""
    execute_evaluate(
        config=config,
        dataset=dataset,
        output_dir=output_dir,
        sample_limit=sample_limit,
        resume=resume,
    )


def main():
    """Main entry point for CLI."""
    app()


if __name__ == "__main__":
    main()
