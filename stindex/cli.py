"""
Command-line interface for STIndex.

Uses configuration files to specify LLM provider and settings.
"""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from stindex import __version__
from stindex.exe import execute_extract

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


def main():
    """Main entry point for CLI."""
    app()


if __name__ == "__main__":
    main()
