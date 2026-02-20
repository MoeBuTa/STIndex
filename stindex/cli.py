"""
Command-line interface for STIndex.

Uses configuration files to specify LLM provider and settings.
LLM model settings (model, temperature, max_tokens, base_url) can be
overridden here; defaults live in stindex/utils/config.py PROVIDER_DEFAULTS.
"""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from stindex import __version__
from stindex.exe import execute_extract, execute_context_aware_evaluation

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
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model name override (e.g., 'gpt-4o', 'Qwen/Qwen3-8B')"),
    temperature: Optional[float] = typer.Option(None, "--temperature", "-t", help="Sampling temperature override"),
    max_tokens: Optional[int] = typer.Option(None, "--max-tokens", help="Max tokens to generate override"),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="LLM server base URL override (for hf provider)"),
    auto_start: bool = typer.Option(True, "--auto-start/--no-auto-start", help="Enable/disable automatic server startup"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Custom output file path"),
):
    """Extract spatiotemporal indices from text."""
    execute_extract(
        text=text,
        config=config,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        base_url=base_url,
        auto_start=auto_start,
        output=output,
    )


@app.command()
def evaluate(
    config: str = typer.Option("extract", "--config", "-c", help="Config file name (default: extract.yml)"),
    dataset: Optional[Path] = typer.Option(None, "--dataset", "-d", help="Path to evaluation dataset JSON"),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-o", help="Output directory for results"),
    sample_limit: Optional[int] = typer.Option(None, "--sample-limit", "-n", help="Limit number of chunks to process"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Model name override"),
    temperature: Optional[float] = typer.Option(None, "--temperature", "-t", help="Sampling temperature override"),
    max_tokens: Optional[int] = typer.Option(None, "--max-tokens", help="Max tokens to generate override"),
    base_url: Optional[str] = typer.Option(None, "--base-url", help="LLM server base URL override"),
):
    """
    Run context-aware evaluation comparing baseline vs context-aware extraction.

    Compares two modes:
    1. Baseline: No context (each chunk extracted independently)
    2. Context-aware: With ExtractionContext (maintains state across chunks)
    """
    execute_context_aware_evaluation(
        config=config,
        dataset=dataset,
        output_dir=output_dir,
        sample_limit=sample_limit,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        base_url=base_url,
    )


def main():
    """Main entry point for CLI."""
    app()


if __name__ == "__main__":
    main()
