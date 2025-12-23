"""
Execution module for CLI commands.

This module contains CLI executables for STIndex operations:
- extract: Simple text extraction
- extract_corpus: Large-scale corpus extraction
- discover_schema: Schema discovery from Q&A datasets
- evaluate: Evaluation pipelines

Core pipeline classes are in stindex.pipeline module.
"""

from .extract import execute_extract
from .evaluate import execute_context_aware_evaluation
from .utils import get_output_dir, save_result, display_json


__all__ = [
    "execute_extract",
    "execute_context_aware_evaluation",
    "get_output_dir",
    "save_result",
    "display_json",
]
