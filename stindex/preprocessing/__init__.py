"""
Generic preprocessing module for STIndex.

Handles web scraping, document parsing, and chunking for various input types:
- Web URLs (HTTP/HTTPS)
- Local files (HTML, PDF, TXT, etc.)
- Raw text strings

Usage:
    from stindex.preprocessing import Preprocessor, InputDocument

    # Web URL
    doc = InputDocument.from_url("https://example.com/article.html", metadata={"source": "example"})
    preprocessor = Preprocessor()
    chunks = preprocessor.process(doc)

    # File path
    doc = InputDocument.from_file("/path/to/document.pdf", metadata={"source": "local"})
    chunks = preprocessor.process(doc)

    # Raw text
    doc = InputDocument.from_text("Your text here", metadata={"title": "My Document"})
    chunks = preprocessor.process(doc)
"""

from stindex.preprocessing.input_models import InputDocument, DocumentChunk
from stindex.preprocessing.processor import Preprocessor

__all__ = [
    "InputDocument",
    "DocumentChunk",
    "Preprocessor",
]
