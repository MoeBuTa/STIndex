"""
Input models for preprocessing pipeline.

Defines data structures for different input types and processing outputs.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse


class InputType(str, Enum):
    """Types of input documents."""
    URL = "url"           # Web URL (requires scraping)
    FILE = "file"         # Local file path (requires reading + parsing)
    TEXT = "text"         # Raw text (requires only chunking)


@dataclass
class InputDocument:
    """
    Input document for preprocessing.

    Supports web URLs, local files, or raw text with optional metadata.
    """

    input_type: InputType
    content: str  # URL, file path, or raw text
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Optional fields
    document_id: Optional[str] = None
    title: Optional[str] = None

    @classmethod
    def from_url(
        cls,
        url: str,
        metadata: Optional[Dict[str, Any]] = None,
        document_id: Optional[str] = None,
        title: Optional[str] = None
    ) -> "InputDocument":
        """
        Create InputDocument from web URL.

        Args:
            url: Web URL to scrape
            metadata: Optional document metadata
            document_id: Optional unique document ID
            title: Optional document title

        Returns:
            InputDocument instance
        """
        metadata = metadata or {}
        metadata["url"] = url

        # Auto-generate document ID from URL if not provided
        if not document_id:
            parsed = urlparse(url)
            document_id = f"url_{parsed.netloc}_{hash(url) % 10000:04d}"

        return cls(
            input_type=InputType.URL,
            content=url,
            metadata=metadata,
            document_id=document_id,
            title=title
        )

    @classmethod
    def from_file(
        cls,
        file_path: str,
        metadata: Optional[Dict[str, Any]] = None,
        document_id: Optional[str] = None,
        title: Optional[str] = None
    ) -> "InputDocument":
        """
        Create InputDocument from local file.

        Args:
            file_path: Path to local file
            metadata: Optional document metadata
            document_id: Optional unique document ID
            title: Optional document title

        Returns:
            InputDocument instance
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        metadata = metadata or {}
        metadata["file_path"] = str(path.absolute())
        metadata["file_name"] = path.name
        metadata["file_extension"] = path.suffix

        # Auto-generate document ID from filename if not provided
        if not document_id:
            document_id = f"file_{path.stem}"

        # Auto-generate title from filename if not provided
        if not title:
            title = path.stem.replace("_", " ").replace("-", " ").title()

        return cls(
            input_type=InputType.FILE,
            content=str(path.absolute()),
            metadata=metadata,
            document_id=document_id,
            title=title
        )

    @classmethod
    def from_text(
        cls,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        document_id: Optional[str] = None,
        title: Optional[str] = None
    ) -> "InputDocument":
        """
        Create InputDocument from raw text.

        Args:
            text: Raw text content
            metadata: Optional document metadata
            document_id: Optional unique document ID
            title: Optional document title

        Returns:
            InputDocument instance
        """
        metadata = metadata or {}

        # Auto-generate document ID if not provided
        if not document_id:
            document_id = f"text_{hash(text[:100]) % 10000:04d}"

        # Auto-generate title if not provided
        if not title:
            # Use first 50 chars as title
            title = text[:50].strip()
            if len(text) > 50:
                title += "..."

        return cls(
            input_type=InputType.TEXT,
            content=text,
            metadata=metadata,
            document_id=document_id,
            title=title
        )


@dataclass
class ParsedDocument:
    """
    Parsed document with structured content.

    Output of parsing step, ready for chunking.
    """

    document_id: str
    title: str
    content: str  # Full text content
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Optional structured elements
    sections: Optional[List[Dict[str, Any]]] = None
    tables: Optional[List[Dict[str, Any]]] = None

    # Processing info
    parsing_method: Optional[str] = None
    parsed_at: Optional[str] = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class DocumentChunk:
    """
    Document chunk for extraction.

    Output of chunking step, ready for dimensional extraction.
    """

    chunk_id: str  # e.g., "doc1_chunk_0"
    chunk_index: int
    total_chunks: int
    text: str

    # Size metrics
    word_count: int
    char_count: int

    # Context preservation
    document_id: str
    document_title: str
    document_metadata: Dict[str, Any] = field(default_factory=dict)

    # Character positions in original document
    start_char: int = 0
    end_char: int = 0

    # Optional previous chunk summary for context
    previous_chunk_summary: Optional[str] = None

    # Hierarchical metadata (for element-based chunking)
    section_hierarchy: Optional[str] = None  # e.g., "Introduction > Background"
    element_types: Optional[List[str]] = None  # e.g., ["title", "text", "text"]
    keywords: Optional[List[str]] = None  # Representative keywords for the chunk
    summary: Optional[str] = None  # Brief summary of chunk content
    preview: Optional[str] = None  # First 1-2 sentences

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "chunk_id": self.chunk_id,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "text": self.text,
            "word_count": self.word_count,
            "char_count": self.char_count,
            "document_id": self.document_id,
            "document_title": self.document_title,
            "document_metadata": self.document_metadata,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "previous_chunk_summary": self.previous_chunk_summary,
            "section_hierarchy": self.section_hierarchy,
            "element_types": self.element_types,
            "keywords": self.keywords,
            "summary": self.summary,
            "preview": self.preview,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentChunk":
        """Create from dictionary."""
        return cls(**data)
