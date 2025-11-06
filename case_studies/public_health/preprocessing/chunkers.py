"""
Document chunking utilities for long health alerts.

Splits long documents into manageable chunks while preserving context
for accurate spatiotemporal extraction.
"""

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional

from loguru import logger


@dataclass
class DocumentChunk:
    """Container for a document chunk."""

    chunk_id: str  # e.g., "doc1_chunk_0"
    chunk_index: int
    total_chunks: int
    text: str
    word_count: int
    char_count: int

    # Context preservation
    document_title: str
    document_metadata: Dict[str, Any]
    previous_chunk_summary: Optional[str] = None  # Summary of previous chunk

    # Character positions in original document
    start_char: int = 0
    end_char: int = 0


class DocumentChunker:
    """Chunks long documents for extraction."""

    def __init__(
        self,
        max_chunk_size: int = 2000,  # characters
        overlap: int = 200,  # character overlap between chunks
        strategy: str = "sliding_window"  # or "semantic", "paragraph"
    ):
        """
        Initialize chunker.

        Args:
            max_chunk_size: Maximum chunk size in characters
            overlap: Overlap between chunks (to preserve context)
            strategy: Chunking strategy
        """
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap
        self.strategy = strategy

    def chunk_document(
        self,
        text: str,
        document_id: str,
        title: str = "",
        metadata: Optional[Dict[str, Any]] = None
    ) -> List[DocumentChunk]:
        """
        Chunk a document into smaller pieces.

        Args:
            text: Full document text
            document_id: Unique document identifier
            title: Document title
            metadata: Document metadata

        Returns:
            List of DocumentChunk objects
        """
        metadata = metadata or {}

        if len(text) <= self.max_chunk_size:
            # No chunking needed
            return [DocumentChunk(
                chunk_id=f"{document_id}_chunk_0",
                chunk_index=0,
                total_chunks=1,
                text=text,
                word_count=len(text.split()),
                char_count=len(text),
                document_title=title,
                document_metadata=metadata,
                start_char=0,
                end_char=len(text)
            )]

        # Choose chunking strategy
        if self.strategy == "sliding_window":
            return self._chunk_sliding_window(text, document_id, title, metadata)
        elif self.strategy == "paragraph":
            return self._chunk_by_paragraph(text, document_id, title, metadata)
        elif self.strategy == "semantic":
            return self._chunk_semantic(text, document_id, title, metadata)
        else:
            raise ValueError(f"Unknown chunking strategy: {self.strategy}")

    def _chunk_sliding_window(
        self,
        text: str,
        document_id: str,
        title: str,
        metadata: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """Chunk using sliding window with overlap."""
        chunks = []
        start = 0
        chunk_index = 0

        while start < len(text):
            # Calculate end position
            end = min(start + self.max_chunk_size, len(text))

            # If not at the end, try to break at a sentence boundary
            if end < len(text):
                # Look for sentence endings within last 200 chars
                search_start = max(start, end - 200)
                last_period = text.rfind('.', search_start, end)
                last_newline = text.rfind('\n', search_start, end)
                break_point = max(last_period, last_newline)

                if break_point > start:
                    end = break_point + 1  # Include the period/newline

            chunk_text = text[start:end].strip()

            if chunk_text:  # Skip empty chunks
                chunks.append(DocumentChunk(
                    chunk_id=f"{document_id}_chunk_{chunk_index}",
                    chunk_index=chunk_index,
                    total_chunks=-1,  # Will update later
                    text=chunk_text,
                    word_count=len(chunk_text.split()),
                    char_count=len(chunk_text),
                    document_title=title,
                    document_metadata=metadata,
                    start_char=start,
                    end_char=end
                ))
                chunk_index += 1

            # Move start position (with overlap)
            start = end - self.overlap
            if start <= 0 or end >= len(text):
                break

        # Update total_chunks
        total = len(chunks)
        for chunk in chunks:
            chunk.total_chunks = total

        return chunks

    def _chunk_by_paragraph(
        self,
        text: str,
        document_id: str,
        title: str,
        metadata: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """Chunk by paragraphs, respecting max_chunk_size."""
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = []
        current_size = 0
        chunk_index = 0
        char_position = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_size = len(para)

            # If single paragraph exceeds max size, use sliding window on it
            if para_size > self.max_chunk_size:
                # First, flush current chunk if any
                if current_chunk:
                    chunk_text = '\n\n'.join(current_chunk)
                    chunks.append(DocumentChunk(
                        chunk_id=f"{document_id}_chunk_{chunk_index}",
                        chunk_index=chunk_index,
                        total_chunks=-1,
                        text=chunk_text,
                        word_count=len(chunk_text.split()),
                        char_count=len(chunk_text),
                        document_title=title,
                        document_metadata=metadata,
                        start_char=char_position - current_size,
                        end_char=char_position
                    ))
                    chunk_index += 1
                    current_chunk = []
                    current_size = 0

                # Split large paragraph using sliding window
                para_chunks = self._chunk_sliding_window(para, f"{document_id}_para", title, metadata)
                for pc in para_chunks:
                    pc.chunk_id = f"{document_id}_chunk_{chunk_index}"
                    pc.chunk_index = chunk_index
                    chunks.append(pc)
                    chunk_index += 1

            # Add paragraph to current chunk
            elif current_size + para_size + 2 <= self.max_chunk_size:  # +2 for \n\n
                current_chunk.append(para)
                current_size += para_size + 2

            # Start new chunk
            else:
                if current_chunk:
                    chunk_text = '\n\n'.join(current_chunk)
                    chunks.append(DocumentChunk(
                        chunk_id=f"{document_id}_chunk_{chunk_index}",
                        chunk_index=chunk_index,
                        total_chunks=-1,
                        text=chunk_text,
                        word_count=len(chunk_text.split()),
                        char_count=len(chunk_text),
                        document_title=title,
                        document_metadata=metadata,
                        start_char=char_position - current_size,
                        end_char=char_position
                    ))
                    chunk_index += 1

                current_chunk = [para]
                current_size = para_size

            char_position += para_size + 2

        # Flush remaining
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            chunks.append(DocumentChunk(
                chunk_id=f"{document_id}_chunk_{chunk_index}",
                chunk_index=chunk_index,
                total_chunks=-1,
                text=chunk_text,
                word_count=len(chunk_text.split()),
                char_count=len(chunk_text),
                document_title=title,
                document_metadata=metadata,
                start_char=char_position - current_size,
                end_char=char_position
            ))

        # Update total_chunks
        total = len(chunks)
        for chunk in chunks:
            chunk.total_chunks = total

        return chunks

    def _chunk_semantic(
        self,
        text: str,
        document_id: str,
        title: str,
        metadata: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """
        Semantic chunking using sentence embeddings.

        Note: This requires sentence-transformers or similar.
        For now, falls back to paragraph chunking.
        """
        logger.warning("Semantic chunking not implemented yet, falling back to paragraph chunking")
        return self._chunk_by_paragraph(text, document_id, title, metadata)

    def chunk_parsed_documents(
        self,
        parsed_docs_file: str,
        output_dir: str = "case_studies/public_health/data/processed"
    ):
        """
        Chunk parsed documents from JSON file.

        Args:
            parsed_docs_file: Path to parsed documents JSON
            output_dir: Directory to save chunked documents
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        with open(parsed_docs_file, 'r', encoding='utf-8') as f:
            parsed_docs = json.load(f)

        all_chunks = []

        for i, doc in enumerate(parsed_docs):
            document_id = f"doc_{i}"
            title = doc.get("title", "Untitled")
            content = doc.get("content", "")
            metadata = doc.get("metadata", {})

            chunks = self.chunk_document(content, document_id, title, metadata)
            all_chunks.extend(chunks)

            logger.info(f"Document '{title}': {len(chunks)} chunks")

        # Save chunks
        chunks_data = [asdict(chunk) for chunk in all_chunks]
        output_file = output_path / f"chunked_{Path(parsed_docs_file).name}"

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)

        logger.info(f"✓ Saved {len(all_chunks)} chunks to {output_file}")


def chunk_all_parsed_documents(
    processed_dir: str = "case_studies/public_health/data/processed",
    max_chunk_size: int = 2000,
    strategy: str = "paragraph"
):
    """
    Chunk all parsed documents in the processed directory.

    Args:
        processed_dir: Directory with parsed documents
        max_chunk_size: Maximum chunk size in characters
        strategy: Chunking strategy
    """
    logger.info("Chunking parsed documents...")

    chunker = DocumentChunker(
        max_chunk_size=max_chunk_size,
        overlap=200,
        strategy=strategy
    )

    processed_path = Path(processed_dir)
    parsed_files = list(processed_path.glob("parsed_*.json"))

    for parsed_file in parsed_files:
        logger.info(f"Chunking {parsed_file.name}...")
        chunker.chunk_parsed_documents(str(parsed_file), processed_dir)

    logger.info("✓ Chunking complete!")


if __name__ == "__main__":
    # Chunk all parsed documents
    chunk_all_parsed_documents()
