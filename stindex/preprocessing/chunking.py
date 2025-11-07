"""
Document chunking module.

Splits long documents into manageable chunks while preserving context
for accurate extraction.
"""

from typing import List, Optional

from loguru import logger

from stindex.preprocessing.input_models import DocumentChunk, ParsedDocument


class DocumentChunker:
    """
    Chunks long documents for extraction.

    Supports multiple chunking strategies:
    - sliding_window: Fixed-size chunks with overlap
    - paragraph: Chunk by paragraphs, respecting max size
    - semantic: Semantic chunking using embeddings (future)
    """

    def __init__(
        self,
        max_chunk_size: int = 2000,  # characters
        overlap: int = 200,  # character overlap between chunks
        strategy: str = "sliding_window"  # or "paragraph", "semantic"
    ):
        """
        Initialize chunker.

        Args:
            max_chunk_size: Maximum chunk size in characters
            overlap: Overlap between chunks (to preserve context)
            strategy: Chunking strategy ("sliding_window", "paragraph", "semantic")
        """
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap
        self.strategy = strategy

    def chunk_text(
        self,
        text: str,
        document_id: str,
        title: str = "",
        metadata: Optional[dict] = None
    ) -> List[DocumentChunk]:
        """
        Chunk text into smaller pieces.

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
                document_id=document_id,
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

    def chunk_parsed_document(self, parsed_doc: ParsedDocument) -> List[DocumentChunk]:
        """
        Chunk a ParsedDocument.

        Args:
            parsed_doc: ParsedDocument object

        Returns:
            List of DocumentChunk objects
        """
        return self.chunk_text(
            text=parsed_doc.content,
            document_id=parsed_doc.document_id,
            title=parsed_doc.title,
            metadata=parsed_doc.metadata
        )

    def _chunk_sliding_window(
        self,
        text: str,
        document_id: str,
        title: str,
        metadata: dict
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
                    document_id=document_id,
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
        metadata: dict
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
                        document_id=document_id,
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
                    pc.document_id = document_id
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
                        document_id=document_id,
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
                document_id=document_id,
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
        metadata: dict
    ) -> List[DocumentChunk]:
        """
        Semantic chunking using sentence embeddings.

        Note: This requires sentence-transformers or similar.
        For now, falls back to paragraph chunking.
        """
        logger.warning("Semantic chunking not implemented yet, falling back to paragraph chunking")
        return self._chunk_by_paragraph(text, document_id, title, metadata)
