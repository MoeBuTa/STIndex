"""
Document chunking module.

Splits long documents into manageable chunks while preserving context
for accurate extraction.
"""

from typing import List, Optional
import re

from loguru import logger

from stindex.preprocess.input_models import DocumentChunk, ParsedDocument


class DocumentChunker:
    """
    Chunks long documents for extraction.

    Supports multiple chunking strategies:
    - sliding_window: Fixed-size chunks with overlap
    - paragraph: Chunk by paragraphs, respecting max size
    - element_based: Chunk by structural elements (titles, tables) with metadata
    - semantic: Semantic chunking using embeddings (future)
    """

    def __init__(
        self,
        max_chunk_size: int = 2000,  # characters
        overlap: int = 200,  # character overlap between chunks
        strategy: str = "sliding_window"  # or "paragraph", "element_based", "semantic"
    ):
        """
        Initialize chunker.

        Args:
            max_chunk_size: Maximum chunk size in characters
            overlap: Overlap between chunks (to preserve context)
            strategy: Chunking strategy ("sliding_window", "paragraph", "element_based", "semantic")
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
                doc_id=f"{document_id}_chunk_0",
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
        elif self.strategy == "element_based":
            # Element-based needs ParsedDocument with sections, fall back to paragraph
            logger.warning("element_based strategy requires ParsedDocument with sections, falling back to paragraph")
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
        # Use element-based chunking if strategy is element_based and sections are available
        if self.strategy == "element_based" and parsed_doc.sections:
            return self._chunk_element_based(parsed_doc)

        # Otherwise, use text-based chunking
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
                    doc_id=f"{document_id}_chunk_{chunk_index}",
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
                        doc_id=f"{document_id}_chunk_{chunk_index}",
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
                    pc.doc_id = f"{document_id}_chunk_{chunk_index}"
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
                        doc_id=f"{document_id}_chunk_{chunk_index}",
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
                doc_id=f"{document_id}_chunk_{chunk_index}",
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

    def _chunk_element_based(
        self,
        parsed_doc: ParsedDocument
    ) -> List[DocumentChunk]:
        """
        Chunk by structural elements with hierarchical metadata.

        Based on Financial Report Chunking (2024) best practices:
        - Start new chunks at titles and table elements
        - Merge elements up to max_chunk_size without breaking structure
        - Keep tables intact (never fragment)
        - Add metadata to each chunk (hierarchy, keywords, preview, summary)

        Args:
            parsed_doc: ParsedDocument with sections

        Returns:
            List of DocumentChunk objects with enriched metadata
        """
        if not parsed_doc.sections:
            logger.warning("No sections found in parsed document, falling back to text chunking")
            return self.chunk_text(
                parsed_doc.content,
                parsed_doc.document_id,
                parsed_doc.title,
                parsed_doc.metadata
            )

        logger.info(f"Element-based chunking: {len(parsed_doc.sections)} sections")

        chunks = []
        current_elements = []
        current_size = 0
        chunk_index = 0
        section_hierarchy = []

        for section in parsed_doc.sections:
            section_type = section.get("type", "other")
            section_text = section.get("text", "")

            # Update section hierarchy
            if section_type == "title":
                # Extract level if available
                level = section.get("level", 1)
                # Trim hierarchy based on level
                section_hierarchy = section_hierarchy[:level-1]
                section_hierarchy.append(section_text)

            # Start new chunk at structural boundaries (titles, tables)
            if section_type in ["title", "table"] and current_elements:
                chunk = self._finalize_element_chunk(
                    current_elements,
                    parsed_doc.document_id,
                    parsed_doc.title,
                    parsed_doc.metadata,
                    chunk_index,
                    section_hierarchy[:-1] if section_type == "title" else section_hierarchy
                )
                chunks.append(chunk)
                chunk_index += 1
                current_elements = []
                current_size = 0

            # Add section to current chunk
            current_elements.append(section)
            current_size += len(section_text)

            # Chunk size limit (but don't break tables or titles)
            if current_size >= self.max_chunk_size and section_type not in ["table", "title"]:
                chunk = self._finalize_element_chunk(
                    current_elements,
                    parsed_doc.document_id,
                    parsed_doc.title,
                    parsed_doc.metadata,
                    chunk_index,
                    list(section_hierarchy)
                )
                chunks.append(chunk)
                chunk_index += 1
                current_elements = []
                current_size = 0

        # Finalize last chunk
        if current_elements:
            chunk = self._finalize_element_chunk(
                current_elements,
                parsed_doc.document_id,
                parsed_doc.title,
                parsed_doc.metadata,
                chunk_index,
                list(section_hierarchy)
            )
            chunks.append(chunk)

        # Update total_chunks
        total = len(chunks)
        for chunk in chunks:
            chunk.total_chunks = total

        logger.info(f"✓ Created {total} element-based chunks")
        return chunks

    def _finalize_element_chunk(
        self,
        elements: List[dict],
        document_id: str,
        document_title: str,
        document_metadata: dict,
        chunk_index: int,
        section_hierarchy: List[str]
    ) -> DocumentChunk:
        """
        Finalize chunk with metadata enrichment.

        Metadata types:
        1. Section hierarchy (from titles)
        2. Keywords (extracted from text)
        3. Preview (first 1-2 sentences)
        4. Element types (for debugging)

        Args:
            elements: List of section dicts
            document_id: Document identifier
            document_title: Document title
            document_metadata: Document metadata
            chunk_index: Chunk index
            section_hierarchy: Current section path

        Returns:
            DocumentChunk with enriched metadata
        """
        # Build chunk text
        chunk_texts = []
        element_types = []

        for element in elements:
            element_type = element.get("type", "other")
            element_types.append(element_type)

            if element_type == "table":
                # For tables, use raw_text from table data
                table_data = element.get("data", {})
                chunk_texts.append(table_data.get("raw_text", str(element)))
            else:
                chunk_texts.append(element.get("text", ""))

        chunk_text = "\n\n".join(chunk_texts)

        # Extract metadata
        keywords = self._extract_keywords(chunk_text)
        preview = self._get_preview(chunk_text)
        hierarchy_str = " > ".join(section_hierarchy) if section_hierarchy else None

        return DocumentChunk(
            doc_id=f"{document_id}_chunk_{chunk_index}",
            chunk_index=chunk_index,
            total_chunks=-1,  # Will be updated later
            text=chunk_text,
            word_count=len(chunk_text.split()),
            char_count=len(chunk_text),
            document_id=document_id,
            document_title=document_title,
            document_metadata=document_metadata,
            start_char=0,  # Not tracked in element-based
            end_char=len(chunk_text),
            section_hierarchy=hierarchy_str,
            element_types=element_types,
            keywords=keywords,
            preview=preview,
            summary=None  # Could be generated with LLM later
        )

    def _extract_keywords(self, text: str, max_keywords: int = 6) -> List[str]:
        """
        Extract representative keywords from text.

        Uses simple frequency-based extraction. Could be enhanced with:
        - TF-IDF
        - KeyBERT
        - LLM extraction (GPT-4)

        Args:
            text: Input text
            max_keywords: Maximum number of keywords

        Returns:
            List of keywords
        """
        # Simple implementation: extract most common meaningful words
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())

        # Filter out common stop words
        stop_words = {
            'that', 'this', 'with', 'from', 'have', 'will', 'been', 'were',
            'said', 'would', 'could', 'their', 'about', 'which', 'there', 'where',
            'when', 'what', 'who', 'whom', 'whose', 'they', 'them', 'these',
            'those', 'then', 'than', 'such', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'under', 'again',
            'further', 'once', 'here', 'more', 'most', 'other', 'some', 'only',
            'very', 'also', 'just', 'should', 'being', 'both', 'each', 'much',
            'many', 'same', 'over', 'because', 'however', 'therefore', 'thus'
        }

        # Count word frequencies
        word_freq = {}
        for word in words:
            if word not in stop_words and len(word) > 3:
                word_freq[word] = word_freq.get(word, 0) + 1

        # Sort by frequency and take top keywords
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        keywords = [word for word, freq in sorted_words[:max_keywords]]

        return keywords

    def _get_preview(self, text: str, num_sentences: int = 2) -> str:
        """
        Extract first N sentences as preview.

        Args:
            text: Input text
            num_sentences: Number of sentences to extract

        Returns:
            Preview text
        """
        # Split into sentences (simple approach)
        sentences = re.split(r'[.!?]+\s+', text.strip())

        # Take first N sentences
        preview_sentences = sentences[:num_sentences]
        preview = '. '.join(preview_sentences)

        # Add period if not already there
        if preview and not preview.endswith(('.', '!', '?')):
            preview += '.'

        # Limit length
        max_length = 200
        if len(preview) > max_length:
            preview = preview[:max_length].rsplit(' ', 1)[0] + '...'

        return preview
