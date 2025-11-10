"""
Preprocessing orchestrator.

Coordinates scraping, parsing, and chunking based on input type.
"""

from typing import List, Optional

from loguru import logger

from stindex.preprocess.chunking import DocumentChunker
from stindex.preprocess.input_models import (
    DocumentChunk,
    InputDocument,
    InputType,
    ParsedDocument,
)
from stindex.preprocess.parsing import DocumentParser
from stindex.preprocess.scraping import WebScraper
from stindex.utils.config import load_preprocess_config


class Preprocessor:
    """
    Main preprocessing orchestrator.

    Handles the full preprocessing pipeline:
    - Web URLs: scraping → parsing → chunking
    - Files: parsing → chunking
    - Text: chunking only

    Usage:
        preprocessor = Preprocessor(
            max_chunk_size=2000,
            chunking_strategy="paragraph"
        )

        # From URL
        doc = InputDocument.from_url("https://example.com/article")
        chunks = preprocessor.process(doc)

        # From file
        doc = InputDocument.from_file("/path/to/document.pdf")
        chunks = preprocessor.process(doc)

        # From text
        doc = InputDocument.from_text("Your text here")
        chunks = preprocessor.process(doc)
    """

    def __init__(self):
        """
        Initialize preprocessor.

        Loads all settings from cfg/preprocess/*.yml files.
        """
        # Load configurations from YAML files
        logger.debug("Loading preprocessing configs from cfg/preprocess/")
        chunking_config = load_preprocess_config('chunking')
        parsing_config = load_preprocess_config('parsing')
        scraping_config = load_preprocess_config('scraping')

        # Chunking settings from config
        chunk_strategy = chunking_config.get('strategy', 'sliding_window')
        strategy_config = chunking_config.get(chunk_strategy, {})
        chunk_size = strategy_config.get('max_chunk_size', 2000)
        overlap = strategy_config.get('overlap', 200)

        # Parsing settings from config
        parse_method = parsing_config.get('parsing_method', 'unstructured')

        # Scraping settings from config
        agent = scraping_config.get('user_agent', 'STIndex-Research/1.0')
        rate = scraping_config.get('rate_limit', 2.0)

        # Initialize components
        self.scraper = WebScraper(
            user_agent=agent,
            rate_limit=rate
        )

        self.parser = DocumentParser(
            parsing_method=parse_method
        )

        self.chunker = DocumentChunker(
            max_chunk_size=chunk_size,
            overlap=overlap,
            strategy=chunk_strategy
        )

        logger.debug(f"Preprocessor initialized from config: chunking={chunk_strategy}, "
                    f"parsing={parse_method}, chunk_size={chunk_size}")

    def process(
        self,
        input_doc: InputDocument,
        skip_chunking: bool = False
    ) -> List[DocumentChunk]:
        """
        Process an input document through the full preprocessing pipeline.

        Args:
            input_doc: InputDocument to process
            skip_chunking: If True, return single chunk with full content

        Returns:
            List of DocumentChunk objects ready for extraction

        Raises:
            ValueError: If processing fails
        """
        logger.info(f"Processing {input_doc.input_type} document: {input_doc.document_id}")

        # Step 1: Get content based on input type
        if input_doc.input_type == InputType.URL:
            parsed_doc = self._process_url(input_doc)
        elif input_doc.input_type == InputType.FILE:
            parsed_doc = self._process_file(input_doc)
        elif input_doc.input_type == InputType.TEXT:
            parsed_doc = self._process_text(input_doc)
        else:
            raise ValueError(f"Unknown input type: {input_doc.input_type}")

        # Step 2: Chunk the parsed document
        if skip_chunking:
            # Return single chunk with full content
            chunks = [DocumentChunk(
                chunk_id=f"{parsed_doc.document_id}_chunk_0",
                chunk_index=0,
                total_chunks=1,
                text=parsed_doc.content,
                word_count=len(parsed_doc.content.split()),
                char_count=len(parsed_doc.content),
                document_id=parsed_doc.document_id,
                document_title=parsed_doc.title,
                document_metadata=parsed_doc.metadata,
                start_char=0,
                end_char=len(parsed_doc.content)
            )]
        else:
            chunks = self.chunker.chunk_parsed_document(parsed_doc)

        logger.info(f"✓ Preprocessing complete: {len(chunks)} chunks")
        return chunks

    def _process_url(self, input_doc: InputDocument) -> ParsedDocument:
        """Process web URL: scrape → parse."""
        url = input_doc.content

        # Scrape
        logger.info(f"Scraping URL: {url}")
        html, error = self.scraper.scrape(url)

        if error:
            raise ValueError(f"Scraping failed: {error}")

        # Parse
        logger.info(f"Parsing HTML content...")
        parsed_doc = self.parser.parse_html_string(
            html_content=html,
            document_id=input_doc.document_id,
            title=input_doc.title,
            metadata=input_doc.metadata
        )

        logger.info(f"✓ Parsed {len(parsed_doc.content)} chars from URL")
        return parsed_doc

    def _process_file(self, input_doc: InputDocument) -> ParsedDocument:
        """Process file: parse only."""
        file_path = input_doc.content

        # Parse
        logger.info(f"Parsing file: {file_path}")
        parsed_doc = self.parser.parse_file(
            file_path=file_path,
            document_id=input_doc.document_id,
            title=input_doc.title,
            metadata=input_doc.metadata
        )

        logger.info(f"✓ Parsed {len(parsed_doc.content)} chars from file")
        return parsed_doc

    def _process_text(self, input_doc: InputDocument) -> ParsedDocument:
        """Process raw text: no parsing needed."""
        text = input_doc.content

        # Parse (just wrap in ParsedDocument)
        logger.info(f"Processing raw text ({len(text)} chars)...")
        parsed_doc = self.parser.parse_text(
            text=text,
            document_id=input_doc.document_id,
            title=input_doc.title,
            metadata=input_doc.metadata
        )

        logger.info(f"✓ Wrapped {len(parsed_doc.content)} chars of raw text")
        return parsed_doc

    def process_batch(
        self,
        input_docs: List[InputDocument],
        skip_chunking: bool = False
    ) -> List[List[DocumentChunk]]:
        """
        Process multiple documents.

        Args:
            input_docs: List of InputDocument objects
            skip_chunking: If True, return single chunk per document

        Returns:
            List of lists of DocumentChunk objects (one list per document)
        """
        all_chunks = []

        for i, input_doc in enumerate(input_docs):
            try:
                logger.info(f"\n[{i+1}/{len(input_docs)}] Processing: {input_doc.document_id}")
                chunks = self.process(input_doc, skip_chunking=skip_chunking)
                all_chunks.append(chunks)
            except Exception as e:
                logger.error(f"Failed to process {input_doc.document_id}: {e}")
                # Continue with next document
                continue

        logger.info(f"\n✓ Batch preprocessing complete: {len(all_chunks)}/{len(input_docs)} documents processed")
        return all_chunks
