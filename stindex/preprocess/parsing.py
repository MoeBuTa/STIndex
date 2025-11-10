"""
Generic document parsing module.

Parses various document formats (HTML, PDF, TXT, etc.) into structured text
using the unstructured library.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from stindex.preprocess.input_models import ParsedDocument


# Ensure NLTK data is available (required by unstructured)
def _ensure_nltk_data():
    """Download NLTK data if not already available."""
    try:
        import nltk
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            logger.info("Downloading required NLTK data...")
            nltk.download('punkt', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            logger.info("✓ NLTK data downloaded")
    except Exception as e:
        logger.warning(f"Failed to setup NLTK data: {e}")


_ensure_nltk_data()


# Import unstructured
try:
    from unstructured.partition.html import partition_html
    from unstructured.partition.auto import partition
    from unstructured.documents.elements import Element, Title, NarrativeText, Table
    from unstructured.cleaners.core import clean_extra_whitespace
    UNSTRUCTURED_AVAILABLE = True
except ImportError:
    logger.warning(
        "unstructured package not found. Install with: pip install 'unstructured[local-inference]'"
    )
    UNSTRUCTURED_AVAILABLE = False


class DocumentParser:
    """
    Generic document parser using unstructured library.

    Supports HTML, PDF, DOCX, TXT, and other formats.
    """

    def __init__(self, parsing_method: str = "unstructured"):
        """
        Initialize document parser.

        Args:
            parsing_method: Parsing method to use ("unstructured" or "simple")
        """
        self.parsing_method = parsing_method

        if parsing_method == "unstructured" and not UNSTRUCTURED_AVAILABLE:
            logger.warning("unstructured not available, falling back to simple parsing")
            self.parsing_method = "simple"

    def parse_html_string(
        self,
        html_content: str,
        document_id: str,
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ParsedDocument:
        """
        Parse HTML content string.

        Args:
            html_content: Raw HTML string
            document_id: Unique document identifier
            title: Optional document title
            metadata: Optional metadata dict

        Returns:
            ParsedDocument object
        """
        metadata = metadata or {}

        if self.parsing_method == "unstructured":
            return self._parse_html_unstructured(html_content, document_id, title, metadata)
        else:
            return self._parse_html_simple(html_content, document_id, title, metadata)

    def parse_file(
        self,
        file_path: str,
        document_id: Optional[str] = None,
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ParsedDocument:
        """
        Parse document from file (auto-detects format).

        Args:
            file_path: Path to document file
            document_id: Optional unique document identifier
            title: Optional document title
            metadata: Optional metadata dict

        Returns:
            ParsedDocument object
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        metadata = metadata or {}
        metadata["file_path"] = str(path.absolute())
        metadata["file_name"] = path.name

        document_id = document_id or f"file_{path.stem}"
        title = title or path.stem.replace("_", " ").replace("-", " ").title()

        logger.info(f"Parsing {path.name} using {self.parsing_method}...")

        if self.parsing_method == "unstructured":
            return self._parse_file_unstructured(path, document_id, title, metadata)
        else:
            return self._parse_file_simple(path, document_id, title, metadata)

    def parse_text(
        self,
        text: str,
        document_id: str,
        title: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ParsedDocument:
        """
        Parse raw text (no parsing needed, just wrap in ParsedDocument).

        Args:
            text: Raw text content
            document_id: Unique document identifier
            title: Optional document title
            metadata: Optional metadata dict

        Returns:
            ParsedDocument object
        """
        metadata = metadata or {}

        if not title:
            title = text[:50].strip()
            if len(text) > 50:
                title += "..."

        return ParsedDocument(
            document_id=document_id,
            title=title,
            content=text,
            metadata=metadata,
            parsing_method="raw_text"
        )

    def _parse_html_unstructured(
        self,
        html_content: str,
        document_id: str,
        title: Optional[str],
        metadata: Dict[str, Any]
    ) -> ParsedDocument:
        """Parse HTML using unstructured library."""
        # Use unstructured to partition HTML
        elements = partition_html(text=html_content)

        # Extract different element types
        extracted_title = ""
        sections = []
        tables = []
        full_text_parts = []

        for element in elements:
            # Clean whitespace
            element_text = clean_extra_whitespace(element.text)

            if isinstance(element, Title):
                if not extracted_title:  # Use first title as document title
                    extracted_title = element_text
                sections.append({
                    "type": "title",
                    "text": element_text,
                    "level": getattr(element, 'level', 1)
                })

            elif isinstance(element, Table):
                # Extract table structure
                table_data = self._extract_table_data(element)
                tables.append(table_data)
                sections.append({
                    "type": "table",
                    "data": table_data
                })

            elif isinstance(element, NarrativeText):
                sections.append({
                    "type": "text",
                    "text": element_text
                })

            else:
                # Other element types
                sections.append({
                    "type": "other",
                    "text": element_text,
                    "element_type": type(element).__name__
                })

            # Add to full text
            if element_text:
                full_text_parts.append(element_text)

        # Combine full text
        full_text = "\n\n".join(full_text_parts)

        return ParsedDocument(
            document_id=document_id,
            title=title or extracted_title or "Untitled",
            content=full_text,
            sections=sections,
            tables=tables,
            metadata=metadata,
            parsing_method="unstructured"
        )

    def _parse_html_simple(
        self,
        html_content: str,
        document_id: str,
        title: Optional[str],
        metadata: Dict[str, Any]
    ) -> ParsedDocument:
        """Parse HTML using simple BeautifulSoup extraction."""
        try:
            from bs4 import BeautifulSoup
        except ImportError:
            raise ImportError("BeautifulSoup4 is required for simple parsing. Install with: pip install beautifulsoup4")

        soup = BeautifulSoup(html_content, 'html.parser')

        # Extract title
        extracted_title = ""
        title_tag = soup.find('title')
        if title_tag:
            extracted_title = title_tag.get_text(strip=True)
        elif soup.find('h1'):
            extracted_title = soup.find('h1').get_text(strip=True)

        # Remove script and style elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header']):
            element.decompose()

        # Extract main content
        main_content = (
            soup.find('main') or
            soup.find('article') or
            soup.find('div', class_='content') or
            soup.find('div', id='content') or
            soup.find('body')
        )

        if main_content:
            text = main_content.get_text(separator='\n', strip=True)
        else:
            text = soup.get_text(separator='\n', strip=True)

        # Clean up extra whitespace
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        text = '\n'.join(lines)

        return ParsedDocument(
            document_id=document_id,
            title=title or extracted_title or "Untitled",
            content=text,
            metadata=metadata,
            parsing_method="simple_beautifulsoup"
        )

    def _parse_file_unstructured(
        self,
        path: Path,
        document_id: str,
        title: str,
        metadata: Dict[str, Any]
    ) -> ParsedDocument:
        """Parse file using unstructured library."""
        # Use unstructured's auto-partition
        elements = partition(str(path))

        # Process elements similar to HTML parsing
        extracted_title = ""
        sections = []
        tables = []
        full_text_parts = []

        for element in elements:
            element_text = clean_extra_whitespace(element.text)

            if isinstance(element, Title):
                if not extracted_title:
                    extracted_title = element_text
                sections.append({"type": "title", "text": element_text})

            elif isinstance(element, Table):
                table_data = self._extract_table_data(element)
                tables.append(table_data)
                sections.append({"type": "table", "data": table_data})

            elif isinstance(element, NarrativeText):
                sections.append({"type": "text", "text": element_text})

            else:
                sections.append({
                    "type": "other",
                    "text": element_text,
                    "element_type": type(element).__name__
                })

            if element_text:
                full_text_parts.append(element_text)

        full_text = "\n\n".join(full_text_parts)

        return ParsedDocument(
            document_id=document_id,
            title=title or extracted_title or path.stem,
            content=full_text,
            sections=sections,
            tables=tables,
            metadata=metadata,
            parsing_method="unstructured"
        )

    def _parse_file_simple(
        self,
        path: Path,
        document_id: str,
        title: str,
        metadata: Dict[str, Any]
    ) -> ParsedDocument:
        """Parse file using simple text extraction."""
        # For simple parsing, just read as text
        try:
            with open(path, 'r', encoding='utf-8') as f:
                text = f.read()
        except UnicodeDecodeError:
            # Try with different encoding
            with open(path, 'r', encoding='latin-1') as f:
                text = f.read()

        return ParsedDocument(
            document_id=document_id,
            title=title,
            content=text,
            metadata=metadata,
            parsing_method="simple_text_read"
        )

    def _extract_table_data(self, table_element: Table) -> Dict[str, Any]:
        """
        Extract structured data from table element.

        Args:
            table_element: Table element from unstructured

        Returns:
            Dict with table data
        """
        table_text = clean_extra_whitespace(table_element.text)

        return {
            "raw_text": table_text,
            "rows": self._parse_table_rows(table_text)
        }

    def _parse_table_rows(self, table_text: str) -> List[List[str]]:
        """
        Parse table text into rows.

        Args:
            table_text: Raw table text

        Returns:
            List of rows (each row is a list of cells)
        """
        rows = []
        for line in table_text.split('\n'):
            if line.strip():
                # Basic cell splitting (would need enhancement for complex tables)
                cells = [cell.strip() for cell in line.split('\t') if cell.strip()]
                if cells:
                    rows.append(cells)
        return rows
