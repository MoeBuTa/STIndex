"""
Generic web scraping module.

Fetches content from web URLs with rate limiting and error handling.
"""

import time
from typing import Optional, Tuple
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from loguru import logger


class WebScraper:
    """
    Generic web scraper with rate limiting and polite behavior.

    Fetches HTML content from web URLs for further parsing.
    """

    def __init__(
        self,
        user_agent: str = "STIndex-Research/1.0",
        rate_limit: float = 2.0,
        timeout: int = 30
    ):
        """
        Initialize web scraper.

        Args:
            user_agent: User agent string for requests
            rate_limit: Minimum seconds between requests
            timeout: Request timeout in seconds
        """
        self.user_agent = user_agent
        self.rate_limit = rate_limit
        self.timeout = timeout

        self.headers = {
            "User-Agent": user_agent
        }

        self.last_request_time = 0

    def _rate_limit_wait(self):
        """Respect rate limiting between requests."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self.last_request_time = time.time()

    def scrape(self, url: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Fetch HTML content from URL.

        Args:
            url: Web URL to scrape

        Returns:
            Tuple of (html_content, error_message)
            - html_content: HTML string if successful, None otherwise
            - error_message: Error message if failed, None otherwise
        """
        # Validate URL
        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                return None, "Invalid URL format"
        except Exception as e:
            return None, f"URL parsing error: {str(e)}"

        # Rate limit
        self._rate_limit_wait()

        # Fetch
        try:
            logger.info(f"Fetching: {url}")
            response = requests.get(
                url,
                headers=self.headers,
                timeout=self.timeout
            )
            response.raise_for_status()

            content_type = response.headers.get('content-type', '').lower()

            # Check if HTML
            if 'text/html' not in content_type and 'application/xhtml' not in content_type:
                logger.warning(f"Non-HTML content type: {content_type}")

            logger.info(f"✓ Fetched {len(response.text)} chars from {url}")
            return response.text, None

        except requests.Timeout:
            error_msg = f"Request timeout after {self.timeout}s"
            logger.error(f"Failed to fetch {url}: {error_msg}")
            return None, error_msg

        except requests.HTTPError as e:
            error_msg = f"HTTP error {e.response.status_code}: {str(e)}"
            logger.error(f"Failed to fetch {url}: {error_msg}")
            return None, error_msg

        except requests.RequestException as e:
            error_msg = f"Request failed: {str(e)}"
            logger.error(f"Failed to fetch {url}: {error_msg}")
            return None, error_msg

        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            logger.error(f"Failed to fetch {url}: {error_msg}")
            return None, error_msg

    def scrape_and_extract_text(self, url: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        Scrape URL and extract text content (convenience method).

        Args:
            url: Web URL to scrape

        Returns:
            Tuple of (text_content, title, error_message)
            - text_content: Extracted text if successful, None otherwise
            - title: Page title if available, None otherwise
            - error_message: Error message if failed, None otherwise
        """
        html, error = self.scrape(url)

        if error:
            return None, None, error

        # Basic text extraction using BeautifulSoup
        try:
            soup = BeautifulSoup(html, 'html.parser')

            # Extract title
            title = None
            title_tag = soup.find('title')
            if title_tag:
                title = title_tag.get_text(strip=True)
            else:
                h1_tag = soup.find('h1')
                if h1_tag:
                    title = h1_tag.get_text(strip=True)

            # Remove script and style elements
            for element in soup(['script', 'style', 'nav', 'footer', 'header']):
                element.decompose()

            # Extract text from main content
            # Try to find main content area
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

            return text, title, None

        except Exception as e:
            error_msg = f"Text extraction failed: {str(e)}"
            logger.error(error_msg)
            return None, None, error_msg


def scrape_url(url: str, user_agent: str = "STIndex-Research/1.0") -> Tuple[Optional[str], Optional[str]]:
    """
    Convenience function to scrape a single URL.

    Args:
        url: Web URL to scrape
        user_agent: User agent string

    Returns:
        Tuple of (html_content, error_message)
    """
    scraper = WebScraper(user_agent=user_agent)
    return scraper.scrape(url)
