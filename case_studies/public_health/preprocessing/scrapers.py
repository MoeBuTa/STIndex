"""
Web scrapers for public health surveillance data.

Scrapes health alerts from official sources:
- WA Health (Western Australia)
- Washington State DOH (USA)
- Australian health statistics
"""

import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from loguru import logger


@dataclass
class HealthAlert:
    """Container for scraped health alert data."""

    title: str
    url: str
    source: str  # "wa_health_au", "wa_doh_us", etc.
    publication_date: Optional[str]  # ISO 8601
    content: str
    raw_html: str
    metadata: dict
    scraped_at: str  # ISO 8601 timestamp


class HealthAlertScraper:
    """Base scraper class for health alerts."""

    def __init__(self, output_dir: str = "case_studies/public_health/data/raw"):
        """
        Initialize scraper.

        Args:
            output_dir: Directory to save raw scraped data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Polite scraping: user agent and rate limiting
        self.headers = {
            "User-Agent": "STIndex-Research-Bot/1.0 (Public Health Case Study; Contact: research@example.com)"
        }
        self.rate_limit = 2.0  # seconds between requests
        self.last_request_time = 0

    def _rate_limit_wait(self):
        """Respect rate limiting."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self.last_request_time = time.time()

    def _get_page(self, url: str) -> Optional[str]:
        """
        Fetch a web page with rate limiting.

        Args:
            url: URL to fetch

        Returns:
            HTML content or None on failure
        """
        self._rate_limit_wait()

        try:
            logger.info(f"Fetching: {url}")
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            logger.error(f"Failed to fetch {url}: {e}")
            return None

    def save_alerts(self, alerts: List[HealthAlert], filename: str):
        """
        Save scraped alerts to JSON file.

        Args:
            alerts: List of HealthAlert objects
            filename: Output filename (relative to output_dir)
        """
        output_path = self.output_dir / filename

        # Convert to dict for JSON serialization
        alerts_data = [asdict(alert) for alert in alerts]

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(alerts_data, f, indent=2, ensure_ascii=False)

        logger.info(f"✓ Saved {len(alerts)} alerts to {output_path}")


class WAHealthAustraliaScraper(HealthAlertScraper):
    """Scraper for WA Health (Western Australia) alerts."""

    BASE_URL = "https://www.health.wa.gov.au"

    def scrape_measles_alert(self) -> List[HealthAlert]:
        """
        Scrape the 2025 measles alert page.

        Returns:
            List of HealthAlert objects
        """
        url = f"{self.BASE_URL}/news/2025/measles-alert"
        html = self._get_page(url)

        if not html:
            return []

        soup = BeautifulSoup(html, 'html.parser')
        alerts = []

        # Extract main content
        main_content = soup.find('div', class_='field-name-body') or soup.find('article')

        if main_content:
            # Extract title
            title_elem = soup.find('h1')
            title = title_elem.get_text(strip=True) if title_elem else "Measles Alert"

            # Extract publication date (if available)
            date_elem = soup.find('time') or soup.find('span', class_='date')
            pub_date = None
            if date_elem:
                try:
                    date_text = date_elem.get('datetime') or date_elem.get_text(strip=True)
                    # Try parsing common date formats
                    pub_date = self._parse_date(date_text)
                except Exception as e:
                    logger.warning(f"Failed to parse date: {e}")

            # Extract text content
            content = main_content.get_text(separator='\n', strip=True)

            # Create alert
            alert = HealthAlert(
                title=title,
                url=url,
                source="wa_health_au",
                publication_date=pub_date,
                content=content,
                raw_html=str(main_content),
                metadata={
                    "region": "Western Australia",
                    "country": "Australia",
                    "disease": "measles",
                },
                scraped_at=datetime.now().isoformat()
            )
            alerts.append(alert)

        return alerts

    def scrape_infectious_disease_alerts(self) -> List[HealthAlert]:
        """
        Scrape infectious disease alerts list page.

        Returns:
            List of HealthAlert objects
        """
        url = f"{self.BASE_URL}/Articles/F_I/Health-alerts-infectious-diseases"
        html = self._get_page(url)

        if not html:
            return []

        soup = BeautifulSoup(html, 'html.parser')
        alerts = []

        # Find all alert links
        alert_links = soup.find_all('a', href=True)

        for link in alert_links:
            href = link.get('href')
            # Filter for PDF alerts
            if href and '.pdf' in href.lower():
                pdf_url = urljoin(self.BASE_URL, href)
                link_text = link.get_text(strip=True)

                # Create metadata alert (PDF content would need separate processing)
                alert = HealthAlert(
                    title=link_text,
                    url=pdf_url,
                    source="wa_health_au",
                    publication_date=None,  # Would need to parse from filename/text
                    content="",  # PDF content requires separate extraction
                    raw_html=str(link),
                    metadata={
                        "region": "Western Australia",
                        "country": "Australia",
                        "format": "pdf",
                    },
                    scraped_at=datetime.now().isoformat()
                )
                alerts.append(alert)

        return alerts

    def _parse_date(self, date_str: str) -> Optional[str]:
        """Parse date string to ISO 8601."""
        # Try common formats
        formats = [
            "%Y-%m-%d",
            "%d/%m/%Y",
            "%d %B %Y",
            "%B %d, %Y",
        ]

        for fmt in formats:
            try:
                dt = datetime.strptime(date_str.strip(), fmt)
                return dt.date().isoformat()
            except ValueError:
                continue

        return None


class WADOHUSAScraper(HealthAlertScraper):
    """Scraper for Washington State Department of Health (USA)."""

    BASE_URL = "https://doh.wa.gov"

    def scrape_measles_cases_2025(self) -> List[HealthAlert]:
        """
        Scrape 2025 measles cases page.

        Returns:
            List of HealthAlert objects
        """
        url = f"{self.BASE_URL}/you-and-your-family/illness-and-disease-z/measles/measles-cases-washington-state-2025"
        html = self._get_page(url)

        if not html:
            return []

        soup = BeautifulSoup(html, 'html.parser')
        alerts = []

        # Extract main content
        main_content = soup.find('main') or soup.find('article')

        if main_content:
            title_elem = soup.find('h1')
            title = title_elem.get_text(strip=True) if title_elem else "Measles Cases - Washington State 2025"

            content = main_content.get_text(separator='\n', strip=True)

            alert = HealthAlert(
                title=title,
                url=url,
                source="wa_doh_us",
                publication_date=None,  # Would need to extract from page
                content=content,
                raw_html=str(main_content),
                metadata={
                    "region": "Washington State",
                    "country": "United States",
                    "disease": "measles",
                },
                scraped_at=datetime.now().isoformat()
            )
            alerts.append(alert)

        return alerts


class AustralianInfluenzaScraper(HealthAlertScraper):
    """Scraper for Australian influenza statistics."""

    BASE_URL = "https://immunisationcoalition.org.au"

    def scrape_influenza_stats(self) -> List[HealthAlert]:
        """
        Scrape influenza statistics page.

        Returns:
            List of HealthAlert objects
        """
        url = f"{self.BASE_URL}/influenza-statistics/"
        html = self._get_page(url)

        if not html:
            return []

        soup = BeautifulSoup(html, 'html.parser')
        alerts = []

        main_content = soup.find('main') or soup.find('article')

        if main_content:
            title = "Australian Influenza Statistics 2025"
            content = main_content.get_text(separator='\n', strip=True)

            alert = HealthAlert(
                title=title,
                url=url,
                source="immunisation_coalition_au",
                publication_date=None,
                content=content,
                raw_html=str(main_content),
                metadata={
                    "region": "Australia",
                    "country": "Australia",
                    "disease": "influenza",
                },
                scraped_at=datetime.now().isoformat()
            )
            alerts.append(alert)

        return alerts


def scrape_all_health_alerts(output_dir: str = "case_studies/public_health/data/raw"):
    """
    Scrape all configured health alert sources.

    Args:
        output_dir: Directory to save scraped data
    """
    logger.info("Starting health alert scraping...")

    # WA Health Australia
    wa_au_scraper = WAHealthAustraliaScraper(output_dir)
    measles_alerts_au = wa_au_scraper.scrape_measles_alert()
    wa_au_scraper.save_alerts(measles_alerts_au, "wa_health_au_measles.json")

    infectious_alerts_au = wa_au_scraper.scrape_infectious_disease_alerts()
    wa_au_scraper.save_alerts(infectious_alerts_au, "wa_health_au_infectious.json")

    # WA DOH USA
    wa_us_scraper = WADOHUSAScraper(output_dir)
    measles_alerts_us = wa_us_scraper.scrape_measles_cases_2025()
    wa_us_scraper.save_alerts(measles_alerts_us, "wa_doh_us_measles.json")

    # Australian Influenza
    flu_scraper = AustralianInfluenzaScraper(output_dir)
    flu_alerts = flu_scraper.scrape_influenza_stats()
    flu_scraper.save_alerts(flu_alerts, "au_influenza_stats.json")

    logger.info("✓ Scraping complete!")


if __name__ == "__main__":
    # Run scrapers
    scrape_all_health_alerts()
