"""
Enhanced geocoder with context-aware disambiguation and caching.

Based on research from:
- geoparsepy's evidential disambiguation approach
- "nearby parent region" and "nearby locations" strategies
- Performance optimization with caching

Includes Google Maps API fallback for better geocoding accuracy.
"""

import hashlib
import json
import os
import re
import time
from typing import Optional, Tuple, List, Dict
from pathlib import Path

from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
from loguru import logger

from stindex.utils.constants import (
    GEOCODE_CACHE_DIR,
    NOMINATIM_RATE_LIMIT,
    GEOCODER_REQUEST_TIMEOUT,
    DEFAULT_USER_AGENT,
)

# Google Maps API (optional)
try:
    import googlemaps
    GOOGLE_MAPS_AVAILABLE = True
except ImportError:
    GOOGLE_MAPS_AVAILABLE = False
    logger.debug("googlemaps package not found. Install with: pip install googlemaps for better geocoding")


class GeocodeCache:
    """Simple file-based cache for geocoding results."""

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize cache.

        Args:
            cache_dir: Directory to store cache files
        """
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = GEOCODE_CACHE_DIR

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "geocode_cache.json"

        # Load existing cache
        self.cache: Dict[str, Dict] = self._load_cache()

    def _load_cache(self) -> Dict[str, Dict]:
        """Load cache from file."""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save_cache(self):
        """Save cache to file."""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save geocode cache: {e}")

    def get(self, location: str, context: Optional[str] = None) -> Optional[Dict]:
        """
        Get cached result.

        Args:
            location: Location name
            context: Optional context for disambiguation

        Returns:
            Cached result or None
        """
        # Create cache key
        key = self._make_key(location, context)
        return self.cache.get(key)

    def set(self, location: str, result: Dict, context: Optional[str] = None):
        """
        Save result to cache.

        Args:
            location: Location name
            result: Geocoding result
            context: Optional context
        """
        key = self._make_key(location, context)
        self.cache[key] = result
        self._save_cache()

    def _make_key(self, location: str, context: Optional[str] = None) -> str:
        """Create cache key from location and context."""
        if context:
            key_str = f"{location}||{context}"
        else:
            key_str = location
        return hashlib.md5(key_str.lower().encode()).hexdigest()

    def clear(self):
        """Clear all cache."""
        self.cache = {}
        if self.cache_file.exists():
            self.cache_file.unlink()


class GeocoderService:
    """
    Enhanced geocoding service with context-aware disambiguation and caching.

    Improvements over basic GeocoderService:
    1. Context-aware disambiguation using nearby locations
    2. Caching for performance (avoid repeated API calls)
    3. Parent region hints for disambiguation
    4. Retry logic with exponential backoff
    5. Batch processing support
    """

    def __init__(
        self,
        user_agent: Optional[str] = None,
        enable_cache: bool = True,
        cache_dir: Optional[str] = None,
        rate_limit: Optional[float] = None,
        google_api_key: Optional[str] = None,
    ):
        """
        Initialize GeocoderService.

        Args:
            user_agent: User agent for Nominatim (uses default from constants if None)
            enable_cache: Enable caching
            cache_dir: Directory for cache files
            rate_limit: Minimum seconds between requests (uses default from constants if None)
            google_api_key: Google Maps API key (optional, for fallback geocoding)
        """
        user_agent = user_agent or DEFAULT_USER_AGENT
        rate_limit = rate_limit or NOMINATIM_RATE_LIMIT

        self.geolocator = Nominatim(user_agent=user_agent, timeout=GEOCODER_REQUEST_TIMEOUT)
        self.rate_limit = rate_limit
        self.last_request_time = 0

        # Caching
        self.enable_cache = enable_cache
        if enable_cache:
            self.cache = GeocodeCache(cache_dir)
        else:
            self.cache = None

        # Disambiguation context
        self.location_context: List[Tuple[float, float]] = []  # List of (lat, lon)

        # Load spaCy model for parent region extraction
        self._nlp = None  # Lazy load on first use

        # Initialize Google Maps API (optional fallback)
        self.gmaps_client = None
        if GOOGLE_MAPS_AVAILABLE:
            api_key = google_api_key or os.getenv('GOOGLE_MAPS_API_KEY')
            if api_key:
                try:
                    self.gmaps_client = googlemaps.Client(key=api_key)
                    logger.info("✓ Google Maps API enabled for geocoding fallback")
                except Exception as e:
                    logger.warning(f"Failed to initialize Google Maps API: {e}")
            else:
                logger.debug("Google Maps API key not provided (optional). Set GOOGLE_MAPS_API_KEY env var for better geocoding.")

    def _rate_limit_wait(self):
        """Enforce rate limiting."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self.last_request_time = time.time()

    @staticmethod
    def _extract_city_from_venue(venue_name: str) -> Optional[str]:
        """
        Extract city/location name from venue name.

        Examples:
        - "Margaret River Emergency Department" → "Margaret River"
        - "Broome Regional Hospital" → "Broome"
        - "Seattle Children's Hospital Emergency Department" → "Seattle"
        - "Perth CBD Medical Centre" → "Perth"

        Args:
            venue_name: Full venue name

        Returns:
            Extracted city name or None
        """
        if not venue_name:
            return None

        # Common venue type suffixes to remove
        suffixes = [
            r'\bEmergency Department\b',
            r'\bRegional Hospital\b',
            r'\bCommunity Hospital\b',
            r'\bMedical Centre\b',
            r'\bMedical Center\b',
            r'\bHealth Centre\b',
            r'\bHealth Center\b',
            r'\bClinic\b',
            r'\bHospital\b',
            r'\bCBD\b',
            r'\bDistrict\b',
            r"Children's\b",
        ]

        city_name = venue_name
        for suffix in suffixes:
            city_name = re.sub(suffix, '', city_name, flags=re.IGNORECASE)

        # Clean up
        city_name = city_name.strip()
        city_name = re.sub(r"'s\b", '', city_name)  # Remove possessive

        # Take first significant part (before commas or dashes)
        if city_name:
            parts = re.split(r'[,\-]', city_name)
            if parts and parts[0].strip():
                return parts[0].strip()

        return None

    def get_coordinates(
        self,
        location: str,
        context: Optional[str] = None,
        parent_region: Optional[str] = None,
    ) -> Optional[Tuple[float, float]]:
        """
        Get coordinates for a location with context-aware disambiguation and Google Maps fallback.

        Fallback strategy:
        1. Check cache first
        2. Try Nominatim with full location name
        3. If fails, extract city from venue name and try Nominatim
        4. If fails and Google Maps available, try Google Maps with full name
        5. If fails, try Google Maps with extracted city

        Args:
            location: Location name
            context: Surrounding text context
            parent_region: Parent region hint (e.g., "Western Australia")

        Returns:
            Tuple of (latitude, longitude) or None
        """
        # Check cache
        if self.enable_cache and self.cache:
            cached = self.cache.get(location, context)
            if cached:
                return (cached['lat'], cached['lon'])

        # Extract parent region from context if not provided
        context_parent_region = None
        if context:
            context_parent_region = self._extract_parent_region(context)

        # Prefer LLM-provided parent_region, but fallback to spaCy extraction
        final_parent_region = parent_region or context_parent_region

        # Prepare search query with disambiguation
        search_query = self._prepare_search_query(location, final_parent_region)

        # Level 1: Try Nominatim with full location name
        result = self._geocode_with_retry(search_query)

        if result:
            coords = (result.latitude, result.longitude)
            logger.debug(f"Geocoded '{location}' with Nominatim (full name)")
            self._cache_and_update_context(location, coords, result.address, context)
            return coords

        # Level 2: Extract city from venue name and try Nominatim
        city = self._extract_city_from_venue(location)
        if city and city != location:
            logger.debug(f"Extracted city '{city}' from venue '{location}'")
            city_query = self._prepare_search_query(city, final_parent_region)
            result = self._geocode_with_retry(city_query)

            if result:
                coords = (result.latitude, result.longitude)
                logger.info(f"✓ Geocoded '{location}' via city extraction → '{city}'")
                self._cache_and_update_context(location, coords, result.address, context)
                return coords

        # Level 3 & 4: Try Google Maps API if available
        if self.gmaps_client:
            coords = self._try_google_maps(location, city, final_parent_region)
            if coords:
                self._cache_and_update_context(location, coords, None, context)
                return coords

        logger.warning(f"All geocoding attempts failed for: {location}")
        return None

    def _try_google_maps(
        self,
        location: str,
        city: Optional[str],
        parent_region: Optional[str]
    ) -> Optional[Tuple[float, float]]:
        """
        Try geocoding with Google Maps API.

        Args:
            location: Full location name
            city: Extracted city name (if available)
            parent_region: Parent region hint

        Returns:
            Tuple of (latitude, longitude) or None
        """
        # Determine region bias from parent_region
        region_code = None
        if parent_region:
            parent_lower = parent_region.lower()
            if 'australia' in parent_lower or 'western australia' in parent_lower:
                region_code = 'au'
            elif 'usa' in parent_lower or 'united states' in parent_lower or 'washington state' in parent_lower:
                region_code = 'us'

        # Try with full location name first
        try:
            query = self._prepare_search_query(location, parent_region)
            logger.debug(f"Google Maps geocoding: {query}")
            results = self.gmaps_client.geocode(query, region=region_code)

            if results and len(results) > 0:
                lat = results[0]['geometry']['location']['lat']
                lon = results[0]['geometry']['location']['lng']
                logger.info(f"✓ Geocoded '{location}' with Google Maps (full name)")
                return (lat, lon)
        except Exception as e:
            logger.debug(f"Google Maps failed for full name '{location}': {e}")

        # Try with extracted city if available
        if city and city != location:
            try:
                query = self._prepare_search_query(city, parent_region)
                logger.debug(f"Google Maps geocoding city: {query}")
                results = self.gmaps_client.geocode(query, region=region_code)

                if results and len(results) > 0:
                    lat = results[0]['geometry']['location']['lat']
                    lon = results[0]['geometry']['location']['lng']
                    logger.info(f"✓ Geocoded '{location}' with Google Maps via city '{city}'")
                    return (lat, lon)
            except Exception as e:
                logger.debug(f"Google Maps failed for city '{city}': {e}")

        return None

    def _cache_and_update_context(
        self,
        location: str,
        coords: Tuple[float, float],
        address: Optional[str],
        context: Optional[str]
    ):
        """Cache result and update location context."""
        # Cache result
        if self.enable_cache and self.cache:
            self.cache.set(
                location,
                {'lat': coords[0], 'lon': coords[1], 'address': address or ''},
                context
            )

        # Update location context for future disambiguations
        self.location_context.append(coords)
        # Keep only recent 10 locations
        if len(self.location_context) > 10:
            self.location_context.pop(0)

    def _extract_parent_region(self, context: str) -> Optional[str]:
        """
        Extract parent region hints from context using spaCy NER.

        Uses spaCy to identify GPE (Geo-Political Entity) tags like countries,
        states, and provinces. This is more robust than hardcoded regex patterns.

        Strategy:
        1. Use spaCy NER to find GPE entities
        2. Also look for comma-separated patterns (e.g., "City, Region")
        3. Return the broadest/last region found

        Examples:
        - "Broome, Western Australia" → "Western Australia"
        - "Tokyo, Japan" → "Japan"
        - "Paris, France" → "France"

        Returns:
            Parent region name or None if not found
        """
        # Lazy load spaCy model
        if self._nlp is None:
            try:
                import spacy
                self._nlp = spacy.load("en_core_web_sm")
            except (ImportError, OSError):
                # If spaCy not available, fall back to simple comma parsing
                return self._extract_from_comma_pattern(context)

        # Process context with spaCy
        doc = self._nlp(context)

        # Extract GPE entities (countries, states, cities, etc.)
        gpe_entities = [ent.text for ent in doc.ents if ent.label_ == "GPE"]

        # Also try comma-separated pattern as fallback for multi-word regions
        # spaCy sometimes misses multi-word regions like "Western Australia"
        comma_region = self._extract_from_comma_pattern(context)

        # Prefer comma-separated pattern if it looks like a region (contains spaces or is longer)
        if comma_region and (
            " " in comma_region or  # Multi-word region
            (gpe_entities and len(comma_region) > len(gpe_entities[-1]))  # Longer than last GPE
        ):
            return comma_region

        # Otherwise return the last GPE entity (usually most relevant)
        if gpe_entities:
            return gpe_entities[-1]

        # Final fallback to comma pattern
        return comma_region

    def _extract_from_comma_pattern(self, context: str) -> Optional[str]:
        """
        Extract parent region from common location patterns.

        Patterns supported:
        - Comma-separated: "Broome, Western Australia" → "Western Australia"
        - Preposition-based: "Broome in Western Australia" → "Western Australia"
        - Multi-level: "New York, NY, USA" → "USA"
        """
        import re

        # Pattern 1: Comma-separated (", Region")
        # This is most reliable, so prioritize it
        comma_pattern = r',\s*([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)'
        comma_matches = re.findall(comma_pattern, context)

        if comma_matches:
            # Return the last comma-separated match (broadest region)
            return comma_matches[-1]

        # Pattern 2: Preposition-based ("in Region", "at Region", "near Region")
        # Only use this as fallback when no comma pattern found
        prep_pattern = r'\b(?:in|at|near)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)'
        prep_matches = re.findall(prep_pattern, context)

        if prep_matches:
            # Return the last preposition-based match
            return prep_matches[-1]

        return None

    def _prepare_search_query(
        self,
        location: str,
        parent_region: Optional[str] = None
    ) -> str:
        """
        Prepare search query with parent region hint.

        Args:
            location: Location name
            parent_region: Parent region hint

        Returns:
            Enhanced search query
        """
        if parent_region:
            # If location already contains parent region, don't duplicate
            if parent_region.lower() in location.lower():
                return location
            else:
                return f"{location}, {parent_region}"
        return location

    def _geocode_with_retry(
        self,
        query: str,
        max_retries: int = 3
    ) -> Optional[any]:
        """
        Geocode with exponential backoff retry.

        Args:
            query: Search query
            max_retries: Maximum number of retries

        Returns:
            Geopy location object or None
        """
        for attempt in range(max_retries):
            try:
                self._rate_limit_wait()
                result = self.geolocator.geocode(query)
                return result
            except GeocoderTimedOut:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    time.sleep(wait_time)
                    continue
                else:
                    return None
            except GeocoderServiceError:
                return None

        return None

    def _apply_nearby_scoring(self, result: any, query: str) -> Optional[any]:
        """
        Apply nearby location scoring for disambiguation.

        Strategy: If we have recent locations in context, prefer results
        closer to them (geoparsepy's "nearby locations" strategy).

        Args:
            result: Initial geocoding result
            query: Search query

        Returns:
            Best result after scoring
        """
        if not self.location_context:
            return result

        # Get multiple results for disambiguation
        self._rate_limit_wait()
        try:
            results = self.geolocator.geocode(query, exactly_one=False, limit=5)
            if not results:
                return result

            # Score based on distance to context locations
            best_result = results[0]
            best_score = float('inf')

            for candidate in results:
                # Calculate average distance to context locations
                avg_distance = self._calculate_avg_distance(
                    (candidate.latitude, candidate.longitude),
                    self.location_context
                )

                if avg_distance < best_score:
                    best_score = avg_distance
                    best_result = candidate

            return best_result
        except Exception:
            return result

    def _calculate_avg_distance(
        self,
        point: Tuple[float, float],
        context_points: List[Tuple[float, float]]
    ) -> float:
        """
        Calculate average distance to context points.

        Uses Haversine distance for geographical coordinates.

        Args:
            point: (lat, lon) to score
            context_points: List of (lat, lon) context locations

        Returns:
            Average distance in kilometers
        """
        from math import radians, cos, sin, asin, sqrt

        def haversine(lat1, lon1, lat2, lon2):
            """Calculate Haversine distance between two points."""
            lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            c = 2 * asin(sqrt(a))
            km = 6371 * c  # Radius of earth in kilometers
            return km

        if not context_points:
            return float('inf')

        distances = [
            haversine(point[0], point[1], ctx[0], ctx[1])
            for ctx in context_points
        ]

        return sum(distances) / len(distances)

    def geocode_batch(
        self,
        locations: List[Tuple[str, Optional[str]]],
        use_context: bool = True
    ) -> List[Optional[Tuple[float, float]]]:
        """
        Geocode multiple locations efficiently with shared context.

        Args:
            locations: List of (location, context) tuples
            use_context: Use contextual disambiguation

        Returns:
            List of (lat, lon) tuples or None
        """
        results = []

        # Reset context for new batch
        if not use_context:
            self.location_context = []

        for location, context in locations:
            # Extract parent region from context
            parent_region = self._extract_parent_region(context) if context else None

            coords = self.get_coordinates(location, context, parent_region)
            results.append(coords)

        return results

    def clear_cache(self):
        """Clear geocoding cache."""
        if self.cache:
            self.cache.clear()

    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        if self.cache:
            return {
                'total_entries': len(self.cache.cache),
                'cache_size_kb': self.cache.cache_file.stat().st_size // 1024 if self.cache.cache_file.exists() else 0
            }
        return {'total_entries': 0, 'cache_size_kb': 0}
