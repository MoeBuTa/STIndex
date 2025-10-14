"""
Enhanced geocoder with context-aware disambiguation and caching.

Based on research from:
- geoparsepy's evidential disambiguation approach
- "nearby parent region" and "nearby locations" strategies
- Performance optimization with caching
"""

import hashlib
import json
import time
from typing import Optional, Tuple, List, Dict
from pathlib import Path

from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError


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
            self.cache_dir = Path.home() / ".stindex" / "geocode_cache"

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


class EnhancedGeocoderService:
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
        user_agent: str = "stindex-enhanced",
        enable_cache: bool = True,
        cache_dir: Optional[str] = None,
        rate_limit: float = 1.0,
    ):
        """
        Initialize EnhancedGeocoderService.

        Args:
            user_agent: User agent for Nominatim
            enable_cache: Enable caching
            cache_dir: Directory for cache files
            rate_limit: Minimum seconds between requests
        """
        self.geolocator = Nominatim(user_agent=user_agent, timeout=10)
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

    def _rate_limit_wait(self):
        """Enforce rate limiting."""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.rate_limit:
            time.sleep(self.rate_limit - elapsed)
        self.last_request_time = time.time()

    def get_coordinates(
        self,
        location: str,
        context: Optional[str] = None,
        parent_region: Optional[str] = None,
    ) -> Optional[Tuple[float, float]]:
        """
        Get coordinates for a location with context-aware disambiguation.

        Disambiguation strategy (inspired by geoparsepy):
        1. Check cache first
        2. If parent region provided, search within that region
        3. If context locations exist, prefer nearby results
        4. Use Nominatim's ranking as fallback

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
        if not parent_region and context:
            parent_region = self._extract_parent_region(context)

        # Prepare search query with disambiguation
        search_query = self._prepare_search_query(location, parent_region)

        # Geocode with retry
        result = self._geocode_with_retry(search_query)

        if result:
            coords = (result.latitude, result.longitude)

            # Apply nearby location scoring if we have context
            if self.location_context:
                result = self._apply_nearby_scoring(result, search_query)
                if result:
                    coords = (result.latitude, result.longitude)

            # Cache result
            if self.enable_cache and self.cache:
                self.cache.set(
                    location,
                    {'lat': coords[0], 'lon': coords[1], 'address': result.address},
                    context
                )

            # Update location context for future disambiguations
            self.location_context.append(coords)
            # Keep only recent 10 locations
            if len(self.location_context) > 10:
                self.location_context.pop(0)

            return coords

        return None

    def _extract_parent_region(self, context: str) -> Optional[str]:
        """
        Extract parent region hints from context.

        Examples:
        - "Broome, Western Australia" → "Western Australia"
        - "Tokyo, Japan" → "Japan"
        - "Paris, France" → "France"
        """
        # Common patterns for parent regions
        region_patterns = [
            # State/Province patterns
            r',\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',  # ", Western Australia"
            # Country patterns
            r'\b(Australia|USA|United States|UK|United Kingdom|Canada|China|Japan|India|France|Germany|Italy|Spain)\b',
        ]

        import re
        for pattern in region_patterns:
            match = re.search(pattern, context)
            if match:
                return match.group(1)

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
