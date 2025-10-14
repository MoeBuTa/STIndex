"""
Geocoding utilities with caching and rate limiting.
"""

import time
from functools import lru_cache
from typing import Optional, Tuple

from geopy.exc import GeocoderTimedOut, GeocoderUnavailable
from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import Nominatim


class GeocoderService:
    """Geocoding service with caching and rate limiting."""

    def __init__(
        self,
        provider: str = "nominatim",
        user_agent: str = "stindex",
        rate_limit_calls: int = 1,
        rate_limit_period: float = 1.0,
        timeout: int = 10,
    ):
        """
        Initialize geocoder service.

        Args:
            provider: Geocoding provider (currently only 'nominatim' supported)
            user_agent: User agent string for API requests
            rate_limit_calls: Number of calls allowed per period
            rate_limit_period: Time period in seconds for rate limiting
            timeout: Request timeout in seconds
        """
        self.provider = provider
        self.user_agent = user_agent
        self.timeout = timeout

        # Initialize geocoder
        if provider == "nominatim":
            self.geocoder = Nominatim(user_agent=user_agent, timeout=timeout)
        else:
            raise ValueError(f"Unsupported geocoding provider: {provider}")

        # Apply rate limiting
        self.geocode = RateLimiter(
            self.geocoder.geocode,
            min_delay_seconds=rate_limit_period / rate_limit_calls,
            max_retries=3,
            error_wait_seconds=5.0,
        )

        self.reverse_geocode = RateLimiter(
            self.geocoder.reverse,
            min_delay_seconds=rate_limit_period / rate_limit_calls,
            max_retries=3,
            error_wait_seconds=5.0,
        )

    @lru_cache(maxsize=1000)
    def get_coordinates(
        self, location_name: str, context: Optional[str] = None
    ) -> Optional[Tuple[float, float]]:
        """
        Get coordinates for a location name.

        Args:
            location_name: Location name to geocode
            context: Additional context to help disambiguation

        Returns:
            Tuple of (latitude, longitude) or None if not found
        """
        try:
            # Combine location with context if provided
            query = f"{location_name}, {context}" if context else location_name

            location = self.geocode(query)

            if location:
                return (location.latitude, location.longitude)
            return None

        except (GeocoderTimedOut, GeocoderUnavailable) as e:
            print(f"Geocoding error for '{location_name}': {e}")
            return None
        except Exception as e:
            print(f"Unexpected error geocoding '{location_name}': {e}")
            return None

    @lru_cache(maxsize=1000)
    def get_location_details(self, location_name: str) -> Optional[dict]:
        """
        Get detailed location information.

        Args:
            location_name: Location name to geocode

        Returns:
            Dictionary with location details or None
        """
        try:
            location = self.geocode(location_name)

            if location:
                # Extract address components
                address = location.raw.get("address", {})

                return {
                    "latitude": location.latitude,
                    "longitude": location.longitude,
                    "address": location.address,
                    "country": address.get("country"),
                    "country_code": address.get("country_code"),
                    "state": address.get("state"),
                    "county": address.get("county"),
                    "city": address.get("city") or address.get("town") or address.get("village"),
                    "postcode": address.get("postcode"),
                    "display_name": location.address,
                }

            return None

        except Exception as e:
            print(f"Error getting details for '{location_name}': {e}")
            return None

    @lru_cache(maxsize=500)
    def reverse_geocode_coords(
        self, latitude: float, longitude: float
    ) -> Optional[dict]:
        """
        Reverse geocode coordinates to location name.

        Args:
            latitude: Latitude coordinate
            longitude: Longitude coordinate

        Returns:
            Dictionary with location details or None
        """
        try:
            location = self.reverse_geocode(f"{latitude}, {longitude}")

            if location:
                address = location.raw.get("address", {})

                return {
                    "address": location.address,
                    "country": address.get("country"),
                    "state": address.get("state"),
                    "city": address.get("city") or address.get("town"),
                }

            return None

        except Exception as e:
            print(f"Error reverse geocoding ({latitude}, {longitude}): {e}")
            return None

    def geocode_batch(
        self, locations: list[str], delay: float = 1.0
    ) -> dict[str, Optional[Tuple[float, float]]]:
        """
        Geocode multiple locations with rate limiting.

        Args:
            locations: List of location names
            delay: Delay between requests in seconds

        Returns:
            Dictionary mapping location names to coordinates
        """
        results = {}

        for location in locations:
            coords = self.get_coordinates(location)
            results[location] = coords

            # Add delay to respect rate limits
            if delay > 0:
                time.sleep(delay)

        return results

    def clear_cache(self):
        """Clear the geocoding cache."""
        self.get_coordinates.cache_clear()
        self.get_location_details.cache_clear()
        self.reverse_geocode_coords.cache_clear()
