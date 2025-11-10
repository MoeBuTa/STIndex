"""
Post-processing module for STIndex.

Handles post-extraction processing:
- Temporal normalization (ISO 8601)
- Spatial geocoding
- Entity validation
"""

# Import only modules that exist
try:
    from stindex.postprocess.spatial.geocoder import GeocoderService
    __all__ = ["GeocoderService"]
except ImportError:
    __all__ = []

