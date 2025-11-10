"""
STIndex Data Warehouse Module.

Dimensional data warehouse for multi-dimensional extraction results.
Combines traditional dimensional modeling with modern capabilities:
- Vector embeddings for semantic search
- PostGIS for spatial queries
- Hierarchical dimensions for analytics
"""

from stindex.warehouse.chunk_labeler import (
    ChunkDimensionalLabels,
    DimensionalChunkLabeler,
)
from stindex.warehouse.etl import DimensionalWarehouseETL

__all__ = [
    "ChunkDimensionalLabels",
    "DimensionalChunkLabeler",
    "DimensionalWarehouseETL",
]

__version__ = "0.6.0"
