"""
STIndex Data Warehouse Module.

Provides two warehouse implementations:

1. FileBasedWarehouse (Recommended for most use cases)
   - Simple file-based storage (JSON, Parquet, GeoJSON)
   - No database server required
   - SQL queries via DuckDB
   - Easy to version control and share

2. DimensionalWarehouseETL (Legacy/Advanced)
   - PostgreSQL-based warehouse
   - Requires database setup
   - Best for production with high query volumes

Quick Start:
    from stindex.warehouse import FileBasedWarehouse, STIndexQueryEngine

    # Save extraction results
    warehouse = FileBasedWarehouse("data/warehouse")
    warehouse.save_document(
        document_id="doc_001",
        extraction_results=results,
        document_metadata={"title": "Report"},
    )

    # Query with SQL
    engine = STIndexQueryEngine("data/warehouse")
    df = engine.query("SELECT * FROM chunks WHERE temporal_year = 2022").to_dataframe()

    # Or use fluent API
    results = (
        engine.select()
        .where_temporal(year=2022)
        .where_spatial(region="Australia")
        .execute()
        .to_dicts()
    )
"""

# File-based warehouse (recommended)
from stindex.warehouse.file_store import FileBasedWarehouse
from stindex.warehouse.query_engine import (
    STIndexQueryEngine,
    QueryBuilder,
    QueryResult,
    PandasQueryEngine,
)

# Chunk labeling (used by both implementations)
from stindex.warehouse.chunk_labeler import (
    ChunkDimensionalLabels,
    DimensionalChunkLabeler,
)

# Legacy PostgreSQL-based ETL (optional import to avoid psycopg2 dependency)
try:
    from stindex.warehouse.etl import DimensionalWarehouseETL
except ImportError:
    DimensionalWarehouseETL = None  # psycopg2 not installed

__all__ = [
    # File-based warehouse (primary)
    "FileBasedWarehouse",
    "STIndexQueryEngine",
    "QueryBuilder",
    "QueryResult",
    "PandasQueryEngine",
    # Chunk labeling
    "ChunkDimensionalLabels",
    "DimensionalChunkLabeler",
    # Legacy PostgreSQL ETL
    "DimensionalWarehouseETL",
]

__version__ = "1.0.0"
