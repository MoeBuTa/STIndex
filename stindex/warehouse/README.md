# STIndex Data Warehouse

## Overview

The STIndex Data Warehouse provides storage and querying for multi-dimensional extraction results. It supports two implementations:

### 1. File-Based Warehouse (Recommended)

Simple, portable storage using JSON, Parquet, and GeoJSON:
- **No database server required** - all data in files
- **SQL queries via DuckDB** - full SQL support without setup
- **Easy to share** - just zip the directory
- **Version control friendly** - JSON files can be tracked in git

### 2. PostgreSQL Warehouse (Legacy/Advanced)

For production deployments with high query volumes:
- PostgreSQL + pgvector + PostGIS
- Requires database setup
- Better for millions of records

## Quick Start (File-Based)

### Installation

```bash
# Core dependencies (already included in stindex)
pip install stindex

# Optional: For SQL queries (recommended)
pip install duckdb

# Optional: For DataFrame operations
pip install pandas pyarrow
```

### Basic Usage

```python
from stindex import DimensionalExtractor
from stindex.warehouse import FileBasedWarehouse, STIndexQueryEngine

# 1. Extract dimensional information
extractor = DimensionalExtractor(config_path="extract")
results = []

texts = [
    "On March 15, 2022, a cyclone hit Broome, Western Australia.",
    "The storm caused flooding in Perth on March 16, 2022."
]

for text in texts:
    result = extractor.extract(text)
    results.append(result)

# 2. Save to warehouse
warehouse = FileBasedWarehouse("data/warehouse")
warehouse.save_document(
    document_id="cyclone_report_001",
    extraction_results=results,
    document_metadata={
        "title": "Cyclone Report",
        "source": "News Agency",
        "publication_date": "2022-03-17",
    }
)

# 3. Query with SQL
engine = STIndexQueryEngine("data/warehouse")

# SQL query
df = engine.query("""
    SELECT temporal_year, COUNT(*) as count
    FROM chunks
    GROUP BY temporal_year
""").to_dataframe()

print(df)
```

### Fluent Query API

```python
from stindex.warehouse import STIndexQueryEngine

engine = STIndexQueryEngine("data/warehouse")

# Filter by temporal and spatial dimensions
results = (
    engine.select("chunk_id", "temporal_normalized", "spatial_text", "chunk_text")
    .where_temporal(year=2022, quarter=1)
    .where_spatial(region="Australia")
    .order_by("temporal_normalized")
    .limit(100)
    .execute()
    .to_dicts()
)

for r in results:
    print(f"{r['temporal_normalized']}: {r['spatial_text']}")
```

### Spatial Queries

```python
# Find events within 100km of Broome
nearby = engine.spatial_query(
    center=(122.2, -18.0),  # (longitude, latitude)
    radius_km=100
)

for event in nearby:
    print(f"{event['spatial_text']}: {event['distance_km']:.1f} km away")
```

### Export to Different Formats

```python
warehouse = FileBasedWarehouse("data/warehouse")

# Export to Parquet (fast queries)
warehouse.export_to_parquet("data/export/chunks.parquet")

# Export to CSV (Excel-compatible)
warehouse.export_to_csv("data/export/chunks.csv")

# Export to GeoJSON (maps)
warehouse.export_geojson("data/export/events.geojson")
```

## Storage Structure

```
data/warehouse/
├── documents/              # Per-document extraction results
│   ├── doc_001.json
│   └── doc_002.json
├── chunks.jsonl            # All chunks (JSON Lines format)
├── chunks.parquet          # Optional: Parquet for fast queries
├── events.geojson          # Spatial events for maps
├── indexes/                # Inverted indexes for fast filtering
│   ├── temporal_index.json
│   └── spatial_index.json
└── metadata.json           # Schema version and statistics
```

## File Formats

### chunks.jsonl (JSON Lines)

Each line is a JSON object:
```json
{
  "chunk_id": "doc_001_0000",
  "document_id": "doc_001",
  "chunk_index": 0,
  "chunk_text": "On March 15, 2022, a cyclone hit Broome...",
  "temporal_normalized": "2022-03-15",
  "temporal_year": 2022,
  "temporal_quarter": 1,
  "temporal_month": 3,
  "temporal_labels": ["2022", "2022-Q1", "2022-03", "2022-03-15"],
  "spatial_text": "Broome",
  "latitude": -17.9614,
  "longitude": 122.2359,
  "spatial_labels": ["Australia", "Western Australia", "Broome"],
  "dimensions": {},
  "confidence_score": 0.95
}
```

### events.geojson

Standard GeoJSON FeatureCollection:
```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Point",
        "coordinates": [122.2359, -17.9614]
      },
      "properties": {
        "id": 0,
        "chunk_id": "doc_001_0000",
        "timestamp": "2022-03-15",
        "location": "Broome"
      }
    }
  ]
}
```

## SQL Query Examples

### Basic Queries

```sql
-- All chunks from 2022
SELECT * FROM chunks WHERE temporal_year = 2022;

-- Chunks in Australia
SELECT * FROM chunks WHERE list_contains(spatial_labels, 'Australia');

-- Count by year
SELECT temporal_year, COUNT(*) as count
FROM chunks
GROUP BY temporal_year
ORDER BY temporal_year;
```

### Temporal Queries

```sql
-- Specific quarter
SELECT * FROM chunks
WHERE temporal_year = 2022 AND temporal_quarter = 1;

-- Date range
SELECT * FROM chunks
WHERE temporal_normalized BETWEEN '2022-01-01' AND '2022-06-30';

-- Monthly trend
SELECT
    temporal_year,
    temporal_month,
    COUNT(*) as events
FROM chunks
WHERE temporal_year = 2022
GROUP BY temporal_year, temporal_month
ORDER BY temporal_month;
```

### Spatial Queries

```sql
-- By region
SELECT * FROM chunks
WHERE list_contains(spatial_labels, 'Western Australia');

-- With coordinates
SELECT * FROM chunks
WHERE latitude IS NOT NULL AND longitude IS NOT NULL;

-- Distance calculation (Haversine)
SELECT
    *,
    6371 * 2 * ASIN(SQRT(
        POWER(SIN(RADIANS(latitude - (-18.0)) / 2), 2) +
        COS(RADIANS(-18.0)) * COS(RADIANS(latitude)) *
        POWER(SIN(RADIANS(longitude - 122.2) / 2), 2)
    )) as distance_km
FROM chunks
WHERE latitude IS NOT NULL
ORDER BY distance_km
LIMIT 10;
```

### Aggregations

```sql
-- Spatial distribution
SELECT
    spatial_labels[1] as country,
    COUNT(*) as chunks
FROM chunks
WHERE len(spatial_labels) > 0
GROUP BY spatial_labels[1]
ORDER BY chunks DESC;

-- Cross-tabulation
SELECT
    temporal_year,
    spatial_labels[1] as country,
    COUNT(*) as events
FROM chunks
WHERE temporal_year IS NOT NULL AND len(spatial_labels) > 0
GROUP BY temporal_year, spatial_labels[1]
ORDER BY temporal_year, events DESC;
```

## Configuration

See `cfg/warehouse.yml`:

```yaml
storage:
  base_dir: "data/warehouse"
  primary_format: "jsonl"  # or "parquet"
  auto_export_parquet: false

query:
  engine: "duckdb"
  duckdb:
    memory_limit: "2GB"
    threads: 4
```

## Pipeline Integration

```python
from stindex import InputDocument, STIndexPipeline
from stindex.warehouse import FileBasedWarehouse

# Run extraction pipeline
pipeline = STIndexPipeline(
    dimension_config="dimensions",
    output_dir="data/output",
)

docs = [InputDocument.from_text("Your text here...")]
results = pipeline.run_pipeline(docs)

# Save to warehouse
warehouse = FileBasedWarehouse("data/warehouse")
for doc_result in results:
    warehouse.save_document(
        document_id=doc_result['document_id'],
        extraction_results=doc_result['extraction_results'],
        document_metadata=doc_result['metadata'],
    )
```

## Comparison: File-Based vs PostgreSQL

| Feature | File-Based | PostgreSQL |
|---------|-----------|------------|
| Setup | None | Database server |
| Dependencies | Optional (duckdb) | psycopg2, PostGIS |
| Query Language | SQL (DuckDB) | SQL (PostgreSQL) |
| Spatial Queries | Yes (basic) | Yes (advanced) |
| Vector Search | No | Yes (pgvector) |
| Scalability | ~1M chunks | ~100M chunks |
| Portability | Excellent | Requires DB |
| Version Control | Yes (JSON) | No |

## When to Use PostgreSQL

Consider PostgreSQL warehouse when you need:
- Vector similarity search (semantic search)
- Advanced spatial queries (PostGIS)
- High query concurrency
- Data larger than ~1M chunks
- Production deployment with multiple users

See `stindex/warehouse/etl.py` for PostgreSQL implementation.

## Troubleshooting

### DuckDB Not Installed

```bash
pip install duckdb
```

If DuckDB is not available, use `PandasQueryEngine` as fallback:

```python
from stindex.warehouse import PandasQueryEngine

engine = PandasQueryEngine("data/warehouse")
df = engine.filter_temporal(year=2022)
```

### Slow Queries

1. Export to Parquet for faster queries:
   ```python
   warehouse.export_to_parquet()
   ```

2. Use indexed columns (temporal_year, spatial_labels)

3. Limit result sets with `.limit()`

### Memory Issues

Configure DuckDB memory limit in `cfg/warehouse.yml`:
```yaml
query:
  duckdb:
    memory_limit: "4GB"
```
