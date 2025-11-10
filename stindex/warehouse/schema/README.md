# STIndex Data Warehouse Schema

## Overview

This directory contains the complete schema definition for the STIndex dimensional data warehouse. The warehouse uses a **hybrid snowflake/star architecture** with advanced capabilities for:

- **Dimensional Analytics**: Multi-level hierarchies for temporal, spatial, and categorical dimensions
- **Semantic Search**: Vector embeddings with pgvector for similarity search
- **Spatial Queries**: PostGIS-powered geographic queries
- **Hybrid Queries**: Combined semantic, dimensional, and spatial filtering

## Architecture

### Schema Pattern

The warehouse follows **Kimball dimensional modeling** with modern enhancements:

- **Snowflake Schema**: For hierarchical dimensions (temporal, spatial, event taxonomies)
- **Star Schema**: For flat dimensions (document metadata)
- **Hybrid Fact Table**: Combines dimensional FKs, vector embeddings, and spatial data

### Key Design Principles

1. **Hierarchical Dimensions**: Enable drill-down and roll-up queries
2. **Denormalized Labels**: Fast filtering without joins using array types
3. **Materialized Views**: Pre-computed joins for performance
4. **Flexible Metadata**: JSONB fields for extensibility

## Schema Files

### Core Schema Files

| File | Description | Tables Created |
|------|-------------|----------------|
| `01_temporal_dimension.sql` | Temporal hierarchy (Year → Quarter → Month → Day) | dim_year, dim_quarter, dim_month, dim_date, dim_time, dim_temporal |
| `02_spatial_dimension.sql` | Spatial hierarchy (Continent → Country → State → Region → City) | dim_continent, dim_country, dim_state, dim_region, dim_city, dim_suburb, dim_address, dim_spatial |
| `03_categorical_dimensions.sql` | Event, Entity, and Document dimensions | dim_event_*, dim_entity_*, dim_document |
| `04_fact_table.sql` | Central fact table with hybrid capabilities | fact_document_chunks + views |

### Setup Files

| File | Description |
|------|-------------|
| `create_schema.sql` | Master script to create all tables |
| `populate_dimensions.sql` | Pre-populate reference data (dates, countries, taxonomies) |

## Installation

### Prerequisites

- PostgreSQL 15+
- pgvector extension
- PostGIS extension
- pg_trgm extension

### Installation Steps

```bash
# 1. Install PostgreSQL with extensions
sudo apt-get install postgresql-15 postgresql-15-pgvector postgresql-15-postgis

# 2. Create database
createdb stindex_warehouse

# 3. Create schema
cd stindex/warehouse/schema
psql stindex_warehouse -f create_schema.sql

# 4. Populate reference data
psql stindex_warehouse -f populate_dimensions.sql
```

## Schema Details

### Temporal Dimension

**Hierarchy**: Year → Quarter → Month → Day → Time

**Key Features**:
- Complete date range (2000-2050) pre-populated
- Business calendar attributes (fiscal year, quarters)
- Holiday flags and weekend indicators
- ISO 8601 week numbering
- Time of day dimension with business hours

**Use Cases**:
- Trend analysis by year/quarter/month
- Weekend vs weekday patterns
- Business hours filtering
- Holiday impact analysis

### Spatial Dimension

**Hierarchy**: Continent → Country → State → Region → City → Suburb → Address

**Key Features**:
- PostGIS GEOGRAPHY type for accurate distance calculations
- Bounding boxes for spatial filtering
- Geocoded coordinates with confidence scores
- Parent region context for disambiguation

**Use Cases**:
- Geographic distribution analysis
- Radius-based searches (e.g., events within 100km)
- Hierarchical rollups (city → state → country)
- Cross-geographic comparisons

### Event Dimension

**Hierarchy**: Category → Type → Subtype

**Example Taxonomy**:
```
natural_disaster
  → storm
    → tropical_cyclone
    → tornado
    → blizzard
  → earthquake
  → wildfire
```

**Key Features**:
- Extensible taxonomy (add new categories dynamically)
- Severity and impact scale attributes
- Confidence scores from extraction

### Entity Dimension

**Hierarchy**: Category → Type

**Example Taxonomy**:
```
organization
  → government_agency
  → corporation
  → ngo
person
  → politician
  → celebrity
  → scientist
```

**Key Features**:
- Wikidata/Wikipedia linking for entity resolution
- Aliases array for alternative names
- Flexible JSONB attributes for entity-specific data

### Document Dimension

**Flat Structure** (no hierarchy)

**Key Features**:
- SHA-256 hash for deduplication
- Source metadata (URL, file path, publication info)
- Processing metadata (config used, timestamp)
- Document characteristics (word count, page count)

### Fact Table: fact_document_chunks

**Central Table** connecting all dimensions

**Key Columns**:

| Column Type | Purpose | Example |
|-------------|---------|---------|
| Dimensional FKs | Link to dimension tables | `temporal_dim_id`, `spatial_dim_id` |
| Chunk Content | Original and cleaned text | `chunk_text`, `chunk_text_clean` |
| Vector Embedding | Semantic search (1536-d) | `chunk_vector` |
| Spatial Data | PostGIS POINT geometry | `location_geom` |
| Label Arrays | Fast filtering | `temporal_labels`, `spatial_labels` |
| Hierarchy Paths | Drill-down queries | `temporal_path`, `spatial_path` |
| Confidence Scores | Quality filtering | `confidence_score`, `dimension_confidences` |

## Query Capabilities

### 1. Semantic Search

```sql
-- Find similar chunks using vector similarity
SELECT chunk_text,
       1 - (chunk_vector <=> '[0.1, 0.2, ...]'::vector) AS similarity
FROM fact_document_chunks
ORDER BY chunk_vector <=> '[0.1, 0.2, ...]'::vector
LIMIT 10;
```

### 2. Dimensional Filtering

```sql
-- Filter by temporal and spatial dimensions
SELECT chunk_text
FROM fact_document_chunks f
JOIN dim_date d ON f.temporal_dim_id = d.temporal_id
WHERE d.year = 2022
  AND 'Australia' = ANY(f.spatial_labels);
```

### 3. Spatial Queries

```sql
-- Find chunks within 100km of Broome, Australia
SELECT chunk_text
FROM fact_document_chunks
WHERE ST_DWithin(
    location_geom,
    ST_MakePoint(122.2, -18.0)::geography,  -- Broome coords
    100000  -- 100km radius (meters)
);
```

### 4. Hierarchical Rollup

```sql
-- Event counts by state and region
SELECT
    s.state_name,
    r.region_name,
    COUNT(*) as event_count
FROM fact_document_chunks f
JOIN dim_spatial sp ON f.spatial_dim_id = sp.spatial_id
JOIN dim_state s ON sp.state_id = s.state_id
LEFT JOIN dim_region r ON sp.region_id = r.region_id
GROUP BY ROLLUP(s.state_name, r.region_name)
ORDER BY s.state_name, r.region_name;
```

### 5. Hybrid Query (Semantic + Dimensional + Spatial)

```sql
-- Combine all three query types
SELECT
    f.chunk_text,
    1 - (f.chunk_vector <=> '[...]'::vector) AS similarity,
    d.full_date,
    s.city_name
FROM fact_document_chunks f
JOIN dim_temporal t ON f.temporal_dim_id = t.temporal_id
JOIN dim_date d ON t.date_id = d.date_id
JOIN dim_spatial sp ON f.spatial_dim_id = sp.spatial_id
JOIN dim_city c ON sp.city_id = c.city_id
WHERE
    d.year >= 2020
    AND ST_DWithin(f.location_geom, ST_MakePoint(122.2, -18.0)::geography, 100000)
ORDER BY f.chunk_vector <=> '[...]'::vector
LIMIT 10;
```

## Materialized Views

Pre-computed views for performance:

| View | Purpose | Refresh Strategy |
|------|---------|------------------|
| `mv_temporal_hierarchy` | Denormalized temporal hierarchy | Refresh after dimension updates |
| `mv_spatial_hierarchy` | Denormalized spatial hierarchy | Refresh after dimension updates |
| `mv_event_hierarchy` | Denormalized event hierarchy | Refresh after taxonomy updates |
| `mv_entity_hierarchy` | Denormalized entity hierarchy | Refresh after taxonomy updates |

**Refresh Command**:
```sql
REFRESH MATERIALIZED VIEW mv_temporal_hierarchy;
REFRESH MATERIALIZED VIEW mv_spatial_hierarchy;
REFRESH MATERIALIZED VIEW mv_event_hierarchy;
REFRESH MATERIALIZED VIEW mv_entity_hierarchy;
```

## Statistics Views

Built-in views for monitoring:

| View | Purpose |
|------|---------|
| `v_dimension_statistics` | Overall dimension coverage |
| `v_extraction_by_model` | Performance by LLM model |
| `v_temporal_distribution` | Temporal distribution of chunks |
| `v_spatial_distribution` | Spatial distribution of chunks |

## Indexes

### Vector Indexes

- **IVFFlat**: Used for < 1M vectors (faster updates)
- **HNSW**: Recommended for > 1M vectors (faster queries)

### Spatial Indexes

- **GIST**: Used for PostGIS geography columns
- Enables efficient radius-based searches

### Array Indexes

- **GIN**: Used for label arrays and JSONB columns
- Enables fast filtering on hierarchical labels

## Performance Considerations

### Index Selection

- Vector index lists parameter: `lists = 100` (adjust based on dataset size)
- Rebuild vector index after bulk inserts: `REINDEX INDEX idx_fact_chunks_vector;`

### Partitioning

For datasets > 10M chunks, consider partitioning by extraction timestamp:

```sql
ALTER TABLE fact_document_chunks PARTITION BY RANGE (extraction_timestamp);
```

### Vacuum and Analyze

Run regularly for optimal performance:

```bash
# Analyze tables to update statistics
psql stindex_warehouse -c "ANALYZE fact_document_chunks;"

# Vacuum to reclaim space
psql stindex_warehouse -c "VACUUM ANALYZE fact_document_chunks;"
```

## Schema Evolution

### Adding New Dimensions

1. Create dimension hierarchy tables in new SQL file
2. Add FK column to `fact_document_chunks`
3. Add label array column (e.g., `custom_labels TEXT[]`)
4. Add hierarchy path column (e.g., `custom_path TEXT`)
5. Update ETL code to populate new dimension

### Modifying Taxonomies

Event and entity taxonomies can be extended dynamically:

```sql
-- Add new event category
INSERT INTO dim_event_category (category_name, category_description)
VALUES ('cyber_security', 'Cybersecurity incidents and breaches');

-- Add event type under new category
INSERT INTO dim_event_type (type_name, type_description, event_category_id)
SELECT 'data_breach', 'Unauthorized data access', event_category_id
FROM dim_event_category
WHERE category_name = 'cyber_security';
```

## Troubleshooting

### Vector Index Build Fails

```sql
-- Check pgvector version
SELECT * FROM pg_extension WHERE extname = 'vector';

-- Rebuild index with smaller lists parameter
DROP INDEX idx_fact_chunks_vector;
CREATE INDEX idx_fact_chunks_vector
ON fact_document_chunks
USING ivfflat (chunk_vector vector_cosine_ops)
WITH (lists = 50);
```

### Slow Spatial Queries

```sql
-- Check if PostGIS index exists
SELECT indexname FROM pg_indexes
WHERE tablename = 'fact_document_chunks'
  AND indexname LIKE '%geom%';

-- Rebuild spatial index
REINDEX INDEX idx_fact_chunks_spatial_geom;
```

### Materialized View Out of Date

```sql
-- Check last refresh time
SELECT schemaname, matviewname, last_refresh
FROM pg_stat_user_tables
WHERE relname LIKE 'mv_%';

-- Refresh all materialized views
REFRESH MATERIALIZED VIEW mv_temporal_hierarchy;
REFRESH MATERIALIZED VIEW mv_spatial_hierarchy;
REFRESH MATERIALIZED VIEW mv_event_hierarchy;
REFRESH MATERIALIZED VIEW mv_entity_hierarchy;
```

## Next Steps

After schema creation:

1. **Phase 2**: Implement chunk labeling (`stindex/warehouse/chunk_labeler.py`)
2. **Phase 3**: Build ETL pipeline (`stindex/warehouse/etl.py`)
3. **Phase 4**: Create query API (`stindex/warehouse/query_api.py`)
4. **Phase 5**: Testing and optimization

## References

- [dimensional-data-warehouse.md](../../docs/research/dimensional-data-warehouse.md) - Research and design decisions
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [pgvector Documentation](https://github.com/pgvector/pgvector)
- [PostGIS Documentation](https://postgis.net/documentation/)
- [Kimball Dimensional Modeling](https://www.kimballgroup.com/data-warehouse-business-intelligence-resources/kimball-techniques/)

## License

See project root LICENSE file.
