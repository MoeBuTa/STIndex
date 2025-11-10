-- ==============================================================================
-- STIndex Data Warehouse - Master Schema
-- ==============================================================================
-- Description: Master DDL script to create complete dimensional data warehouse
-- Architecture: Hybrid snowflake/star schema with vector and spatial capabilities
-- Prerequisites:
--   - PostgreSQL 15+
--   - pgvector extension
--   - PostGIS extension
--   - pg_trgm extension
-- ==============================================================================

-- Create database (run this separately as superuser)
-- CREATE DATABASE stindex_warehouse;

-- Connect to database
\c stindex_warehouse

-- ==============================================================================
-- Enable Required Extensions
-- ==============================================================================
CREATE EXTENSION IF NOT EXISTS vector;      -- pgvector for vector embeddings
CREATE EXTENSION IF NOT EXISTS postgis;     -- PostGIS for spatial queries
CREATE EXTENSION IF NOT EXISTS pg_trgm;     -- Fuzzy text search

-- ==============================================================================
-- Schema 1: Temporal Dimension (Snowflake Pattern)
-- ==============================================================================
\echo 'Creating temporal dimension tables...'
\i 01_temporal_dimension.sql

-- ==============================================================================
-- Schema 2: Spatial Dimension (Snowflake Pattern)
-- ==============================================================================
\echo 'Creating spatial dimension tables...'
\i 02_spatial_dimension.sql

-- ==============================================================================
-- Schema 3: Categorical Dimensions (Event, Entity, Document)
-- ==============================================================================
\echo 'Creating categorical dimension tables...'
\i 03_categorical_dimensions.sql

-- ==============================================================================
-- Schema 4: Fact Table (Hybrid Architecture)
-- ==============================================================================
\echo 'Creating fact table...'
\i 04_fact_table.sql

-- ==============================================================================
-- Verification Queries
-- ==============================================================================
\echo 'Schema creation complete!'
\echo ''
\echo 'Verifying schema...'

-- List all tables
SELECT
    schemaname,
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

-- Count tables by type
SELECT
    'Temporal Dimension Tables' AS category,
    COUNT(*) AS table_count
FROM pg_tables
WHERE schemaname = 'public'
    AND tablename LIKE 'dim_%'
    AND tablename IN ('dim_year', 'dim_quarter', 'dim_month', 'dim_date', 'dim_time', 'dim_temporal')

UNION ALL

SELECT
    'Spatial Dimension Tables' AS category,
    COUNT(*) AS table_count
FROM pg_tables
WHERE schemaname = 'public'
    AND tablename LIKE 'dim_%'
    AND tablename IN ('dim_continent', 'dim_country', 'dim_state', 'dim_region', 'dim_city', 'dim_suburb', 'dim_address', 'dim_spatial')

UNION ALL

SELECT
    'Event Dimension Tables' AS category,
    COUNT(*) AS table_count
FROM pg_tables
WHERE schemaname = 'public'
    AND tablename LIKE 'dim_event%'

UNION ALL

SELECT
    'Entity Dimension Tables' AS category,
    COUNT(*) AS table_count
FROM pg_tables
WHERE schemaname = 'public'
    AND tablename LIKE 'dim_entity%'

UNION ALL

SELECT
    'Document Dimension Tables' AS category,
    COUNT(*) AS table_count
FROM pg_tables
WHERE schemaname = 'public'
    AND tablename = 'dim_document'

UNION ALL

SELECT
    'Fact Tables' AS category,
    COUNT(*) AS table_count
FROM pg_tables
WHERE schemaname = 'public'
    AND tablename LIKE 'fact_%';

-- List all materialized views
\echo ''
\echo 'Materialized views:'
SELECT
    schemaname,
    matviewname,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||matviewname)) AS size
FROM pg_matviews
WHERE schemaname = 'public';

-- List all regular views
\echo ''
\echo 'Regular views:'
SELECT
    schemaname,
    viewname
FROM pg_views
WHERE schemaname = 'public'
ORDER BY viewname;

-- ==============================================================================
-- Database Statistics
-- ==============================================================================
\echo ''
\echo 'Database statistics:'
SELECT
    pg_size_pretty(pg_database_size('stindex_warehouse')) AS database_size;

\echo ''
\echo '================================'
\echo 'Schema creation completed successfully!'
\echo 'Next steps:'
\echo '  1. Run populate_dimensions.sql to pre-populate reference data'
\echo '  2. Run refresh_materialized_views.sql to initialize materialized views'
\echo '  3. Test the schema with sample queries in examples/queries.sql'
\echo '================================'
