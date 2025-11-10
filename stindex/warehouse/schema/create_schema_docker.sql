-- ==============================================================================
-- STIndex Data Warehouse - Docker Schema Creation Script
-- ==============================================================================
-- This is a consolidated version of all schema files for Docker deployment.
-- It combines:
--   - 01_temporal_dimension.sql
--   - 02_spatial_dimension.sql
--   - 03_categorical_dimensions.sql
--   - 04_fact_table.sql
-- ==============================================================================

\echo '================================'
\echo 'Creating STIndex Data Warehouse'
\echo '================================'

-- Ensure extensions are installed
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS pg_trgm;

\echo 'Extensions verified.'
\echo ''

-- Run individual schema files
\echo 'Creating temporal dimension tables...'
\i /docker-entrypoint-initdb.d/01_temporal_dimension.sql

\echo ''
\echo 'Creating spatial dimension tables...'
\i /docker-entrypoint-initdb.d/02_spatial_dimension.sql

\echo ''
\echo 'Creating categorical dimension tables...'
\i /docker-entrypoint-initdb.d/03_categorical_dimensions.sql

\echo ''
\echo 'Creating fact table...'
\i /docker-entrypoint-initdb.d/04_fact_table.sql

\echo ''
\echo '================================'
\echo 'Schema creation completed!'
\echo '================================'
\echo ''
\echo 'Next step: Populate reference data'
\echo '  docker exec -i stindex-warehouse psql -U stindex -d stindex_warehouse < stindex/warehouse/schema/populate_dimensions.sql'
\echo ''
