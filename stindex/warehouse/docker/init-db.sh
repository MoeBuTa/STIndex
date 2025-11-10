#!/bin/bash
# ==============================================================================
# PostgreSQL Database Initialization Script for Docker
# ==============================================================================
# This script runs when the PostgreSQL container is first created.
# It installs required extensions and optionally creates the warehouse schema.
# ==============================================================================

set -e

# Switch to the stindex_warehouse database
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    -- ==============================================================================
    -- Install Required Extensions
    -- ==============================================================================

    -- pgvector for vector embeddings (already included in pgvector/pgvector image)
    CREATE EXTENSION IF NOT EXISTS vector;

    -- PostGIS for spatial queries
    CREATE EXTENSION IF NOT EXISTS postgis;

    -- pg_trgm for fuzzy text search
    CREATE EXTENSION IF NOT EXISTS pg_trgm;

    -- ==============================================================================
    -- Verify Extensions
    -- ==============================================================================

    SELECT
        extname AS extension,
        extversion AS version
    FROM pg_extension
    WHERE extname IN ('vector', 'postgis', 'pg_trgm')
    ORDER BY extname;

EOSQL

echo "✓ Extensions installed successfully:"
echo "  - vector (pgvector)"
echo "  - postgis"
echo "  - pg_trgm"
echo ""
echo "Database is ready for schema creation!"
echo ""
echo "To create the warehouse schema, run:"
echo "  docker exec -i stindex-warehouse psql -U stindex -d stindex_warehouse < stindex/warehouse/schema/create_schema_docker.sql"
