-- ==============================================================================
-- Fact Table Schema (Hybrid Architecture)
-- ==============================================================================
-- Description: Central fact table for document chunks with hybrid capabilities
-- Features:
--   - Dimensional foreign keys for analytical queries
--   - Vector embeddings for semantic search (pgvector)
--   - Geographic coordinates for spatial queries (PostGIS)
--   - Dimensional label arrays for filtering
--   - Hierarchy paths for drill-down queries
-- Based on: dimensional-data-warehouse.md research findings
-- ==============================================================================

-- Ensure required extensions are enabled
CREATE EXTENSION IF NOT EXISTS vector;      -- pgvector for embeddings
CREATE EXTENSION IF NOT EXISTS postgis;     -- PostGIS for spatial queries
CREATE EXTENSION IF NOT EXISTS pg_trgm;     -- Fuzzy text search

-- ------------------------------------------------------------------------------
-- Fact Table: Document Chunks
-- ------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS fact_document_chunks (
    -- Primary key
    chunk_id               BIGSERIAL PRIMARY KEY,

    -- Document reference
    document_id            BIGINT NOT NULL REFERENCES dim_document(document_id),
    chunk_index            INT NOT NULL,                 -- Position of chunk in document (0-based)
    chunk_hash             VARCHAR(64) NOT NULL,         -- SHA-256 hash of chunk content

    -- =========================================================================
    -- DIMENSIONAL FOREIGN KEYS (for warehouse queries)
    -- =========================================================================

    -- Temporal dimension (can be null if no temporal entity in chunk)
    temporal_dim_id        BIGINT REFERENCES dim_temporal(temporal_id),

    -- Spatial dimension (can be null if no spatial entity in chunk)
    spatial_dim_id         BIGINT REFERENCES dim_spatial(spatial_id),

    -- Event dimension (can be null if no event entity in chunk)
    event_dim_id           BIGINT REFERENCES dim_event(event_id),

    -- Entity dimension (can be null if no named entity in chunk)
    entity_dim_id          BIGINT REFERENCES dim_entity(entity_id),

    -- =========================================================================
    -- CHUNK CONTENT
    -- =========================================================================

    -- Original chunk text
    chunk_text             TEXT NOT NULL,

    -- Cleaned chunk text (for vectorization)
    chunk_text_clean       TEXT,

    -- Chunk size metrics
    chunk_size_chars       INT NOT NULL,
    chunk_size_words       INT NOT NULL,
    chunk_size_tokens      INT,                          -- Token count (for LLM context)

    -- =========================================================================
    -- VECTOR EMBEDDING (for semantic search)
    -- =========================================================================

    -- Vector embedding for semantic similarity search
    chunk_vector           VECTOR(1536),                 -- OpenAI embedding dimension (configurable)

    -- Embedding metadata
    embedding_model        VARCHAR(100),                 -- 'text-embedding-3-small', 'all-MiniLM-L6-v2', etc.
    embedding_timestamp    TIMESTAMP,

    -- =========================================================================
    -- SPATIAL DATA (for PostGIS queries)
    -- =========================================================================

    -- Geographic point (if spatial entity present in chunk)
    location_geom          GEOGRAPHY(POINT, 4326),

    -- Coordinates (denormalized for convenience)
    latitude               DECIMAL(10, 7),
    longitude              DECIMAL(10, 7),

    -- =========================================================================
    -- DIMENSIONAL LABELS (for filtering)
    -- =========================================================================

    -- Hierarchical labels for each dimension (array of all hierarchy levels)
    temporal_labels        TEXT[],                       -- e.g., ["2022", "2022-Q1", "2022-03", "2022-03-15"]
    spatial_labels         TEXT[],                       -- e.g., ["Australia", "Western Australia", "Kimberley", "Broome"]
    event_labels           TEXT[],                       -- e.g., ["natural_disaster", "storm", "cyclone"]
    entity_labels          TEXT[],                       -- e.g., ["organization", "government_agency", "WHO"]

    -- =========================================================================
    -- HIERARCHY PATHS (for drill-down queries)
    -- =========================================================================

    -- Full hierarchy paths (human-readable)
    temporal_path          TEXT,                         -- "2022 > Q1 > March > 2022-03-15"
    spatial_path           TEXT,                         -- "Australia > Western Australia > Kimberley > Broome"
    event_path             TEXT,                         -- "natural_disaster > storm > cyclone"
    entity_path            TEXT,                         -- "organization > government_agency > WHO"

    -- Document section hierarchy (from preprocessing)
    section_hierarchy      TEXT,                         -- "Report > Section 3 > Subsection 3.2"

    -- =========================================================================
    -- EXTRACTION METADATA
    -- =========================================================================

    -- Overall confidence score (average across all dimensions)
    confidence_score       FLOAT CHECK (confidence_score BETWEEN 0 AND 1),

    -- Per-dimension confidence scores (JSON)
    dimension_confidences  JSONB,                        -- {"temporal": 0.95, "spatial": 0.87, "event": 0.92}

    -- Extraction context (JSON)
    extraction_context     JSONB,                        -- Prior references, document metadata, etc.

    -- Reflection scores (if two-pass reflection enabled)
    reflection_scores      JSONB,                        -- {"relevance": 0.9, "accuracy": 0.85, "consistency": 0.88}
    reflection_passed      BOOLEAN DEFAULT TRUE,         -- Whether reflection filtering passed

    -- Number of entities extracted per dimension
    entity_counts          JSONB,                        -- {"temporal": 2, "spatial": 1, "event": 1, "entity": 3}

    -- =========================================================================
    -- PROCESSING METADATA
    -- =========================================================================

    -- Processing timestamps
    extraction_timestamp   TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    loading_timestamp      TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,

    -- Processing configuration (reference to config used)
    extraction_config_id   VARCHAR(100),                 -- Config version/identifier

    -- LLM metadata (for tracking which model was used)
    llm_provider           VARCHAR(50),                  -- 'openai', 'anthropic', 'hf'
    llm_model              VARCHAR(100),                 -- 'gpt-4o-mini', 'claude-3-sonnet', 'Qwen/Qwen3-8B'
    llm_token_usage        JSONB,                        -- {"input_tokens": 1500, "output_tokens": 200}

    -- =========================================================================
    -- METADATA
    -- =========================================================================

    created_at             TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at             TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- =========================================================================
    -- CONSTRAINTS
    -- =========================================================================

    -- Unique constraint: one chunk per document index
    UNIQUE(document_id, chunk_index),

    -- Unique constraint: chunk hash per document (prevent duplicate chunks)
    UNIQUE(document_id, chunk_hash)
);

-- ------------------------------------------------------------------------------
-- Indexes for Performance
-- ------------------------------------------------------------------------------

-- Primary dimensional indexes (for warehouse queries)
CREATE INDEX IF NOT EXISTS idx_fact_chunks_document ON fact_document_chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_fact_chunks_temporal ON fact_document_chunks(temporal_dim_id);
CREATE INDEX IF NOT EXISTS idx_fact_chunks_spatial ON fact_document_chunks(spatial_dim_id);
CREATE INDEX IF NOT EXISTS idx_fact_chunks_event ON fact_document_chunks(event_dim_id);
CREATE INDEX IF NOT EXISTS idx_fact_chunks_entity ON fact_document_chunks(entity_dim_id);

-- Vector similarity index (for semantic search)
-- Using IVFFlat index (faster than HNSW for updates, good for < 1M vectors)
CREATE INDEX IF NOT EXISTS idx_fact_chunks_vector
ON fact_document_chunks
USING ivfflat (chunk_vector vector_cosine_ops)
WITH (lists = 100);
-- Note: For > 1M vectors, consider HNSW index:
-- CREATE INDEX idx_fact_chunks_vector_hnsw ON fact_document_chunks
-- USING hnsw (chunk_vector vector_cosine_ops)
-- WITH (m = 16, ef_construction = 64);

-- Spatial index (for PostGIS queries)
CREATE INDEX IF NOT EXISTS idx_fact_chunks_spatial_geom
ON fact_document_chunks
USING GIST (location_geom);

-- Label array indexes (for filtering by hierarchical labels)
CREATE INDEX IF NOT EXISTS idx_fact_chunks_temporal_labels
ON fact_document_chunks
USING GIN (temporal_labels);

CREATE INDEX IF NOT EXISTS idx_fact_chunks_spatial_labels
ON fact_document_chunks
USING GIN (spatial_labels);

CREATE INDEX IF NOT EXISTS idx_fact_chunks_event_labels
ON fact_document_chunks
USING GIN (event_labels);

CREATE INDEX IF NOT EXISTS idx_fact_chunks_entity_labels
ON fact_document_chunks
USING GIN (entity_labels);

-- Full-text search index (for keyword search)
CREATE INDEX IF NOT EXISTS idx_fact_chunks_text_search
ON fact_document_chunks
USING GIN (to_tsvector('english', chunk_text));

-- Composite indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_fact_chunks_doc_chunk
ON fact_document_chunks(document_id, chunk_index);

CREATE INDEX IF NOT EXISTS idx_fact_chunks_extraction_timestamp
ON fact_document_chunks(extraction_timestamp);

CREATE INDEX IF NOT EXISTS idx_fact_chunks_confidence
ON fact_document_chunks(confidence_score);

CREATE INDEX IF NOT EXISTS idx_fact_chunks_reflection
ON fact_document_chunks(reflection_passed)
WHERE reflection_passed = TRUE;

-- JSONB indexes for metadata queries
CREATE INDEX IF NOT EXISTS idx_fact_chunks_dim_confidences
ON fact_document_chunks
USING GIN (dimension_confidences);

CREATE INDEX IF NOT EXISTS idx_fact_chunks_reflection_scores
ON fact_document_chunks
USING GIN (reflection_scores);

CREATE INDEX IF NOT EXISTS idx_fact_chunks_entity_counts
ON fact_document_chunks
USING GIN (entity_counts);

-- ------------------------------------------------------------------------------
-- Partitioning (Optional - for large datasets)
-- ------------------------------------------------------------------------------
-- Uncomment to enable partitioning by extraction timestamp (monthly partitions)
-- ALTER TABLE fact_document_chunks PARTITION BY RANGE (extraction_timestamp);
--
-- -- Create initial partitions
-- CREATE TABLE fact_document_chunks_2024_01 PARTITION OF fact_document_chunks
--     FOR VALUES FROM ('2024-01-01') TO ('2024-02-01');
--
-- CREATE TABLE fact_document_chunks_2024_02 PARTITION OF fact_document_chunks
--     FOR VALUES FROM ('2024-02-01') TO ('2024-03-01');
--
-- -- Continue creating partitions as needed...

-- ------------------------------------------------------------------------------
-- Triggers for Updated Timestamp
-- ------------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_fact_chunks_updated_at
BEFORE UPDATE ON fact_document_chunks
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();

-- ------------------------------------------------------------------------------
-- Helper Functions for Fact Table
-- ------------------------------------------------------------------------------

-- Function to calculate average confidence score across dimensions
CREATE OR REPLACE FUNCTION calculate_avg_confidence(dimension_confidences JSONB)
RETURNS FLOAT AS $$
DECLARE
    total FLOAT := 0;
    count INT := 0;
    key TEXT;
    value FLOAT;
BEGIN
    FOR key, value IN SELECT * FROM jsonb_each_text(dimension_confidences)
    LOOP
        total := total + value::FLOAT;
        count := count + 1;
    END LOOP;

    IF count = 0 THEN
        RETURN NULL;
    END IF;

    RETURN total / count;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Function to get chunks within spatial radius and temporal range
CREATE OR REPLACE FUNCTION get_chunks_spatiotemporal(
    center_lat DECIMAL,
    center_lon DECIMAL,
    radius_km DECIMAL,
    start_date DATE,
    end_date DATE
) RETURNS TABLE (
    chunk_id BIGINT,
    chunk_text TEXT,
    distance_km DECIMAL,
    extraction_date DATE,
    confidence_score FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        f.chunk_id,
        f.chunk_text,
        ST_Distance(
            f.location_geom,
            ST_MakePoint(center_lon, center_lat)::geography
        ) / 1000.0 AS distance_km,
        t.full_date AS extraction_date,
        f.confidence_score
    FROM fact_document_chunks f
    JOIN dim_temporal temp ON f.temporal_dim_id = temp.temporal_id
    JOIN dim_date t ON temp.date_id = t.date_id
    WHERE
        ST_DWithin(
            f.location_geom,
            ST_MakePoint(center_lon, center_lat)::geography,
            radius_km * 1000  -- Convert km to meters
        )
        AND t.full_date BETWEEN start_date AND end_date
    ORDER BY distance_km, t.full_date;
END;
$$ LANGUAGE plpgsql;

-- ------------------------------------------------------------------------------
-- Statistics Views
-- ------------------------------------------------------------------------------

-- View: Summary statistics by dimension
CREATE VIEW IF NOT EXISTS v_dimension_statistics AS
SELECT
    COUNT(*) AS total_chunks,
    COUNT(DISTINCT document_id) AS total_documents,
    COUNT(temporal_dim_id) AS chunks_with_temporal,
    COUNT(spatial_dim_id) AS chunks_with_spatial,
    COUNT(event_dim_id) AS chunks_with_event,
    COUNT(entity_dim_id) AS chunks_with_entity,
    AVG(confidence_score) AS avg_confidence,
    AVG(chunk_size_words) AS avg_chunk_words,
    COUNT(chunk_vector) AS chunks_with_embeddings,
    COUNT(location_geom) AS chunks_with_locations
FROM fact_document_chunks;

-- View: Extraction statistics by LLM model
CREATE VIEW IF NOT EXISTS v_extraction_by_model AS
SELECT
    llm_provider,
    llm_model,
    COUNT(*) AS total_chunks,
    AVG(confidence_score) AS avg_confidence,
    AVG((llm_token_usage->>'input_tokens')::INT) AS avg_input_tokens,
    AVG((llm_token_usage->>'output_tokens')::INT) AS avg_output_tokens
FROM fact_document_chunks
WHERE llm_token_usage IS NOT NULL
GROUP BY llm_provider, llm_model;

-- View: Temporal distribution of chunks
CREATE VIEW IF NOT EXISTS v_temporal_distribution AS
SELECT
    EXTRACT(YEAR FROM d.full_date) AS year,
    EXTRACT(QUARTER FROM d.full_date) AS quarter,
    COUNT(*) AS chunk_count,
    AVG(f.confidence_score) AS avg_confidence
FROM fact_document_chunks f
JOIN dim_temporal t ON f.temporal_dim_id = t.temporal_id
JOIN dim_date d ON t.date_id = d.date_id
GROUP BY EXTRACT(YEAR FROM d.full_date), EXTRACT(QUARTER FROM d.full_date)
ORDER BY year, quarter;

-- View: Spatial distribution of chunks by country
CREATE VIEW IF NOT EXISTS v_spatial_distribution AS
SELECT
    co.country_name,
    st.state_name,
    COUNT(*) AS chunk_count,
    AVG(f.confidence_score) AS avg_confidence
FROM fact_document_chunks f
JOIN dim_spatial s ON f.spatial_dim_id = s.spatial_id
LEFT JOIN dim_country co ON s.country_id = co.country_id
LEFT JOIN dim_state st ON s.state_id = st.state_id
GROUP BY co.country_name, st.state_name
ORDER BY chunk_count DESC;

-- ------------------------------------------------------------------------------
-- Comments
-- ------------------------------------------------------------------------------
COMMENT ON TABLE fact_document_chunks IS 'Central fact table for document chunks with hybrid capabilities (vector, spatial, dimensional)';
COMMENT ON COLUMN fact_document_chunks.chunk_vector IS 'Vector embedding for semantic similarity search (dimension: 1536 for OpenAI)';
COMMENT ON COLUMN fact_document_chunks.location_geom IS 'PostGIS POINT geometry for spatial queries';
COMMENT ON COLUMN fact_document_chunks.temporal_labels IS 'Hierarchical temporal labels for filtering (e.g., ["2022", "2022-Q1", "2022-03"])';
COMMENT ON COLUMN fact_document_chunks.spatial_labels IS 'Hierarchical spatial labels for filtering (e.g., ["Australia", "WA", "Broome"])';
COMMENT ON COLUMN fact_document_chunks.reflection_passed IS 'Whether chunk passed two-pass reflection quality filtering';
COMMENT ON VIEW v_dimension_statistics IS 'Summary statistics showing dimension coverage and data quality';
COMMENT ON VIEW v_extraction_by_model IS 'Extraction statistics grouped by LLM provider and model';
COMMENT ON VIEW v_temporal_distribution IS 'Distribution of chunks over time (by year and quarter)';
COMMENT ON VIEW v_spatial_distribution IS 'Distribution of chunks by geographic location (country and state)';
