-- ==============================================================================
-- Categorical Dimension Schemas (Event, Entity, Document)
-- ==============================================================================
-- Description: Categorical dimension schemas for events, entities, and documents
-- Event Hierarchy: Category → Type → Subtype (e.g., natural_disaster > storm > cyclone)
-- Entity Hierarchy: Category → Type (e.g., organization > government_agency)
-- Document: Flat dimension with metadata
-- Based on: dimensional-data-warehouse.md research findings
-- ==============================================================================

-- ------------------------------------------------------------------------------
-- Event Dimension Tables (Snowflake Pattern)
-- ------------------------------------------------------------------------------

-- Event Category Hierarchy Table (Top Level)
CREATE TABLE IF NOT EXISTS dim_event_category (
    event_category_id     BIGSERIAL PRIMARY KEY,
    category_name         VARCHAR(100) UNIQUE NOT NULL,  -- 'natural_disaster', 'political_event', 'economic_event', etc.
    category_description  TEXT,

    -- Metadata
    created_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index for category lookups
CREATE INDEX IF NOT EXISTS idx_dim_event_category_name ON dim_event_category(category_name);

-- Event Type Hierarchy Table (Middle Level)
CREATE TABLE IF NOT EXISTS dim_event_type (
    event_type_id         BIGSERIAL PRIMARY KEY,
    type_name             VARCHAR(100) NOT NULL,         -- 'storm', 'earthquake', 'election', 'recession', etc.
    type_description      TEXT,
    event_category_id     BIGINT NOT NULL REFERENCES dim_event_category(event_category_id),

    -- Metadata
    created_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Ensure unique event types per category
    UNIQUE(event_category_id, type_name)
);

-- Index for type lookups
CREATE INDEX IF NOT EXISTS idx_dim_event_type_name ON dim_event_type(type_name);
CREATE INDEX IF NOT EXISTS idx_dim_event_type_category ON dim_event_type(event_category_id);

-- Event Subtype Hierarchy Table (Bottom Level)
CREATE TABLE IF NOT EXISTS dim_event_subtype (
    event_subtype_id      BIGSERIAL PRIMARY KEY,
    subtype_name          VARCHAR(100) NOT NULL,         -- 'tropical_cyclone', 'tornado', 'blizzard', etc.
    subtype_description   TEXT,
    event_type_id         BIGINT NOT NULL REFERENCES dim_event_type(event_type_id),

    -- Metadata
    created_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Ensure unique event subtypes per type
    UNIQUE(event_type_id, subtype_name)
);

-- Index for subtype lookups
CREATE INDEX IF NOT EXISTS idx_dim_event_subtype_name ON dim_event_subtype(subtype_name);
CREATE INDEX IF NOT EXISTS idx_dim_event_subtype_type ON dim_event_subtype(event_type_id);

-- Base Event Dimension Table
CREATE TABLE IF NOT EXISTS dim_event (
    event_id              BIGSERIAL PRIMARY KEY,

    -- Original extraction data
    original_text         TEXT NOT NULL,                 -- "tropical cyclone", "election", etc.

    -- Foreign keys to hierarchy (snowflake pattern)
    event_subtype_id      BIGINT REFERENCES dim_event_subtype(event_subtype_id),
    event_type_id         BIGINT REFERENCES dim_event_type(event_type_id),
    event_category_id     BIGINT REFERENCES dim_event_category(event_category_id),

    -- Additional attributes
    severity              VARCHAR(20),                   -- 'low', 'moderate', 'high', 'severe', 'extreme'
    impact_scale          VARCHAR(50),                   -- 'local', 'regional', 'national', 'international', 'global'

    -- Extraction metadata
    confidence            FLOAT CHECK (confidence BETWEEN 0 AND 1),
    extraction_method     VARCHAR(50),                   -- 'llm', 'rule_based', 'hybrid'

    -- Metadata
    created_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- At least one hierarchy level must be specified
    CONSTRAINT chk_event_hierarchy CHECK (
        event_subtype_id IS NOT NULL OR
        event_type_id IS NOT NULL OR
        event_category_id IS NOT NULL
    )
);

-- Indexes for event lookups
CREATE INDEX IF NOT EXISTS idx_dim_event_original_text ON dim_event(original_text);
CREATE INDEX IF NOT EXISTS idx_dim_event_subtype ON dim_event(event_subtype_id);
CREATE INDEX IF NOT EXISTS idx_dim_event_type ON dim_event(event_type_id);
CREATE INDEX IF NOT EXISTS idx_dim_event_category ON dim_event(event_category_id);
CREATE INDEX IF NOT EXISTS idx_dim_event_severity ON dim_event(severity);
CREATE INDEX IF NOT EXISTS idx_dim_event_confidence ON dim_event(confidence);

-- ------------------------------------------------------------------------------
-- Entity Dimension Tables (Snowflake Pattern)
-- ------------------------------------------------------------------------------

-- Entity Category Hierarchy Table (Top Level)
CREATE TABLE IF NOT EXISTS dim_entity_category (
    entity_category_id    BIGSERIAL PRIMARY KEY,
    category_name         VARCHAR(100) UNIQUE NOT NULL,  -- 'person', 'organization', 'location', 'product', etc.
    category_description  TEXT,

    -- Metadata
    created_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index for category lookups
CREATE INDEX IF NOT EXISTS idx_dim_entity_category_name ON dim_entity_category(category_name);

-- Entity Type Hierarchy Table (Bottom Level)
CREATE TABLE IF NOT EXISTS dim_entity_type (
    entity_type_id        BIGSERIAL PRIMARY KEY,
    type_name             VARCHAR(100) NOT NULL,         -- 'politician', 'celebrity', 'government_agency', 'corporation', etc.
    type_description      TEXT,
    entity_category_id    BIGINT NOT NULL REFERENCES dim_entity_category(entity_category_id),

    -- Metadata
    created_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Ensure unique entity types per category
    UNIQUE(entity_category_id, type_name)
);

-- Index for type lookups
CREATE INDEX IF NOT EXISTS idx_dim_entity_type_name ON dim_entity_type(type_name);
CREATE INDEX IF NOT EXISTS idx_dim_entity_type_category ON dim_entity_type(entity_category_id);

-- Base Entity Dimension Table
CREATE TABLE IF NOT EXISTS dim_entity (
    entity_id             BIGSERIAL PRIMARY KEY,

    -- Original extraction data
    original_text         TEXT NOT NULL,                 -- "WHO", "John Smith", "Apple Inc.", etc.
    entity_name           VARCHAR(200) NOT NULL,         -- Normalized entity name

    -- Foreign keys to hierarchy (snowflake pattern)
    entity_type_id        BIGINT REFERENCES dim_entity_type(entity_type_id),
    entity_category_id    BIGINT REFERENCES dim_entity_category(entity_category_id),

    -- Additional attributes
    entity_description    TEXT,
    aliases               TEXT[],                        -- Array of alternative names

    -- Entity-specific attributes (JSON for flexibility)
    attributes            JSONB,                         -- e.g., {"role": "Prime Minister", "party": "Liberal"}

    -- External identifiers
    wikidata_id           VARCHAR(50),                   -- Wikidata QID (e.g., Q408)
    wikipedia_url         TEXT,

    -- Extraction metadata
    confidence            FLOAT CHECK (confidence BETWEEN 0 AND 1),
    extraction_method     VARCHAR(50),                   -- 'llm', 'ner', 'hybrid'

    -- Metadata
    created_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- At least one hierarchy level must be specified
    CONSTRAINT chk_entity_hierarchy CHECK (
        entity_type_id IS NOT NULL OR
        entity_category_id IS NOT NULL
    )
);

-- Indexes for entity lookups
CREATE INDEX IF NOT EXISTS idx_dim_entity_original_text ON dim_entity(original_text);
CREATE INDEX IF NOT EXISTS idx_dim_entity_name ON dim_entity(entity_name);
CREATE INDEX IF NOT EXISTS idx_dim_entity_type ON dim_entity(entity_type_id);
CREATE INDEX IF NOT EXISTS idx_dim_entity_category ON dim_entity(entity_category_id);
CREATE INDEX IF NOT EXISTS idx_dim_entity_wikidata ON dim_entity(wikidata_id);
CREATE INDEX IF NOT EXISTS idx_dim_entity_confidence ON dim_entity(confidence);
CREATE INDEX IF NOT EXISTS idx_dim_entity_aliases ON dim_entity USING GIN(aliases);
CREATE INDEX IF NOT EXISTS idx_dim_entity_attributes ON dim_entity USING GIN(attributes);

-- ------------------------------------------------------------------------------
-- Document Dimension Table (Star Pattern - Flat)
-- ------------------------------------------------------------------------------
-- Document metadata dimension (no hierarchy, flat structure)
CREATE TABLE IF NOT EXISTS dim_document (
    document_id           BIGSERIAL PRIMARY KEY,

    -- Document identifiers
    document_hash         VARCHAR(64) UNIQUE NOT NULL,   -- SHA-256 hash of document content
    document_url          TEXT,                          -- Original URL (if web document)
    document_path         TEXT,                          -- File path (if local document)

    -- Document metadata
    document_title        TEXT,
    document_type         VARCHAR(50),                   -- 'html', 'pdf', 'docx', 'txt', 'markdown', etc.
    document_language     VARCHAR(10),                   -- 'en', 'es', 'fr', etc.

    -- Publication information
    publication_date      DATE,
    publication_source    VARCHAR(200),                  -- 'The New York Times', 'BBC', etc.
    author                VARCHAR(200),
    publisher             VARCHAR(200),

    -- Document characteristics
    word_count            INT,
    char_count            INT,
    page_count            INT,
    total_chunks          INT,                           -- Number of chunks generated

    -- Processing metadata
    processing_timestamp  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    preprocessing_config  JSONB,                         -- Preprocessing settings used
    extraction_config     JSONB,                         -- Extraction settings used

    -- Document source metadata (flexible JSON)
    source_metadata       JSONB,                         -- Additional metadata from source

    -- Metadata
    created_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for document lookups
CREATE INDEX IF NOT EXISTS idx_dim_document_hash ON dim_document(document_hash);
CREATE INDEX IF NOT EXISTS idx_dim_document_url ON dim_document(document_url);
CREATE INDEX IF NOT EXISTS idx_dim_document_type ON dim_document(document_type);
CREATE INDEX IF NOT EXISTS idx_dim_document_pub_date ON dim_document(publication_date);
CREATE INDEX IF NOT EXISTS idx_dim_document_source ON dim_document(publication_source);
CREATE INDEX IF NOT EXISTS idx_dim_document_source_metadata ON dim_document USING GIN(source_metadata);

-- ------------------------------------------------------------------------------
-- Views for Easy Querying
-- ------------------------------------------------------------------------------

-- Materialized view for complete event hierarchy
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_event_hierarchy AS
SELECT
    e.event_id,
    e.original_text,
    e.severity,
    e.confidence,

    -- Subtype level
    es.subtype_name AS event_subtype,
    es.subtype_description AS subtype_description,

    -- Type level
    et.type_name AS event_type,
    et.type_description AS type_description,

    -- Category level
    ec.category_name AS event_category,
    ec.category_description AS category_description

FROM dim_event e
LEFT JOIN dim_event_subtype es ON e.event_subtype_id = es.event_subtype_id
LEFT JOIN dim_event_type et ON e.event_type_id = et.event_type_id OR es.event_type_id = et.event_type_id
LEFT JOIN dim_event_category ec ON e.event_category_id = ec.event_category_id OR et.event_category_id = ec.event_category_id;

-- Index on materialized view
CREATE INDEX IF NOT EXISTS idx_mv_event_hierarchy_id ON mv_event_hierarchy(event_id);
CREATE INDEX IF NOT EXISTS idx_mv_event_hierarchy_subtype ON mv_event_hierarchy(event_subtype);
CREATE INDEX IF NOT EXISTS idx_mv_event_hierarchy_type ON mv_event_hierarchy(event_type);
CREATE INDEX IF NOT EXISTS idx_mv_event_hierarchy_category ON mv_event_hierarchy(event_category);

-- Materialized view for complete entity hierarchy
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_entity_hierarchy AS
SELECT
    e.entity_id,
    e.original_text,
    e.entity_name,
    e.confidence,
    e.attributes,

    -- Type level
    et.type_name AS entity_type,
    et.type_description AS type_description,

    -- Category level
    ec.category_name AS entity_category,
    ec.category_description AS category_description

FROM dim_entity e
LEFT JOIN dim_entity_type et ON e.entity_type_id = et.entity_type_id
LEFT JOIN dim_entity_category ec ON e.entity_category_id = ec.entity_category_id OR et.entity_category_id = ec.entity_category_id;

-- Index on materialized view
CREATE INDEX IF NOT EXISTS idx_mv_entity_hierarchy_id ON mv_entity_hierarchy(entity_id);
CREATE INDEX IF NOT EXISTS idx_mv_entity_hierarchy_name ON mv_entity_hierarchy(entity_name);
CREATE INDEX IF NOT EXISTS idx_mv_entity_hierarchy_type ON mv_entity_hierarchy(entity_type);
CREATE INDEX IF NOT EXISTS idx_mv_entity_hierarchy_category ON mv_entity_hierarchy(entity_category);

-- View for event with full hierarchy path
CREATE VIEW IF NOT EXISTS v_event_with_path AS
SELECT
    e.event_id,
    e.original_text,
    e.severity,

    -- Full hierarchy path
    CONCAT_WS(' > ',
        ec.category_name,
        et.type_name,
        es.subtype_name
    ) AS hierarchy_path

FROM dim_event e
LEFT JOIN dim_event_subtype es ON e.event_subtype_id = es.event_subtype_id
LEFT JOIN dim_event_type et ON e.event_type_id = et.event_type_id OR es.event_type_id = et.event_type_id
LEFT JOIN dim_event_category ec ON e.event_category_id = ec.event_category_id OR et.event_category_id = ec.event_category_id;

-- View for entity with full hierarchy path
CREATE VIEW IF NOT EXISTS v_entity_with_path AS
SELECT
    e.entity_id,
    e.original_text,
    e.entity_name,

    -- Full hierarchy path
    CONCAT_WS(' > ',
        ec.category_name,
        et.type_name,
        e.entity_name
    ) AS hierarchy_path

FROM dim_entity e
LEFT JOIN dim_entity_type et ON e.entity_type_id = et.entity_type_id
LEFT JOIN dim_entity_category ec ON e.entity_category_id = ec.entity_category_id OR et.entity_category_id = ec.entity_category_id;

-- ------------------------------------------------------------------------------
-- Comments
-- ------------------------------------------------------------------------------
COMMENT ON TABLE dim_event_category IS 'Event category hierarchy dimension table (top level)';
COMMENT ON TABLE dim_event_type IS 'Event type hierarchy dimension table (middle level)';
COMMENT ON TABLE dim_event_subtype IS 'Event subtype hierarchy dimension table (bottom level)';
COMMENT ON TABLE dim_event IS 'Base event dimension linking to extracted entities';
COMMENT ON TABLE dim_entity_category IS 'Entity category hierarchy dimension table (top level)';
COMMENT ON TABLE dim_entity_type IS 'Entity type hierarchy dimension table (bottom level)';
COMMENT ON TABLE dim_entity IS 'Base entity dimension linking to extracted entities';
COMMENT ON TABLE dim_document IS 'Document metadata dimension (flat structure)';
COMMENT ON MATERIALIZED VIEW mv_event_hierarchy IS 'Denormalized view of complete event hierarchy for performance';
COMMENT ON MATERIALIZED VIEW mv_entity_hierarchy IS 'Denormalized view of complete entity hierarchy for performance';
COMMENT ON VIEW v_event_with_path IS 'View showing full hierarchy path for each event';
COMMENT ON VIEW v_entity_with_path IS 'View showing full hierarchy path for each entity';
