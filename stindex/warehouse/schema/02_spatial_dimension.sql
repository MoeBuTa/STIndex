-- ==============================================================================
-- Spatial Dimension Schema (Snowflake Pattern)
-- ==============================================================================
-- Description: Snowflake schema for spatial dimension with geographic hierarchy
-- Hierarchy: Continent → Country → State/Province → Region → City → Suburb → Address
-- Based on: dimensional-data-warehouse.md research findings
-- Requires: PostGIS extension for geographic data types
-- ==============================================================================

-- Enable PostGIS extension
CREATE EXTENSION IF NOT EXISTS postgis;

-- ------------------------------------------------------------------------------
-- Continent Hierarchy Table
-- ------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS dim_continent (
    continent_id          BIGSERIAL PRIMARY KEY,
    continent_name        VARCHAR(50) UNIQUE NOT NULL,  -- 'Oceania', 'Asia', 'Europe', etc.
    continent_code        VARCHAR(2) UNIQUE,            -- 'OC', 'AS', 'EU', etc.

    -- Metadata
    created_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index for continent lookups
CREATE INDEX IF NOT EXISTS idx_dim_continent_name ON dim_continent(continent_name);

-- ------------------------------------------------------------------------------
-- Country Hierarchy Table
-- ------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS dim_country (
    country_id            BIGSERIAL PRIMARY KEY,
    country_name          VARCHAR(100) NOT NULL,
    country_code_iso2     CHAR(2) UNIQUE NOT NULL,      -- 'AU', 'US', 'GB', etc.
    country_code_iso3     CHAR(3) UNIQUE NOT NULL,      -- 'AUS', 'USA', 'GBR', etc.
    continent_id          BIGINT NOT NULL REFERENCES dim_continent(continent_id),

    -- Geographic attributes
    capital               VARCHAR(100),
    currency              VARCHAR(50),
    phone_code            VARCHAR(10),
    population            BIGINT,
    area_km2              DECIMAL(15, 2),

    -- Bounding box for spatial queries
    bbox_north            DECIMAL(10, 7),
    bbox_south            DECIMAL(10, 7),
    bbox_east             DECIMAL(10, 7),
    bbox_west             DECIMAL(10, 7),

    -- Metadata
    created_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Ensure unique country names per continent
    UNIQUE(continent_id, country_name)
);

-- Indexes for country lookups
CREATE INDEX IF NOT EXISTS idx_dim_country_name ON dim_country(country_name);
CREATE INDEX IF NOT EXISTS idx_dim_country_iso2 ON dim_country(country_code_iso2);
CREATE INDEX IF NOT EXISTS idx_dim_country_iso3 ON dim_country(country_code_iso3);
CREATE INDEX IF NOT EXISTS idx_dim_country_continent ON dim_country(continent_id);

-- ------------------------------------------------------------------------------
-- State/Province Hierarchy Table
-- ------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS dim_state (
    state_id              BIGSERIAL PRIMARY KEY,
    state_name            VARCHAR(100) NOT NULL,
    state_code            VARCHAR(10),                  -- 'WA', 'NSW', 'CA', etc.
    state_type            VARCHAR(20),                  -- 'state', 'province', 'territory', 'region'
    country_id            BIGINT NOT NULL REFERENCES dim_country(country_id),

    -- Geographic attributes
    capital               VARCHAR(100),
    population            BIGINT,
    area_km2              DECIMAL(15, 2),

    -- Bounding box
    bbox_north            DECIMAL(10, 7),
    bbox_south            DECIMAL(10, 7),
    bbox_east             DECIMAL(10, 7),
    bbox_west             DECIMAL(10, 7),

    -- Metadata
    created_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Ensure unique states per country
    UNIQUE(country_id, state_name)
);

-- Indexes for state lookups
CREATE INDEX IF NOT EXISTS idx_dim_state_name ON dim_state(state_name);
CREATE INDEX IF NOT EXISTS idx_dim_state_code ON dim_state(state_code);
CREATE INDEX IF NOT EXISTS idx_dim_state_country ON dim_state(country_id);

-- ------------------------------------------------------------------------------
-- Region Hierarchy Table
-- ------------------------------------------------------------------------------
-- Regions are sub-state administrative divisions (e.g., "Kimberley" in WA)
CREATE TABLE IF NOT EXISTS dim_region (
    region_id             BIGSERIAL PRIMARY KEY,
    region_name           VARCHAR(100) NOT NULL,
    region_type           VARCHAR(20),                  -- 'region', 'district', 'county', etc.
    state_id              BIGINT NOT NULL REFERENCES dim_state(state_id),

    -- Geographic attributes
    population            BIGINT,
    area_km2              DECIMAL(15, 2),

    -- Bounding box
    bbox_north            DECIMAL(10, 7),
    bbox_south            DECIMAL(10, 7),
    bbox_east             DECIMAL(10, 7),
    bbox_west             DECIMAL(10, 7),

    -- Metadata
    created_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Ensure unique regions per state
    UNIQUE(state_id, region_name)
);

-- Indexes for region lookups
CREATE INDEX IF NOT EXISTS idx_dim_region_name ON dim_region(region_name);
CREATE INDEX IF NOT EXISTS idx_dim_region_state ON dim_region(state_id);

-- ------------------------------------------------------------------------------
-- City Hierarchy Table
-- ------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS dim_city (
    city_id               BIGSERIAL PRIMARY KEY,
    city_name             VARCHAR(100) NOT NULL,
    city_type             VARCHAR(20),                  -- 'city', 'town', 'village', 'municipality'

    -- Foreign keys (city can belong to region or directly to state)
    region_id             BIGINT REFERENCES dim_region(region_id),
    state_id              BIGINT NOT NULL REFERENCES dim_state(state_id),

    -- Geographic coordinates (city center)
    latitude              DECIMAL(10, 7),
    longitude             DECIMAL(10, 7),
    elevation_m           INT,

    -- PostGIS geometry (POINT)
    geom                  GEOGRAPHY(POINT, 4326),

    -- Additional attributes
    population            INT,
    area_km2              DECIMAL(10, 2),
    timezone              VARCHAR(50),

    -- Metadata
    created_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Ensure unique cities per state
    UNIQUE(state_id, city_name)
);

-- Indexes for city lookups
CREATE INDEX IF NOT EXISTS idx_dim_city_name ON dim_city(city_name);
CREATE INDEX IF NOT EXISTS idx_dim_city_region ON dim_city(region_id);
CREATE INDEX IF NOT EXISTS idx_dim_city_state ON dim_city(state_id);
CREATE INDEX IF NOT EXISTS idx_dim_city_geom ON dim_city USING GIST(geom);

-- ------------------------------------------------------------------------------
-- Suburb Hierarchy Table
-- ------------------------------------------------------------------------------
-- Suburbs are neighborhoods or districts within cities
CREATE TABLE IF NOT EXISTS dim_suburb (
    suburb_id             BIGSERIAL PRIMARY KEY,
    suburb_name           VARCHAR(100) NOT NULL,
    city_id               BIGINT NOT NULL REFERENCES dim_city(city_id),

    -- Geographic coordinates (suburb center)
    latitude              DECIMAL(10, 7),
    longitude             DECIMAL(10, 7),
    geom                  GEOGRAPHY(POINT, 4326),

    -- Additional attributes
    postal_code           VARCHAR(20),
    population            INT,
    area_km2              DECIMAL(10, 2),

    -- Metadata
    created_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Ensure unique suburbs per city
    UNIQUE(city_id, suburb_name)
);

-- Indexes for suburb lookups
CREATE INDEX IF NOT EXISTS idx_dim_suburb_name ON dim_suburb(suburb_name);
CREATE INDEX IF NOT EXISTS idx_dim_suburb_city ON dim_suburb(city_id);
CREATE INDEX IF NOT EXISTS idx_dim_suburb_postal ON dim_suburb(postal_code);
CREATE INDEX IF NOT EXISTS idx_dim_suburb_geom ON dim_suburb USING GIST(geom);

-- ------------------------------------------------------------------------------
-- Address Hierarchy Table
-- ------------------------------------------------------------------------------
-- Addresses are specific street addresses
CREATE TABLE IF NOT EXISTS dim_address (
    address_id            BIGSERIAL PRIMARY KEY,
    street_number         VARCHAR(50),
    street_name           VARCHAR(200) NOT NULL,
    unit_number           VARCHAR(50),
    postal_code           VARCHAR(20),
    suburb_id             BIGINT NOT NULL REFERENCES dim_suburb(suburb_id),

    -- Geographic coordinates (address location)
    latitude              DECIMAL(10, 7),
    longitude             DECIMAL(10, 7),
    geom                  GEOGRAPHY(POINT, 4326),

    -- Full formatted address
    formatted_address     TEXT,

    -- Metadata
    created_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for address lookups
CREATE INDEX IF NOT EXISTS idx_dim_address_street ON dim_address(street_name);
CREATE INDEX IF NOT EXISTS idx_dim_address_postal ON dim_address(postal_code);
CREATE INDEX IF NOT EXISTS idx_dim_address_suburb ON dim_address(suburb_id);
CREATE INDEX IF NOT EXISTS idx_dim_address_geom ON dim_address USING GIST(geom);

-- ------------------------------------------------------------------------------
-- Base Spatial Dimension Table
-- ------------------------------------------------------------------------------
-- This table links to extracted spatial entities and connects to hierarchy
CREATE TABLE IF NOT EXISTS dim_spatial (
    spatial_id            BIGSERIAL PRIMARY KEY,

    -- Original extraction data
    original_text         TEXT NOT NULL,                -- "Broome", "Western Australia", etc.
    location_type         VARCHAR(20) NOT NULL,         -- 'address', 'suburb', 'city', 'region', 'state', 'country', 'continent'

    -- Geocoded coordinates
    latitude              DECIMAL(10, 7),
    longitude             DECIMAL(10, 7),
    geom                  GEOGRAPHY(POINT, 4326),       -- PostGIS POINT geometry

    -- Foreign keys to hierarchy (snowflake pattern)
    -- Depending on location_type, different FKs will be populated
    address_id            BIGINT REFERENCES dim_address(address_id),
    suburb_id             BIGINT REFERENCES dim_suburb(suburb_id),
    city_id               BIGINT REFERENCES dim_city(city_id),
    region_id             BIGINT REFERENCES dim_region(region_id),
    state_id              BIGINT REFERENCES dim_state(state_id),
    country_id            BIGINT REFERENCES dim_country(country_id),
    continent_id          BIGINT REFERENCES dim_continent(continent_id),

    -- Extraction metadata
    confidence            FLOAT CHECK (confidence BETWEEN 0 AND 1),
    extraction_method     VARCHAR(50),                  -- 'llm', 'geocoding', 'hybrid'
    geocoding_provider    VARCHAR(50),                  -- 'nominatim', 'google', 'osm', etc.

    -- Parent region context (from LLM extraction)
    parent_region         TEXT,                         -- e.g., "Western Australia" for "Broome"

    -- Additional attributes
    population            INT,
    area_km2              DECIMAL(10, 2),
    elevation_m           INT,
    timezone              VARCHAR(50),

    -- Metadata
    created_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Ensure unique spatial entities (by original text and geocoded location)
    UNIQUE(original_text, latitude, longitude)
);

-- Indexes for spatial lookups
CREATE INDEX IF NOT EXISTS idx_dim_spatial_original_text ON dim_spatial(original_text);
CREATE INDEX IF NOT EXISTS idx_dim_spatial_type ON dim_spatial(location_type);
CREATE INDEX IF NOT EXISTS idx_dim_spatial_geom ON dim_spatial USING GIST(geom);
CREATE INDEX IF NOT EXISTS idx_dim_spatial_city ON dim_spatial(city_id);
CREATE INDEX IF NOT EXISTS idx_dim_spatial_region ON dim_spatial(region_id);
CREATE INDEX IF NOT EXISTS idx_dim_spatial_state ON dim_spatial(state_id);
CREATE INDEX IF NOT EXISTS idx_dim_spatial_country ON dim_spatial(country_id);
CREATE INDEX IF NOT EXISTS idx_dim_spatial_confidence ON dim_spatial(confidence);

-- ------------------------------------------------------------------------------
-- Helper Functions
-- ------------------------------------------------------------------------------

-- Function to calculate distance between two points (in kilometers)
CREATE OR REPLACE FUNCTION calculate_distance(
    lat1 DECIMAL, lon1 DECIMAL,
    lat2 DECIMAL, lon2 DECIMAL
) RETURNS DECIMAL AS $$
BEGIN
    RETURN ST_Distance(
        ST_MakePoint(lon1, lat1)::geography,
        ST_MakePoint(lon2, lat2)::geography
    ) / 1000.0;  -- Convert meters to kilometers
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Function to get all locations within radius (in kilometers)
CREATE OR REPLACE FUNCTION get_locations_within_radius(
    center_lat DECIMAL,
    center_lon DECIMAL,
    radius_km DECIMAL
) RETURNS TABLE (
    spatial_id BIGINT,
    original_text TEXT,
    latitude DECIMAL,
    longitude DECIMAL,
    distance_km DECIMAL
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        s.spatial_id,
        s.original_text,
        s.latitude,
        s.longitude,
        ST_Distance(
            s.geom,
            ST_MakePoint(center_lon, center_lat)::geography
        ) / 1000.0 AS distance_km
    FROM dim_spatial s
    WHERE ST_DWithin(
        s.geom,
        ST_MakePoint(center_lon, center_lat)::geography,
        radius_km * 1000  -- Convert km to meters
    )
    ORDER BY distance_km;
END;
$$ LANGUAGE plpgsql;

-- ------------------------------------------------------------------------------
-- Views for Easy Querying
-- ------------------------------------------------------------------------------

-- Materialized view for complete spatial hierarchy (performance optimization)
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_spatial_hierarchy AS
SELECT
    s.spatial_id,
    s.original_text,
    s.location_type,
    s.latitude,
    s.longitude,
    s.confidence,

    -- City level
    c.city_name,
    c.city_type,

    -- Region level
    r.region_name,
    r.region_type,

    -- State level
    st.state_name,
    st.state_code,

    -- Country level
    co.country_name,
    co.country_code_iso2,
    co.country_code_iso3,

    -- Continent level
    cont.continent_name,
    cont.continent_code

FROM dim_spatial s
LEFT JOIN dim_city c ON s.city_id = c.city_id
LEFT JOIN dim_region r ON s.region_id = r.region_id
LEFT JOIN dim_state st ON s.state_id = st.state_id
LEFT JOIN dim_country co ON s.country_id = co.country_id
LEFT JOIN dim_continent cont ON s.continent_id = cont.continent_id;

-- Index on materialized view
CREATE INDEX IF NOT EXISTS idx_mv_spatial_hierarchy_id ON mv_spatial_hierarchy(spatial_id);
CREATE INDEX IF NOT EXISTS idx_mv_spatial_hierarchy_city ON mv_spatial_hierarchy(city_name);
CREATE INDEX IF NOT EXISTS idx_mv_spatial_hierarchy_region ON mv_spatial_hierarchy(region_name);
CREATE INDEX IF NOT EXISTS idx_mv_spatial_hierarchy_state ON mv_spatial_hierarchy(state_name);
CREATE INDEX IF NOT EXISTS idx_mv_spatial_hierarchy_country ON mv_spatial_hierarchy(country_name);
CREATE INDEX IF NOT EXISTS idx_mv_spatial_hierarchy_continent ON mv_spatial_hierarchy(continent_name);

-- View for spatial entity with full hierarchy path
CREATE VIEW IF NOT EXISTS v_spatial_with_path AS
SELECT
    s.spatial_id,
    s.original_text,
    s.location_type,
    s.latitude,
    s.longitude,

    -- Full hierarchy path
    CONCAT_WS(' > ',
        cont.continent_name,
        co.country_name,
        st.state_name,
        r.region_name,
        c.city_name,
        sub.suburb_name,
        a.formatted_address
    ) AS hierarchy_path

FROM dim_spatial s
LEFT JOIN dim_address a ON s.address_id = a.address_id
LEFT JOIN dim_suburb sub ON s.suburb_id = sub.suburb_id
LEFT JOIN dim_city c ON s.city_id = c.city_id
LEFT JOIN dim_region r ON s.region_id = r.region_id
LEFT JOIN dim_state st ON s.state_id = st.state_id
LEFT JOIN dim_country co ON s.country_id = co.country_id
LEFT JOIN dim_continent cont ON s.continent_id = cont.continent_id;

-- ------------------------------------------------------------------------------
-- Comments
-- ------------------------------------------------------------------------------
COMMENT ON TABLE dim_continent IS 'Continent hierarchy dimension table';
COMMENT ON TABLE dim_country IS 'Country hierarchy dimension table';
COMMENT ON TABLE dim_state IS 'State/Province hierarchy dimension table';
COMMENT ON TABLE dim_region IS 'Region/District hierarchy dimension table';
COMMENT ON TABLE dim_city IS 'City hierarchy dimension table';
COMMENT ON TABLE dim_suburb IS 'Suburb/Neighborhood hierarchy dimension table';
COMMENT ON TABLE dim_address IS 'Address dimension table';
COMMENT ON TABLE dim_spatial IS 'Base spatial dimension linking to extracted entities';
COMMENT ON MATERIALIZED VIEW mv_spatial_hierarchy IS 'Denormalized view of complete spatial hierarchy for performance';
COMMENT ON VIEW v_spatial_with_path IS 'View showing full hierarchy path for each spatial entity';
