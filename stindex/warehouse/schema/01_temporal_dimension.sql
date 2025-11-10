-- ==============================================================================
-- Temporal Dimension Schema (Snowflake Pattern)
-- ==============================================================================
-- Description: Snowflake schema for temporal dimension with hierarchical tables
-- Hierarchy: Year → Quarter → Month → Day
-- Based on: dimensional-data-warehouse.md research findings
-- ==============================================================================

-- ------------------------------------------------------------------------------
-- Year Hierarchy Table
-- ------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS dim_year (
    year_id               BIGSERIAL PRIMARY KEY,
    year                  INT UNIQUE NOT NULL,
    decade                INT NOT NULL,
    century               INT NOT NULL,
    is_leap_year          BOOLEAN NOT NULL,

    -- Metadata
    created_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index for year lookups
CREATE INDEX IF NOT EXISTS idx_dim_year_year ON dim_year(year);

-- ------------------------------------------------------------------------------
-- Quarter Hierarchy Table
-- ------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS dim_quarter (
    quarter_id            BIGSERIAL PRIMARY KEY,
    quarter               INT NOT NULL CHECK (quarter BETWEEN 1 AND 4),
    quarter_name          VARCHAR(10) NOT NULL,  -- 'Q1', 'Q2', 'Q3', 'Q4'
    year_id               BIGINT NOT NULL REFERENCES dim_year(year_id),

    -- Metadata
    created_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Ensure unique quarters per year
    UNIQUE(year_id, quarter)
);

-- Index for quarter lookups
CREATE INDEX IF NOT EXISTS idx_dim_quarter_year ON dim_quarter(year_id);

-- ------------------------------------------------------------------------------
-- Month Hierarchy Table
-- ------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS dim_month (
    month_id              BIGSERIAL PRIMARY KEY,
    month                 INT NOT NULL CHECK (month BETWEEN 1 AND 12),
    month_name            VARCHAR(10) NOT NULL,  -- 'January', 'February', ...
    month_abbr            CHAR(3) NOT NULL,      -- 'Jan', 'Feb', ...
    quarter_id            BIGINT NOT NULL REFERENCES dim_quarter(quarter_id),

    -- Metadata
    created_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Ensure unique months per quarter
    UNIQUE(quarter_id, month)
);

-- Index for month lookups
CREATE INDEX IF NOT EXISTS idx_dim_month_quarter ON dim_month(quarter_id);

-- ------------------------------------------------------------------------------
-- Date Hierarchy Table (Base Table)
-- ------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS dim_date (
    date_id               BIGSERIAL PRIMARY KEY,
    full_date             DATE UNIQUE NOT NULL,

    -- Day level attributes
    day_of_week           INT NOT NULL CHECK (day_of_week BETWEEN 0 AND 6),  -- 0=Monday, 6=Sunday
    day_name              VARCHAR(10) NOT NULL,   -- 'Monday', 'Tuesday', ...
    day_abbr              CHAR(3) NOT NULL,       -- 'Mon', 'Tue', ...
    day_of_month          INT NOT NULL CHECK (day_of_month BETWEEN 1 AND 31),
    day_of_year           INT NOT NULL CHECK (day_of_year BETWEEN 1 AND 366),
    is_weekend            BOOLEAN NOT NULL,
    is_holiday            BOOLEAN DEFAULT FALSE,
    holiday_name          VARCHAR(100),

    -- Week level attributes
    week_of_year          INT NOT NULL CHECK (week_of_year BETWEEN 1 AND 53),
    week_of_month         INT NOT NULL CHECK (week_of_month BETWEEN 1 AND 6),
    iso_week              INT NOT NULL CHECK (iso_week BETWEEN 1 AND 53),  -- ISO 8601 week number

    -- Foreign key to month (snowflake pattern)
    month_id              BIGINT NOT NULL REFERENCES dim_month(month_id),

    -- Denormalized quarter and year for performance (hybrid approach)
    quarter               INT NOT NULL CHECK (quarter BETWEEN 1 AND 4),
    quarter_name          VARCHAR(10) NOT NULL,
    year                  INT NOT NULL,

    -- Business calendar attributes
    fiscal_year           INT,
    fiscal_quarter        INT CHECK (fiscal_quarter BETWEEN 1 AND 4),
    fiscal_month          INT CHECK (fiscal_month BETWEEN 1 AND 12),

    -- Additional attributes
    season                VARCHAR(10),  -- 'Spring', 'Summer', 'Autumn', 'Winter'

    -- Metadata
    created_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for date lookups
CREATE INDEX IF NOT EXISTS idx_dim_date_full_date ON dim_date(full_date);
CREATE INDEX IF NOT EXISTS idx_dim_date_month ON dim_date(month_id);
CREATE INDEX IF NOT EXISTS idx_dim_date_year ON dim_date(year);
CREATE INDEX IF NOT EXISTS idx_dim_date_quarter ON dim_date(quarter, year);
CREATE INDEX IF NOT EXISTS idx_dim_date_is_weekend ON dim_date(is_weekend);
CREATE INDEX IF NOT EXISTS idx_dim_date_is_holiday ON dim_date(is_holiday);

-- ------------------------------------------------------------------------------
-- Time Dimension Table (for datetime/time temporals)
-- ------------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS dim_time (
    time_id               BIGSERIAL PRIMARY KEY,
    full_time             TIME UNIQUE NOT NULL,

    -- Second level
    second                INT NOT NULL CHECK (second BETWEEN 0 AND 59),

    -- Minute level
    minute                INT NOT NULL CHECK (minute BETWEEN 0 AND 59),
    minute_of_day         INT NOT NULL CHECK (minute_of_day BETWEEN 0 AND 1439),

    -- Hour level
    hour                  INT NOT NULL CHECK (hour BETWEEN 0 AND 23),
    hour_12               INT NOT NULL CHECK (hour_12 BETWEEN 1 AND 12),
    am_pm                 VARCHAR(2) NOT NULL CHECK (am_pm IN ('AM', 'PM')),

    -- Time period categories
    time_period           VARCHAR(20) NOT NULL,  -- 'Early Morning', 'Morning', 'Afternoon', 'Evening', 'Night'
    business_hours        BOOLEAN NOT NULL,      -- Is within standard business hours (9 AM - 5 PM)?

    -- Metadata
    created_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Index for time lookups
CREATE INDEX IF NOT EXISTS idx_dim_time_full_time ON dim_time(full_time);
CREATE INDEX IF NOT EXISTS idx_dim_time_hour ON dim_time(hour);
CREATE INDEX IF NOT EXISTS idx_dim_time_business_hours ON dim_time(business_hours);

-- ------------------------------------------------------------------------------
-- Base Temporal Dimension Table
-- ------------------------------------------------------------------------------
-- This table links to extracted temporal entities and connects to hierarchy
CREATE TABLE IF NOT EXISTS dim_temporal (
    temporal_id           BIGSERIAL PRIMARY KEY,

    -- Original extraction data
    original_text         TEXT NOT NULL,                     -- "March 15, 2022"
    normalized_value      TEXT NOT NULL,                     -- "2022-03-15" (ISO 8601)
    temporal_type         VARCHAR(20) NOT NULL,              -- 'date', 'time', 'datetime', 'duration', 'period'

    -- Foreign keys to hierarchy tables (nullable for durations/periods)
    date_id               BIGINT REFERENCES dim_date(date_id),
    time_id               BIGINT REFERENCES dim_time(time_id),

    -- For date ranges/periods (start and end dates)
    start_date_id         BIGINT REFERENCES dim_date(date_id),
    end_date_id           BIGINT REFERENCES dim_date(date_id),

    -- Extraction metadata
    confidence            FLOAT CHECK (confidence BETWEEN 0 AND 1),
    extraction_method     VARCHAR(50),                       -- 'llm', 'rule_based', 'hybrid'

    -- Additional attributes
    granularity           VARCHAR(20),                       -- 'year', 'quarter', 'month', 'week', 'day', 'hour', 'minute', 'second'
    is_approximate        BOOLEAN DEFAULT FALSE,             -- e.g., "early 2022", "around March"

    -- Metadata
    created_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at            TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Ensure unique temporal entities (by normalized value and type)
    UNIQUE(normalized_value, temporal_type)
);

-- Indexes for temporal lookups
CREATE INDEX IF NOT EXISTS idx_dim_temporal_normalized ON dim_temporal(normalized_value);
CREATE INDEX IF NOT EXISTS idx_dim_temporal_type ON dim_temporal(temporal_type);
CREATE INDEX IF NOT EXISTS idx_dim_temporal_date ON dim_temporal(date_id);
CREATE INDEX IF NOT EXISTS idx_dim_temporal_time ON dim_temporal(time_id);
CREATE INDEX IF NOT EXISTS idx_dim_temporal_confidence ON dim_temporal(confidence);
CREATE INDEX IF NOT EXISTS idx_dim_temporal_granularity ON dim_temporal(granularity);

-- ------------------------------------------------------------------------------
-- Helper Functions
-- ------------------------------------------------------------------------------

-- Function to get season from month (Northern Hemisphere)
CREATE OR REPLACE FUNCTION get_season(month INT) RETURNS VARCHAR(10) AS $$
BEGIN
    RETURN CASE
        WHEN month IN (12, 1, 2) THEN 'Winter'
        WHEN month IN (3, 4, 5) THEN 'Spring'
        WHEN month IN (6, 7, 8) THEN 'Summer'
        WHEN month IN (9, 10, 11) THEN 'Autumn'
    END;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Function to get time period from hour
CREATE OR REPLACE FUNCTION get_time_period(hour INT) RETURNS VARCHAR(20) AS $$
BEGIN
    RETURN CASE
        WHEN hour >= 0 AND hour < 6 THEN 'Night'
        WHEN hour >= 6 AND hour < 9 THEN 'Early Morning'
        WHEN hour >= 9 AND hour < 12 THEN 'Morning'
        WHEN hour >= 12 AND hour < 17 THEN 'Afternoon'
        WHEN hour >= 17 AND hour < 21 THEN 'Evening'
        ELSE 'Night'
    END;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- Function to check if time is within business hours (9 AM - 5 PM)
CREATE OR REPLACE FUNCTION is_business_hours(hour INT) RETURNS BOOLEAN AS $$
BEGIN
    RETURN hour >= 9 AND hour < 17;
END;
$$ LANGUAGE plpgsql IMMUTABLE;

-- ------------------------------------------------------------------------------
-- Views for Easy Querying
-- ------------------------------------------------------------------------------

-- Materialized view for complete temporal hierarchy (performance optimization)
CREATE MATERIALIZED VIEW IF NOT EXISTS mv_temporal_hierarchy AS
SELECT
    t.temporal_id,
    t.original_text,
    t.normalized_value,
    t.temporal_type,
    t.confidence,

    -- Date hierarchy
    d.full_date,
    d.day_name,
    d.day_of_month,
    d.is_weekend,
    d.is_holiday,

    -- Month level
    m.month,
    m.month_name,
    m.month_abbr,

    -- Quarter level
    q.quarter,
    q.quarter_name,

    -- Year level
    y.year,
    y.decade,
    y.is_leap_year

FROM dim_temporal t
LEFT JOIN dim_date d ON t.date_id = d.date_id
LEFT JOIN dim_month m ON d.month_id = m.month_id
LEFT JOIN dim_quarter q ON m.quarter_id = q.quarter_id
LEFT JOIN dim_year y ON q.year_id = y.year_id;

-- Index on materialized view
CREATE INDEX IF NOT EXISTS idx_mv_temporal_hierarchy_id ON mv_temporal_hierarchy(temporal_id);
CREATE INDEX IF NOT EXISTS idx_mv_temporal_hierarchy_year ON mv_temporal_hierarchy(year);
CREATE INDEX IF NOT EXISTS idx_mv_temporal_hierarchy_quarter ON mv_temporal_hierarchy(year, quarter);
CREATE INDEX IF NOT EXISTS idx_mv_temporal_hierarchy_month ON mv_temporal_hierarchy(year, month);
CREATE INDEX IF NOT EXISTS idx_mv_temporal_hierarchy_date ON mv_temporal_hierarchy(full_date);

-- ------------------------------------------------------------------------------
-- Comments
-- ------------------------------------------------------------------------------
COMMENT ON TABLE dim_year IS 'Year hierarchy dimension table';
COMMENT ON TABLE dim_quarter IS 'Quarter hierarchy dimension table';
COMMENT ON TABLE dim_month IS 'Month hierarchy dimension table';
COMMENT ON TABLE dim_date IS 'Date dimension with full temporal attributes';
COMMENT ON TABLE dim_time IS 'Time of day dimension';
COMMENT ON TABLE dim_temporal IS 'Base temporal dimension linking to extracted entities';
COMMENT ON MATERIALIZED VIEW mv_temporal_hierarchy IS 'Denormalized view of complete temporal hierarchy for performance';
