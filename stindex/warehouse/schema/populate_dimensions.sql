-- ==============================================================================
-- Pre-populate Dimension Tables with Reference Data
-- ==============================================================================
-- Description: Populate dimension tables with reference data
-- Tables populated:
--   - dim_year, dim_quarter, dim_month, dim_date (2000-2050)
--   - dim_time (all times of day)
--   - dim_continent, dim_country (major countries)
--   - dim_event_category, dim_event_type (common event taxonomies)
--   - dim_entity_category, dim_entity_type (common entity types)
-- ==============================================================================

\c stindex_warehouse

-- ==============================================================================
-- Temporal Dimensions: Pre-populate Date/Time Tables
-- ==============================================================================
\echo 'Populating temporal dimension tables...'

-- Populate dim_year (2000-2050)
INSERT INTO dim_year (year, decade, century, is_leap_year)
SELECT
    year_val AS year,
    (year_val / 10) * 10 AS decade,
    (year_val / 100) * 100 AS century,
    (year_val % 4 = 0 AND (year_val % 100 != 0 OR year_val % 400 = 0)) AS is_leap_year
FROM generate_series(2000, 2050) AS year_val
ON CONFLICT (year) DO NOTHING;

-- Populate dim_quarter (4 quarters per year, 2000-2050)
INSERT INTO dim_quarter (quarter, quarter_name, year_id)
SELECT
    q AS quarter,
    'Q' || q AS quarter_name,
    y.year_id
FROM generate_series(1, 4) AS q
CROSS JOIN dim_year y
ON CONFLICT (year_id, quarter) DO NOTHING;

-- Populate dim_month (12 months per quarter, 2000-2050)
INSERT INTO dim_month (month, month_name, month_abbr, quarter_id)
SELECT
    m AS month,
    CASE m
        WHEN 1 THEN 'January'
        WHEN 2 THEN 'February'
        WHEN 3 THEN 'March'
        WHEN 4 THEN 'April'
        WHEN 5 THEN 'May'
        WHEN 6 THEN 'June'
        WHEN 7 THEN 'July'
        WHEN 8 THEN 'August'
        WHEN 9 THEN 'September'
        WHEN 10 THEN 'October'
        WHEN 11 THEN 'November'
        WHEN 12 THEN 'December'
    END AS month_name,
    CASE m
        WHEN 1 THEN 'Jan'
        WHEN 2 THEN 'Feb'
        WHEN 3 THEN 'Mar'
        WHEN 4 THEN 'Apr'
        WHEN 5 THEN 'May'
        WHEN 6 THEN 'Jun'
        WHEN 7 THEN 'Jul'
        WHEN 8 THEN 'Aug'
        WHEN 9 THEN 'Sep'
        WHEN 10 THEN 'Oct'
        WHEN 11 THEN 'Nov'
        WHEN 12 THEN 'Dec'
    END AS month_abbr,
    q.quarter_id
FROM generate_series(1, 12) AS m
CROSS JOIN dim_quarter q
WHERE q.quarter = CEIL(m / 3.0)
ON CONFLICT (quarter_id, month) DO NOTHING;

-- Populate dim_date (all dates from 2000-01-01 to 2050-12-31)
\echo 'Populating dim_date (this may take a minute)...'
INSERT INTO dim_date (
    full_date,
    day_of_week,
    day_name,
    day_abbr,
    day_of_month,
    day_of_year,
    is_weekend,
    week_of_year,
    week_of_month,
    iso_week,
    month_id,
    quarter,
    quarter_name,
    year,
    fiscal_year,
    fiscal_quarter,
    fiscal_month,
    season
)
SELECT
    date_series::DATE AS full_date,
    EXTRACT(ISODOW FROM date_series) - 1 AS day_of_week,  -- 0=Monday, 6=Sunday
    TO_CHAR(date_series, 'Day') AS day_name,
    SUBSTRING(TO_CHAR(date_series, 'Day'), 1, 3) AS day_abbr,
    EXTRACT(DAY FROM date_series) AS day_of_month,
    EXTRACT(DOY FROM date_series) AS day_of_year,
    EXTRACT(ISODOW FROM date_series) IN (6, 7) AS is_weekend,
    EXTRACT(WEEK FROM date_series) AS week_of_year,
    CEIL(EXTRACT(DAY FROM date_series) / 7.0) AS week_of_month,
    EXTRACT(WEEK FROM date_series) AS iso_week,
    m.month_id,
    EXTRACT(QUARTER FROM date_series) AS quarter,
    'Q' || EXTRACT(QUARTER FROM date_series) AS quarter_name,
    EXTRACT(YEAR FROM date_series) AS year,
    EXTRACT(YEAR FROM date_series) AS fiscal_year,  -- Assuming calendar year = fiscal year
    EXTRACT(QUARTER FROM date_series) AS fiscal_quarter,
    EXTRACT(MONTH FROM date_series) AS fiscal_month,
    get_season(EXTRACT(MONTH FROM date_series)::INT) AS season
FROM generate_series(
    '2000-01-01'::TIMESTAMP,
    '2050-12-31'::TIMESTAMP,
    '1 day'::INTERVAL
) AS date_series
JOIN dim_year y ON EXTRACT(YEAR FROM date_series) = y.year
JOIN dim_quarter q ON EXTRACT(QUARTER FROM date_series) = q.quarter AND q.year_id = y.year_id
JOIN dim_month m ON EXTRACT(MONTH FROM date_series) = m.month AND m.quarter_id = q.quarter_id
ON CONFLICT (full_date) DO NOTHING;

-- Populate dim_time (all times of day, one per minute)
\echo 'Populating dim_time...'
INSERT INTO dim_time (
    full_time,
    second,
    minute,
    minute_of_day,
    hour,
    hour_12,
    am_pm,
    time_period,
    business_hours
)
SELECT
    time_series::TIME AS full_time,
    EXTRACT(SECOND FROM time_series) AS second,
    EXTRACT(MINUTE FROM time_series) AS minute,
    EXTRACT(HOUR FROM time_series) * 60 + EXTRACT(MINUTE FROM time_series) AS minute_of_day,
    EXTRACT(HOUR FROM time_series) AS hour,
    CASE
        WHEN EXTRACT(HOUR FROM time_series) = 0 THEN 12
        WHEN EXTRACT(HOUR FROM time_series) <= 12 THEN EXTRACT(HOUR FROM time_series)
        ELSE EXTRACT(HOUR FROM time_series) - 12
    END AS hour_12,
    CASE
        WHEN EXTRACT(HOUR FROM time_series) < 12 THEN 'AM'
        ELSE 'PM'
    END AS am_pm,
    get_time_period(EXTRACT(HOUR FROM time_series)::INT) AS time_period,
    is_business_hours(EXTRACT(HOUR FROM time_series)::INT) AS business_hours
FROM generate_series(
    '00:00:00'::TIME,
    '23:59:00'::TIME,
    '1 minute'::INTERVAL
) AS time_series
ON CONFLICT (full_time) DO NOTHING;

\echo 'Temporal dimensions populated successfully!'

-- ==============================================================================
-- Spatial Dimensions: Pre-populate Continents and Major Countries
-- ==============================================================================
\echo 'Populating spatial dimension tables...'

-- Populate dim_continent
INSERT INTO dim_continent (continent_name, continent_code) VALUES
    ('Africa', 'AF'),
    ('Antarctica', 'AN'),
    ('Asia', 'AS'),
    ('Europe', 'EU'),
    ('North America', 'NA'),
    ('Oceania', 'OC'),
    ('South America', 'SA')
ON CONFLICT (continent_name) DO NOTHING;

-- Populate dim_country (major countries)
INSERT INTO dim_country (
    country_name,
    country_code_iso2,
    country_code_iso3,
    continent_id,
    capital,
    currency,
    phone_code
)
SELECT
    country_name,
    iso2,
    iso3,
    c.continent_id,
    capital,
    currency,
    phone_code
FROM (VALUES
    -- Oceania
    ('Australia', 'AU', 'AUS', 'Oceania', 'Canberra', 'AUD', '+61'),
    ('New Zealand', 'NZ', 'NZL', 'Oceania', 'Wellington', 'NZD', '+64'),
    ('Fiji', 'FJ', 'FJI', 'Oceania', 'Suva', 'FJD', '+679'),

    -- Asia
    ('China', 'CN', 'CHN', 'Asia', 'Beijing', 'CNY', '+86'),
    ('Japan', 'JP', 'JPN', 'Asia', 'Tokyo', 'JPY', '+81'),
    ('India', 'IN', 'IND', 'Asia', 'New Delhi', 'INR', '+91'),
    ('Indonesia', 'ID', 'IDN', 'Asia', 'Jakarta', 'IDR', '+62'),
    ('Singapore', 'SG', 'SGP', 'Asia', 'Singapore', 'SGD', '+65'),
    ('South Korea', 'KR', 'KOR', 'Asia', 'Seoul', 'KRW', '+82'),
    ('Thailand', 'TH', 'THA', 'Asia', 'Bangkok', 'THB', '+66'),
    ('Vietnam', 'VN', 'VNM', 'Asia', 'Hanoi', 'VND', '+84'),

    -- North America
    ('United States', 'US', 'USA', 'North America', 'Washington D.C.', 'USD', '+1'),
    ('Canada', 'CA', 'CAN', 'North America', 'Ottawa', 'CAD', '+1'),
    ('Mexico', 'MX', 'MEX', 'North America', 'Mexico City', 'MXN', '+52'),

    -- Europe
    ('United Kingdom', 'GB', 'GBR', 'Europe', 'London', 'GBP', '+44'),
    ('France', 'FR', 'FRA', 'Europe', 'Paris', 'EUR', '+33'),
    ('Germany', 'DE', 'DEU', 'Europe', 'Berlin', 'EUR', '+49'),
    ('Italy', 'IT', 'ITA', 'Europe', 'Rome', 'EUR', '+39'),
    ('Spain', 'ES', 'ESP', 'Europe', 'Madrid', 'EUR', '+34'),

    -- South America
    ('Brazil', 'BR', 'BRA', 'South America', 'Brasília', 'BRL', '+55'),
    ('Argentina', 'AR', 'ARG', 'South America', 'Buenos Aires', 'ARS', '+54'),
    ('Chile', 'CL', 'CHL', 'South America', 'Santiago', 'CLP', '+56'),

    -- Africa
    ('South Africa', 'ZA', 'ZAF', 'Africa', 'Pretoria', 'ZAR', '+27'),
    ('Egypt', 'EG', 'EGY', 'Africa', 'Cairo', 'EGP', '+20'),
    ('Nigeria', 'NG', 'NGA', 'Africa', 'Abuja', 'NGN', '+234')
) AS countries(country_name, iso2, iso3, continent_name, capital, currency, phone_code)
JOIN dim_continent c ON c.continent_name = countries.continent_name
ON CONFLICT (country_code_iso2) DO NOTHING;

-- Populate Australian states (example)
INSERT INTO dim_state (state_name, state_code, state_type, country_id, capital)
SELECT
    state_name,
    state_code,
    'state',
    co.country_id,
    capital
FROM (VALUES
    ('New South Wales', 'NSW', 'Sydney'),
    ('Victoria', 'VIC', 'Melbourne'),
    ('Queensland', 'QLD', 'Brisbane'),
    ('Western Australia', 'WA', 'Perth'),
    ('South Australia', 'SA', 'Adelaide'),
    ('Tasmania', 'TAS', 'Hobart'),
    ('Australian Capital Territory', 'ACT', 'Canberra'),
    ('Northern Territory', 'NT', 'Darwin')
) AS states(state_name, state_code, capital)
CROSS JOIN dim_country co
WHERE co.country_code_iso2 = 'AU'
ON CONFLICT (country_id, state_name) DO NOTHING;

\echo 'Spatial dimensions populated successfully!'

-- ==============================================================================
-- Event Dimensions: Pre-populate Event Taxonomies
-- ==============================================================================
\echo 'Populating event dimension tables...'

-- Populate dim_event_category
INSERT INTO dim_event_category (category_name, category_description) VALUES
    ('natural_disaster', 'Natural disasters and extreme weather events'),
    ('political_event', 'Political events, elections, and government activities'),
    ('economic_event', 'Economic events, market changes, and financial news'),
    ('social_event', 'Social movements, protests, and cultural events'),
    ('technological_event', 'Technology launches, innovations, and cybersecurity incidents'),
    ('health_event', 'Public health events, disease outbreaks, and medical breakthroughs'),
    ('conflict', 'Wars, conflicts, and military operations'),
    ('environmental_event', 'Environmental changes, conservation, and climate events')
ON CONFLICT (category_name) DO NOTHING;

-- Populate dim_event_type (common event types under natural_disaster category)
INSERT INTO dim_event_type (type_name, type_description, event_category_id)
SELECT
    type_name,
    type_description,
    ec.event_category_id
FROM (VALUES
    ('storm', 'Severe storms including hurricanes, typhoons, and cyclones', 'natural_disaster'),
    ('flood', 'Flooding events including flash floods and river floods', 'natural_disaster'),
    ('earthquake', 'Seismic events and earthquakes', 'natural_disaster'),
    ('wildfire', 'Forest fires and bushfires', 'natural_disaster'),
    ('drought', 'Prolonged periods of insufficient rainfall', 'natural_disaster'),
    ('heat_wave', 'Extended periods of extreme heat', 'natural_disaster'),
    ('landslide', 'Landslides and mudslides', 'natural_disaster'),
    ('tsunami', 'Ocean-based tidal waves', 'natural_disaster'),
    ('election', 'Political elections and voting events', 'political_event'),
    ('legislation', 'New laws and policy changes', 'political_event'),
    ('outbreak', 'Disease outbreaks and epidemics', 'health_event'),
    ('vaccination_campaign', 'Mass vaccination programs', 'health_event')
) AS types(type_name, type_description, category_name)
JOIN dim_event_category ec ON ec.category_name = types.category_name
ON CONFLICT (event_category_id, type_name) DO NOTHING;

-- Populate dim_event_subtype (specific storm types)
INSERT INTO dim_event_subtype (subtype_name, subtype_description, event_type_id)
SELECT
    subtype_name,
    subtype_description,
    et.event_type_id
FROM (VALUES
    ('tropical_cyclone', 'Tropical cyclone or typhoon', 'storm'),
    ('tornado', 'Rotating column of air', 'storm'),
    ('blizzard', 'Severe snowstorm', 'storm'),
    ('thunderstorm', 'Storm with thunder and lightning', 'storm'),
    ('hailstorm', 'Storm producing large hail', 'storm')
) AS subtypes(subtype_name, subtype_description, type_name)
JOIN dim_event_type et ON et.type_name = subtypes.type_name
ON CONFLICT (event_type_id, subtype_name) DO NOTHING;

\echo 'Event dimensions populated successfully!'

-- ==============================================================================
-- Entity Dimensions: Pre-populate Entity Taxonomies
-- ==============================================================================
\echo 'Populating entity dimension tables...'

-- Populate dim_entity_category
INSERT INTO dim_entity_category (category_name, category_description) VALUES
    ('person', 'Individual persons including politicians, celebrities, and public figures'),
    ('organization', 'Organizations including companies, NGOs, and government agencies'),
    ('location', 'Named locations beyond standard geographic hierarchy'),
    ('product', 'Products, brands, and commercial items'),
    ('event', 'Named events like conferences, festivals, and competitions')
ON CONFLICT (category_name) DO NOTHING;

-- Populate dim_entity_type
INSERT INTO dim_entity_type (type_name, type_description, entity_category_id)
SELECT
    type_name,
    type_description,
    ec.entity_category_id
FROM (VALUES
    ('politician', 'Government officials and political figures', 'person'),
    ('celebrity', 'Public figures from entertainment and sports', 'person'),
    ('scientist', 'Researchers and academics', 'person'),
    ('government_agency', 'Government departments and agencies', 'organization'),
    ('corporation', 'Private companies and businesses', 'organization'),
    ('ngo', 'Non-governmental organizations', 'organization'),
    ('university', 'Educational institutions', 'organization'),
    ('landmark', 'Notable landmarks and monuments', 'location'),
    ('building', 'Notable buildings and structures', 'location')
) AS types(type_name, type_description, category_name)
JOIN dim_entity_category ec ON ec.category_name = types.category_name
ON CONFLICT (entity_category_id, type_name) DO NOTHING;

\echo 'Entity dimensions populated successfully!'

-- ==============================================================================
-- Refresh Materialized Views
-- ==============================================================================
\echo 'Refreshing materialized views...'

REFRESH MATERIALIZED VIEW mv_temporal_hierarchy;
REFRESH MATERIALIZED VIEW mv_spatial_hierarchy;
REFRESH MATERIALIZED VIEW mv_event_hierarchy;
REFRESH MATERIALIZED VIEW mv_entity_hierarchy;

\echo 'Materialized views refreshed!'

-- ==============================================================================
-- Statistics and Verification
-- ==============================================================================
\echo ''
\echo 'Dimension population summary:'

SELECT
    'dim_year' AS table_name,
    COUNT(*) AS row_count
FROM dim_year

UNION ALL SELECT 'dim_quarter', COUNT(*) FROM dim_quarter
UNION ALL SELECT 'dim_month', COUNT(*) FROM dim_month
UNION ALL SELECT 'dim_date', COUNT(*) FROM dim_date
UNION ALL SELECT 'dim_time', COUNT(*) FROM dim_time
UNION ALL SELECT 'dim_continent', COUNT(*) FROM dim_continent
UNION ALL SELECT 'dim_country', COUNT(*) FROM dim_country
UNION ALL SELECT 'dim_state', COUNT(*) FROM dim_state
UNION ALL SELECT 'dim_event_category', COUNT(*) FROM dim_event_category
UNION ALL SELECT 'dim_event_type', COUNT(*) FROM dim_event_type
UNION ALL SELECT 'dim_event_subtype', COUNT(*) FROM dim_event_subtype
UNION ALL SELECT 'dim_entity_category', COUNT(*) FROM dim_entity_category
UNION ALL SELECT 'dim_entity_type', COUNT(*) FROM dim_entity_type;

\echo ''
\echo '================================'
\echo 'Dimension population completed successfully!'
\echo 'The warehouse is ready for ETL pipeline integration.'
\echo '================================'
