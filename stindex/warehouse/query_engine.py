"""
Query engine for file-based warehouse using DuckDB.

Provides SQL interface for querying JSON, Parquet, and GeoJSON files
without requiring a database server. DuckDB reads files directly.

Features:
- SQL queries on JSON Lines / Parquet / CSV files
- Spatial queries (distance, bounding box)
- Temporal filtering with date functions
- Aggregations and grouping
- Export to pandas DataFrames

Example:
    from stindex.warehouse.query_engine import STIndexQueryEngine

    engine = STIndexQueryEngine("data/warehouse")

    # SQL query
    df = engine.query("SELECT * FROM chunks WHERE temporal_year = 2022")

    # Fluent API
    results = (
        engine.select()
        .where_temporal(year=2022, quarter=1)
        .where_spatial(region="Australia")
        .limit(100)
        .execute()
    )

    # Spatial query
    nearby = engine.spatial_query(
        center=(122.2, -18.0),
        radius_km=100
    )
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from loguru import logger


class STIndexQueryEngine:
    """
    SQL query engine for file-based warehouse using DuckDB.

    Supports querying JSON Lines, Parquet, CSV, and GeoJSON files
    with full SQL capabilities including spatial functions.
    """

    def __init__(self, base_dir: str = "data/warehouse"):
        """
        Initialize query engine.

        Args:
            base_dir: Base directory of the file warehouse
        """
        self.base_dir = Path(base_dir)
        self._duckdb = None
        self._con = None

        # Check for DuckDB availability
        self._check_duckdb()

    def _check_duckdb(self) -> bool:
        """Check if DuckDB is available."""
        try:
            import duckdb
            self._duckdb = duckdb
            return True
        except ImportError:
            logger.warning(
                "DuckDB not installed. SQL queries will not be available. "
                "Install with: pip install duckdb"
            )
            return False

    def _get_connection(self):
        """Get or create DuckDB connection."""
        if self._duckdb is None:
            raise ImportError("DuckDB is required: pip install duckdb")

        if self._con is None:
            self._con = self._duckdb.connect(":memory:")
            # Install and load spatial extension for geo queries
            try:
                self._con.execute("INSTALL spatial; LOAD spatial;")
            except Exception:
                logger.debug("Spatial extension not available")

        return self._con

    # =========================================================================
    # RAW SQL QUERIES
    # =========================================================================

    def query(self, sql: str) -> "QueryResult":
        """
        Execute raw SQL query on warehouse files.

        The following tables/views are available:
        - chunks: All extraction chunks (from chunks.jsonl or chunks.parquet)
        - events: Spatial events (from events.geojson)

        Args:
            sql: SQL query string

        Returns:
            QueryResult with data and methods for conversion

        Example:
            result = engine.query('''
                SELECT temporal_year, COUNT(*) as count
                FROM chunks
                WHERE spatial_labels @> ['Australia']
                GROUP BY temporal_year
            ''')
            df = result.to_dataframe()
        """
        con = self._get_connection()

        # Register data sources as views
        self._register_sources(con)

        # Execute query
        result = con.execute(sql)

        return QueryResult(result, self._duckdb)

    def query_df(self, sql: str):
        """Execute SQL query and return pandas DataFrame directly."""
        return self.query(sql).to_dataframe()

    def _register_sources(self, con) -> None:
        """Register warehouse files as queryable tables."""
        # Prefer Parquet if available, fall back to JSON Lines
        parquet_path = self.base_dir / "chunks.parquet"
        jsonl_path = self.base_dir / "chunks.jsonl"

        if parquet_path.exists():
            con.execute(f"""
                CREATE OR REPLACE VIEW chunks AS
                SELECT * FROM read_parquet('{parquet_path}')
            """)
        elif jsonl_path.exists():
            con.execute(f"""
                CREATE OR REPLACE VIEW chunks AS
                SELECT * FROM read_json_auto('{jsonl_path}', format='newline_delimited')
            """)
        else:
            # Create empty view
            con.execute("""
                CREATE OR REPLACE VIEW chunks AS
                SELECT NULL as doc_id WHERE 1=0
            """)

        # Register GeoJSON for events
        geojson_path = self.base_dir / "events.geojson"
        if geojson_path.exists():
            # DuckDB can read GeoJSON with spatial extension
            try:
                con.execute(f"""
                    CREATE OR REPLACE VIEW events AS
                    SELECT
                        properties.id as event_id,
                        coalesce(properties.doc_id, properties.chunk_id) as doc_id,
                        properties.document_id as document_id,
                        properties.timestamp as timestamp,
                        properties.location as location,
                        ST_X(geom) as longitude,
                        ST_Y(geom) as latitude,
                        properties
                    FROM ST_Read('{geojson_path}')
                """)
            except Exception:
                # Fall back to JSON parsing
                con.execute(f"""
                    CREATE OR REPLACE VIEW events AS
                    SELECT
                        f.properties.id as event_id,
                        coalesce(f.properties.doc_id, f.properties.chunk_id) as doc_id,
                        f.properties.document_id as document_id,
                        f.properties.timestamp as timestamp,
                        f.properties.location as location,
                        f.geometry.coordinates[1] as longitude,
                        f.geometry.coordinates[2] as latitude
                    FROM (
                        SELECT unnest(features) as f
                        FROM read_json_auto('{geojson_path}')
                    )
                """)

    # =========================================================================
    # FLUENT QUERY BUILDER
    # =========================================================================

    def select(self, *columns: str) -> "QueryBuilder":
        """
        Start building a query with SELECT.

        Args:
            columns: Column names to select (default: all)

        Returns:
            QueryBuilder for method chaining

        Example:
            results = (
                engine.select("doc_id", "temporal_normalized", "spatial_text")
                .where_temporal(year=2022)
                .where_spatial(region="Australia")
                .limit(10)
                .execute()
            )
        """
        return QueryBuilder(self, list(columns) if columns else ["*"])

    # =========================================================================
    # CONVENIENCE METHODS
    # =========================================================================

    def get_temporal_distribution(self) -> Dict[int, int]:
        """Get distribution of chunks by year."""
        result = self.query("""
            SELECT temporal_year, COUNT(*) as count
            FROM chunks
            WHERE temporal_year IS NOT NULL
            GROUP BY temporal_year
            ORDER BY temporal_year
        """)
        df = result.to_dataframe()
        return dict(zip(df["temporal_year"], df["count"]))

    def get_spatial_distribution(self, level: str = "country") -> Dict[str, int]:
        """
        Get distribution of chunks by spatial region.

        Args:
            level: Hierarchy level (uses first N labels)

        Returns:
            Dict of region -> count
        """
        result = self.query("""
            SELECT spatial_labels[1] as region, COUNT(*) as count
            FROM chunks
            WHERE spatial_labels IS NOT NULL AND len(spatial_labels) > 0
            GROUP BY spatial_labels[1]
            ORDER BY count DESC
            LIMIT 50
        """)
        df = result.to_dataframe()
        return dict(zip(df["region"], df["count"]))

    def spatial_query(
        self,
        center: Tuple[float, float],
        radius_km: float,
    ) -> List[Dict[str, Any]]:
        """
        Find chunks within radius of a point.

        Args:
            center: (longitude, latitude) of center point
            radius_km: Radius in kilometers

        Returns:
            List of matching chunks with distance
        """
        lon, lat = center

        # Haversine distance in SQL
        result = self.query(f"""
            SELECT
                *,
                6371 * 2 * ASIN(SQRT(
                    POWER(SIN(RADIANS(latitude - {lat}) / 2), 2) +
                    COS(RADIANS({lat})) * COS(RADIANS(latitude)) *
                    POWER(SIN(RADIANS(longitude - {lon}) / 2), 2)
                )) as distance_km
            FROM chunks
            WHERE latitude IS NOT NULL AND longitude IS NOT NULL
            HAVING distance_km <= {radius_km}
            ORDER BY distance_km
        """)

        return result.to_dicts()

    def temporal_query(
        self,
        year: Optional[int] = None,
        quarter: Optional[int] = None,
        month: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Find chunks within temporal range.

        Args:
            year: Filter by year
            quarter: Filter by quarter (1-4)
            month: Filter by month (1-12)
            start_date: Start date (ISO format)
            end_date: End date (ISO format)

        Returns:
            List of matching chunks
        """
        conditions = []

        if year:
            conditions.append(f"temporal_year = {year}")
        if quarter:
            conditions.append(f"temporal_quarter = {quarter}")
        if month:
            conditions.append(f"temporal_month = {month}")
        if start_date:
            conditions.append(f"temporal_normalized >= '{start_date}'")
        if end_date:
            conditions.append(f"temporal_normalized <= '{end_date}'")

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        result = self.query(f"""
            SELECT * FROM chunks
            WHERE {where_clause}
            ORDER BY temporal_normalized
        """)

        return result.to_dicts()

    def dimension_query(self, **dimension_filters) -> List[Dict[str, Any]]:
        """
        Query by custom dimensions.

        Args:
            **dimension_filters: Dimension name -> value pairs

        Returns:
            List of matching chunks

        Example:
            results = engine.dimension_query(
                event_type="exposure_site",
                severity="high"
            )
        """
        conditions = []

        for dim_name, dim_value in dimension_filters.items():
            # Check if it's a standard column or in dimensions JSON
            if dim_name in ("temporal_year", "temporal_quarter", "spatial_text"):
                conditions.append(f"{dim_name} = '{dim_value}'")
            else:
                # Query inside dimensions JSON
                conditions.append(f"dimensions->>'{dim_name}' IS NOT NULL")

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        result = self.query(f"""
            SELECT * FROM chunks
            WHERE {where_clause}
        """)

        return result.to_dicts()

    def aggregate(
        self,
        group_by: Union[str, List[str]],
        agg_func: str = "COUNT",
        agg_column: str = "*",
    ) -> Dict[str, int]:
        """
        Aggregate chunks by dimension.

        Args:
            group_by: Column(s) to group by
            agg_func: Aggregation function (COUNT, SUM, AVG, etc.)
            agg_column: Column to aggregate

        Returns:
            Dict of group -> aggregated value
        """
        if isinstance(group_by, list):
            group_cols = ", ".join(group_by)
        else:
            group_cols = group_by

        result = self.query(f"""
            SELECT {group_cols}, {agg_func}({agg_column}) as value
            FROM chunks
            GROUP BY {group_cols}
            ORDER BY value DESC
        """)

        df = result.to_dataframe()

        if isinstance(group_by, list):
            # Multi-column grouping
            return df.to_dict(orient="records")
        else:
            return dict(zip(df[group_by], df["value"]))


class QueryBuilder:
    """Fluent query builder for warehouse queries."""

    def __init__(self, engine: STIndexQueryEngine, columns: List[str]):
        self.engine = engine
        self.columns = columns
        self.conditions = []
        self.order_by_clause = None
        self.limit_value = None
        self.offset_value = None

    def where(self, condition: str) -> "QueryBuilder":
        """Add a raw WHERE condition."""
        self.conditions.append(condition)
        return self

    def where_temporal(
        self,
        year: Optional[int] = None,
        quarter: Optional[int] = None,
        month: Optional[int] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> "QueryBuilder":
        """Add temporal filter conditions."""
        if year:
            self.conditions.append(f"temporal_year = {year}")
        if quarter:
            self.conditions.append(f"temporal_quarter = {quarter}")
        if month:
            self.conditions.append(f"temporal_month = {month}")
        if start_date:
            self.conditions.append(f"temporal_normalized >= '{start_date}'")
        if end_date:
            self.conditions.append(f"temporal_normalized <= '{end_date}'")
        return self

    def where_spatial(
        self,
        region: Optional[str] = None,
        country: Optional[str] = None,
        has_coordinates: bool = False,
    ) -> "QueryBuilder":
        """Add spatial filter conditions."""
        if region:
            self.conditions.append(f"list_contains(spatial_labels, '{region}')")
        if country:
            self.conditions.append(f"list_contains(spatial_labels, '{country}')")
        if has_coordinates:
            self.conditions.append("latitude IS NOT NULL AND longitude IS NOT NULL")
        return self

    def where_dimension(self, dimension: str, value: str) -> "QueryBuilder":
        """Add custom dimension filter."""
        self.conditions.append(f"dimensions->>'{dimension}' = '{value}'")
        return self

    def order_by(self, column: str, desc: bool = False) -> "QueryBuilder":
        """Add ORDER BY clause."""
        direction = "DESC" if desc else "ASC"
        self.order_by_clause = f"{column} {direction}"
        return self

    def limit(self, n: int) -> "QueryBuilder":
        """Add LIMIT clause."""
        self.limit_value = n
        return self

    def offset(self, n: int) -> "QueryBuilder":
        """Add OFFSET clause."""
        self.offset_value = n
        return self

    def build_sql(self) -> str:
        """Build the SQL query string."""
        columns_str = ", ".join(self.columns)
        sql = f"SELECT {columns_str} FROM chunks"

        if self.conditions:
            sql += " WHERE " + " AND ".join(self.conditions)

        if self.order_by_clause:
            sql += f" ORDER BY {self.order_by_clause}"

        if self.limit_value:
            sql += f" LIMIT {self.limit_value}"

        if self.offset_value:
            sql += f" OFFSET {self.offset_value}"

        return sql

    def execute(self) -> "QueryResult":
        """Execute the built query."""
        sql = self.build_sql()
        return self.engine.query(sql)

    def to_dataframe(self):
        """Execute and return as DataFrame."""
        return self.execute().to_dataframe()

    def to_dicts(self) -> List[Dict[str, Any]]:
        """Execute and return as list of dicts."""
        return self.execute().to_dicts()


class QueryResult:
    """Wrapper for DuckDB query results with conversion methods."""

    def __init__(self, result, duckdb_module):
        self._result = result
        self._duckdb = duckdb_module

    def to_dataframe(self):
        """Convert result to pandas DataFrame."""
        return self._result.fetchdf()

    def to_dicts(self) -> List[Dict[str, Any]]:
        """Convert result to list of dictionaries."""
        df = self.to_dataframe()
        return df.to_dict(orient="records")

    def to_list(self) -> List[Tuple]:
        """Convert result to list of tuples."""
        return self._result.fetchall()

    def fetchone(self) -> Optional[Tuple]:
        """Fetch single row."""
        return self._result.fetchone()

    def fetchall(self) -> List[Tuple]:
        """Fetch all rows."""
        return self._result.fetchall()

    @property
    def columns(self) -> List[str]:
        """Get column names."""
        return [desc[0] for desc in self._result.description]


# ============================================================================
# PANDAS-BASED FALLBACK (when DuckDB not available)
# ============================================================================

class PandasQueryEngine:
    """
    Fallback query engine using pandas when DuckDB is not available.

    Provides basic filtering capabilities without SQL.
    """

    def __init__(self, base_dir: str = "data/warehouse"):
        self.base_dir = Path(base_dir)
        self._df = None

    def _load_data(self):
        """Load data into pandas DataFrame."""
        if self._df is not None:
            return self._df

        try:
            import pandas as pd
        except ImportError:
            raise ImportError("pandas is required: pip install pandas")

        parquet_path = self.base_dir / "chunks.parquet"
        jsonl_path = self.base_dir / "chunks.jsonl"

        if parquet_path.exists():
            self._df = pd.read_parquet(parquet_path)
        elif jsonl_path.exists():
            self._df = pd.read_json(jsonl_path, lines=True)
        else:
            self._df = pd.DataFrame()

        return self._df

    def filter_temporal(
        self,
        year: Optional[int] = None,
        quarter: Optional[int] = None,
        month: Optional[int] = None,
    ):
        """Filter by temporal dimensions."""
        df = self._load_data()

        if year:
            df = df[df["temporal_year"] == year]
        if quarter:
            df = df[df["temporal_quarter"] == quarter]
        if month:
            df = df[df["temporal_month"] == month]

        return df

    def filter_spatial(self, region: str):
        """Filter by spatial region."""
        df = self._load_data()
        return df[df["spatial_labels"].apply(lambda x: region in x if x else False)]

    def get_all(self):
        """Get all data as DataFrame."""
        return self._load_data()
