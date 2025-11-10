"""
Data Warehouse ETL Pipeline for STIndex.

Handles loading of multi-dimensional extraction results into the dimensional warehouse.
Implements upsert logic for dimension tables and batch insertion for fact table.
"""

import hashlib
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import psycopg2
from loguru import logger
from psycopg2.extras import RealDictCursor, execute_batch

from stindex.warehouse.chunk_labeler import ChunkDimensionalLabels, DimensionalChunkLabeler
from stindex.llm.response.dimension_models import MultiDimensionalResult


class DimensionalWarehouseETL:
    """
    ETL pipeline for loading extraction results into dimensional warehouse.

    Handles:
    - Dimension table upserts (with caching for performance)
    - Fact table insertions (with batch processing)
    - Transaction management
    - Error handling and rollback
    """

    def __init__(
        self,
        db_connection_string: str,
        chunk_labeler: Optional[DimensionalChunkLabeler] = None,
        batch_size: int = 100,
    ):
        """
        Initialize ETL pipeline.

        Args:
            db_connection_string: PostgreSQL connection string
                (e.g., "postgresql://user:password@localhost:5432/stindex_warehouse")
            chunk_labeler: Optional chunk labeler (creates if not provided)
            batch_size: Batch size for fact table insertions
        """
        self.db_connection_string = db_connection_string
        self.conn = None
        self.cursor = None

        self.chunk_labeler = chunk_labeler or DimensionalChunkLabeler()
        self.batch_size = batch_size

        # Dimension caches (to avoid repeated lookups)
        self._temporal_cache: Dict[str, int] = {}
        self._spatial_cache: Dict[str, int] = {}
        self._event_cache: Dict[str, int] = {}
        self._entity_cache: Dict[str, int] = {}
        self._document_cache: Dict[str, int] = {}

        # Date hierarchy caches
        self._date_cache: Dict[str, int] = {}
        self._month_cache: Dict[Tuple[int, int], int] = {}  # (quarter_id, month)
        self._quarter_cache: Dict[Tuple[int, int], int] = {}  # (year_id, quarter)
        self._year_cache: Dict[int, int] = {}  # year value -> year_id

        logger.info(f"ETL pipeline initialized with batch_size={batch_size}")

    def connect(self) -> None:
        """Establish database connection."""
        if self.conn is None or self.conn.closed:
            logger.info("Connecting to warehouse database...")
            self.conn = psycopg2.connect(self.db_connection_string)
            self.cursor = self.conn.cursor(cursor_factory=RealDictCursor)
            logger.success("Database connection established")

    def disconnect(self) -> None:
        """Close database connection."""
        if self.cursor:
            self.cursor.close()
        if self.conn:
            self.conn.close()
        logger.info("Database connection closed")

    def load_extraction_results(
        self,
        extraction_results: List[MultiDimensionalResult],
        document_metadata: Dict[str, Any],
        document_text: str,
        embedding_model: Optional[str] = None,
        embeddings: Optional[List[List[float]]] = None,
    ) -> int:
        """
        Load extraction results into warehouse.

        Args:
            extraction_results: List of extraction results (one per chunk)
            document_metadata: Document metadata (title, URL, publication date, etc.)
            document_text: Full document text
            embedding_model: Name of embedding model used (e.g., 'text-embedding-3-small')
            embeddings: Optional chunk embeddings (same length as extraction_results)

        Returns:
            Number of chunks loaded

        Raises:
            Exception: If loading fails (transaction is rolled back)
        """
        try:
            self.connect()
            self.conn.autocommit = False  # Use transactions

            # 1. Upsert document dimension
            logger.info("Upserting document dimension...")
            document_id = self._upsert_document_dimension(document_metadata, document_text)

            # 2. Process each chunk
            logger.info(f"Loading {len(extraction_results)} chunks...")
            chunks_loaded = 0

            for chunk_idx, extraction_result in enumerate(extraction_results):
                # Get chunk text from extraction result
                chunk_text = extraction_result.input_text

                # Generate dimensional labels
                labels = self.chunk_labeler.label_chunk(
                    chunk_text=chunk_text,
                    extraction_result=extraction_result,
                    chunk_index=chunk_idx,
                    document_id=document_id,
                )

                # Get embedding for this chunk (if provided)
                chunk_embedding = embeddings[chunk_idx] if embeddings else None

                # Upsert dimension tables and get dimension IDs
                temporal_dim_id = self._upsert_temporal_dimensions(extraction_result, labels)
                spatial_dim_id = self._upsert_spatial_dimensions(extraction_result, labels)
                event_dim_id = self._upsert_event_dimensions(extraction_result, labels)
                entity_dim_id = self._upsert_entity_dimensions(extraction_result, labels)

                # Insert fact record
                self._insert_fact_chunk(
                    labels=labels,
                    document_id=document_id,
                    temporal_dim_id=temporal_dim_id,
                    spatial_dim_id=spatial_dim_id,
                    event_dim_id=event_dim_id,
                    entity_dim_id=entity_dim_id,
                    embedding=chunk_embedding,
                    embedding_model=embedding_model,
                    extraction_result=extraction_result,
                )

                chunks_loaded += 1

                if (chunk_idx + 1) % self.batch_size == 0:
                    logger.debug(f"Processed {chunk_idx + 1}/{len(extraction_results)} chunks")

            # Commit transaction
            self.conn.commit()
            logger.success(f"✓ Loaded {chunks_loaded} chunks for document {document_id}")

            return chunks_loaded

        except Exception as e:
            logger.error(f"ETL failed: {e}")
            if self.conn:
                self.conn.rollback()
            raise

        finally:
            self.disconnect()

    def _upsert_document_dimension(
        self,
        document_metadata: Dict[str, Any],
        document_text: str,
    ) -> int:
        """
        Upsert document dimension table.

        Args:
            document_metadata: Document metadata
            document_text: Full document text

        Returns:
            document_id
        """
        # Compute document hash
        document_hash = hashlib.sha256(document_text.encode('utf-8')).hexdigest()

        # Check cache
        if document_hash in self._document_cache:
            return self._document_cache[document_hash]

        # Extract metadata fields
        document_url = document_metadata.get("url")
        document_path = document_metadata.get("file_path")
        document_title = document_metadata.get("title", "")
        document_type = document_metadata.get("document_type", "unknown")
        document_language = document_metadata.get("language", "en")

        publication_date = document_metadata.get("publication_date")
        publication_source = document_metadata.get("source", "")
        author = document_metadata.get("author")
        publisher = document_metadata.get("publisher")

        word_count = len(document_text.split())
        char_count = len(document_text)
        total_chunks = document_metadata.get("total_chunks", 0)

        # Prepare source metadata JSON
        source_metadata = json.dumps(document_metadata)

        # Upsert query
        query = """
            INSERT INTO dim_document (
                document_hash, document_url, document_path, document_title, document_type,
                document_language, publication_date, publication_source, author, publisher,
                word_count, char_count, total_chunks, source_metadata
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (document_hash) DO UPDATE SET
                document_url = EXCLUDED.document_url,
                document_path = EXCLUDED.document_path,
                document_title = EXCLUDED.document_title,
                updated_at = CURRENT_TIMESTAMP
            RETURNING document_id;
        """

        self.cursor.execute(
            query,
            (
                document_hash, document_url, document_path, document_title, document_type,
                document_language, publication_date, publication_source, author, publisher,
                word_count, char_count, total_chunks, source_metadata
            ),
        )

        result = self.cursor.fetchone()
        document_id = result["document_id"]

        # Cache result
        self._document_cache[document_hash] = document_id

        logger.debug(f"Document dimension upserted: document_id={document_id}")

        return document_id

    def _upsert_temporal_dimensions(
        self,
        extraction_result: MultiDimensionalResult,
        labels: ChunkDimensionalLabels,
    ) -> Optional[int]:
        """
        Upsert temporal dimension tables.

        Inserts/updates temporal hierarchy (year → quarter → month → date → temporal)

        Returns:
            temporal_id (or None if no temporal entity)
        """
        temporal_entities = extraction_result.temporal_entities

        if not temporal_entities:
            return None

        # Get first temporal entity
        entity = temporal_entities[0]

        original_text = entity.get("text", "")
        normalized_value = entity.get("normalized", "")
        temporal_type = entity.get("normalization_type", "date")
        confidence = entity.get("confidence", 1.0)

        # Check cache
        cache_key = f"{normalized_value}:{temporal_type}"
        if cache_key in self._temporal_cache:
            return self._temporal_cache[cache_key]

        # Parse date to get hierarchy IDs
        date_id = None
        if temporal_type in ("date", "datetime") and normalized_value:
            try:
                # Parse date
                if "T" in normalized_value:
                    dt = datetime.fromisoformat(normalized_value)
                else:
                    dt = datetime.fromisoformat(normalized_value + "T00:00:00")

                # Get date_id from pre-populated dim_date table
                date_str = dt.strftime("%Y-%m-%d")
                date_id = self._get_date_id(date_str)

            except (ValueError, IndexError) as e:
                logger.warning(f"Failed to parse temporal value '{normalized_value}': {e}")

        # Upsert dim_temporal
        query = """
            INSERT INTO dim_temporal (
                original_text, normalized_value, temporal_type, date_id,
                confidence, extraction_method, granularity
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (normalized_value, temporal_type) DO UPDATE SET
                confidence = GREATEST(dim_temporal.confidence, EXCLUDED.confidence),
                updated_at = CURRENT_TIMESTAMP
            RETURNING temporal_id;
        """

        self.cursor.execute(
            query,
            (
                original_text, normalized_value, temporal_type, date_id,
                confidence, "llm", temporal_type
            ),
        )

        result = self.cursor.fetchone()
        temporal_id = result["temporal_id"]

        # Cache result
        self._temporal_cache[cache_key] = temporal_id

        logger.debug(f"Temporal dimension upserted: temporal_id={temporal_id}, value={normalized_value}")

        return temporal_id

    def _get_date_id(self, date_str: str) -> Optional[int]:
        """Get date_id from pre-populated dim_date table (with caching)."""
        if date_str in self._date_cache:
            return self._date_cache[date_str]

        query = "SELECT date_id FROM dim_date WHERE full_date = %s"
        self.cursor.execute(query, (date_str,))
        result = self.cursor.fetchone()

        if result:
            date_id = result["date_id"]
            self._date_cache[date_str] = date_id
            return date_id

        return None

    def _upsert_spatial_dimensions(
        self,
        extraction_result: MultiDimensionalResult,
        labels: ChunkDimensionalLabels,
    ) -> Optional[int]:
        """
        Upsert spatial dimension tables.

        For now, just inserts into dim_spatial with coordinates.
        In Phase 3.2, we'll implement full hierarchy upserts.

        Returns:
            spatial_id (or None if no spatial entity)
        """
        spatial_entities = extraction_result.spatial_entities

        if not spatial_entities:
            return None

        # Get first spatial entity
        entity = spatial_entities[0]

        original_text = entity.get("text", "")
        latitude = entity.get("latitude")
        longitude = entity.get("longitude")
        location_type = entity.get("location_type", "city")
        confidence = entity.get("confidence", 1.0)

        if latitude is None or longitude is None:
            return None

        # Check cache
        cache_key = f"{original_text}:{latitude}:{longitude}"
        if cache_key in self._spatial_cache:
            return self._spatial_cache[cache_key]

        # Upsert dim_spatial
        query = """
            INSERT INTO dim_spatial (
                original_text, location_type, latitude, longitude,
                geom, confidence, extraction_method, geocoding_provider
            )
            VALUES (%s, %s, %s, %s, ST_MakePoint(%s, %s)::geography, %s, %s, %s)
            ON CONFLICT (original_text, latitude, longitude) DO UPDATE SET
                confidence = GREATEST(dim_spatial.confidence, EXCLUDED.confidence),
                updated_at = CURRENT_TIMESTAMP
            RETURNING spatial_id;
        """

        self.cursor.execute(
            query,
            (
                original_text, location_type, latitude, longitude,
                longitude, latitude,  # Note: PostGIS uses (lon, lat) order
                confidence, "llm+geocoding", "nominatim"
            ),
        )

        result = self.cursor.fetchone()
        spatial_id = result["spatial_id"]

        # Cache result
        self._spatial_cache[cache_key] = spatial_id

        logger.debug(f"Spatial dimension upserted: spatial_id={spatial_id}, location={original_text}")

        return spatial_id

    def _upsert_event_dimensions(
        self,
        extraction_result: MultiDimensionalResult,
        labels: ChunkDimensionalLabels,
    ) -> Optional[int]:
        """
        Upsert event dimension tables.

        Placeholder for Phase 3.2.

        Returns:
            event_id (or None if no event entity)
        """
        # TODO Phase 3.2: Implement event hierarchy upserts
        return None

    def _upsert_entity_dimensions(
        self,
        extraction_result: MultiDimensionalResult,
        labels: ChunkDimensionalLabels,
    ) -> Optional[int]:
        """
        Upsert entity dimension tables.

        Placeholder for Phase 3.2.

        Returns:
            entity_id (or None if no entity)
        """
        # TODO Phase 3.2: Implement entity hierarchy upserts
        return None

    def _insert_fact_chunk(
        self,
        labels: ChunkDimensionalLabels,
        document_id: int,
        temporal_dim_id: Optional[int],
        spatial_dim_id: Optional[int],
        event_dim_id: Optional[int],
        entity_dim_id: Optional[int],
        embedding: Optional[List[float]],
        embedding_model: Optional[str],
        extraction_result: MultiDimensionalResult,
    ) -> None:
        """
        Insert fact record for chunk.

        Args:
            labels: Dimensional labels
            document_id: Document dimension ID
            temporal_dim_id: Temporal dimension ID
            spatial_dim_id: Spatial dimension ID
            event_dim_id: Event dimension ID
            entity_dim_id: Entity dimension ID
            embedding: Vector embedding
            embedding_model: Embedding model name
            extraction_result: Original extraction result
        """
        # Prepare data
        chunk_text = labels.chunk_text
        chunk_hash = labels.chunk_hash
        chunk_index = labels.chunk_index

        chunk_size_chars = len(chunk_text)
        chunk_size_words = len(chunk_text.split())

        # Vector embedding (convert to PostgreSQL array format)
        chunk_vector = f"[{','.join(map(str, embedding))}]" if embedding else None

        # Geographic coordinates
        latitude = labels.latitude
        longitude = labels.longitude
        location_geom = None
        if latitude is not None and longitude is not None:
            # Will be computed by PostGIS in the query

            pass

        # Label arrays
        temporal_labels = labels.temporal_labels or None
        spatial_labels = labels.spatial_labels or None
        event_labels = labels.event_labels or None
        entity_labels = labels.entity_labels or None

        # Hierarchy paths
        temporal_path = labels.temporal_path
        spatial_path = labels.spatial_path
        event_path = labels.event_path
        entity_path = labels.entity_path

        # Confidence scores
        confidence_score = labels.confidence_score
        dimension_confidences = json.dumps(labels.dimension_confidences) if labels.dimension_confidences else None
        entity_counts = json.dumps(labels.entity_counts) if labels.entity_counts else None

        # LLM metadata (from extraction config)
        extraction_config = extraction_result.extraction_config or {}
        llm_provider = extraction_config.get("llm_provider")
        llm_model = extraction_config.get("model_name")
        llm_token_usage = None
        if extraction_config.get("token_usage"):
            llm_token_usage = json.dumps(extraction_config["token_usage"])

        # Insert fact record
        query = """
            INSERT INTO fact_document_chunks (
                document_id, chunk_index, chunk_hash,
                temporal_dim_id, spatial_dim_id, event_dim_id, entity_dim_id,
                chunk_text, chunk_size_chars, chunk_size_words,
                chunk_vector, embedding_model,
                latitude, longitude, location_geom,
                temporal_labels, spatial_labels, event_labels, entity_labels,
                temporal_path, spatial_path, event_path, entity_path,
                confidence_score, dimension_confidences, entity_counts,
                llm_provider, llm_model, llm_token_usage
            )
            VALUES (
                %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s,
                %s, %s,
                %s, %s,
                CASE WHEN %s IS NOT NULL AND %s IS NOT NULL
                    THEN ST_MakePoint(%s, %s)::geography
                    ELSE NULL
                END,
                %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s
            )
            ON CONFLICT (document_id, chunk_index) DO UPDATE SET
                confidence_score = EXCLUDED.confidence_score,
                updated_at = CURRENT_TIMESTAMP;
        """

        self.cursor.execute(
            query,
            (
                document_id, chunk_index, chunk_hash,
                temporal_dim_id, spatial_dim_id, event_dim_id, entity_dim_id,
                chunk_text, chunk_size_chars, chunk_size_words,
                chunk_vector, embedding_model,
                latitude, longitude,
                latitude, longitude, longitude, latitude,  # For ST_MakePoint (lon, lat order)
                temporal_labels, spatial_labels, event_labels, entity_labels,
                temporal_path, spatial_path, event_path, entity_path,
                confidence_score, dimension_confidences, entity_counts,
                llm_provider, llm_model, llm_token_usage
            ),
        )

        logger.debug(f"Fact chunk inserted: document_id={document_id}, chunk_index={chunk_index}")
