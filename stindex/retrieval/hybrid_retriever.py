"""
Hybrid Retriever for STIndex.

Combines vector similarity search (FAISS) with dimensional filtering
(temporal, spatial) for precise retrieval in RAG systems.

Features:
- Pure vector search
- Pre-filtering: Filter first, then search filtered set
- Post-filtering: Search first, then filter results
- Re-ranking: Combine vector scores with dimensional relevance
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from loguru import logger

from .dimensional_filter import DimensionalFilter, DimensionFilter, SpatialFilter, TemporalFilter


@dataclass
class RetrievalResult:
    """Single retrieval result."""
    doc_id: str
    text: str
    score: float
    vector_score: Optional[float] = None
    dimensional_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "text": self.text,
            "score": self.score,
            "vector_score": self.vector_score,
            "dimensional_score": self.dimensional_score,
            **self.metadata,
        }


@dataclass
class RetrievalResponse:
    """Response from hybrid retrieval."""
    results: List[RetrievalResult]
    query: str
    total_candidates: int
    filtered_candidates: Optional[int] = None
    filter_stats: Optional[Dict[str, Any]] = None

    def get_texts(self) -> List[str]:
        """Get list of retrieved texts."""
        return [r.text for r in self.results]

    def get_context(self, separator: str = "\n\n") -> str:
        """Get concatenated context for LLM."""
        return separator.join(self.get_texts())


class HybridRetriever:
    """
    Hybrid retriever combining FAISS vector search with STIndex dimensional filtering.

    Supports three retrieval modes:
    1. Vector-only: Pure semantic similarity search
    2. Pre-filter: Apply dimensional filters first, then vector search on filtered set
    3. Post-filter: Vector search first, then filter results

    Usage:
        retriever = HybridRetriever(
            vector_index_path="data/vector/rag",
            stindex_warehouse_path="data/warehouse/rag",
        )

        # Pure vector search
        results = retriever.search("What happened in Paris?", k=10)

        # Filtered search
        results = retriever.search(
            "What events occurred?",
            k=10,
            temporal_filter={"year": 2022},
            spatial_filter={"region": "France"},
        )
    """

    def __init__(
        self,
        vector_index_path: str,
        stindex_warehouse_path: Optional[str] = None,
        encoder_model: Optional[str] = None,
        device: str = "cpu",
    ):
        """
        Initialize hybrid retriever.

        Args:
            vector_index_path: Path to FAISS vector index directory
            stindex_warehouse_path: Path to STIndex warehouse (for dimensional filtering)
            encoder_model: Override encoder model (default: from index config)
            device: Device for encoding ('cpu' or 'cuda')
        """
        self.vector_index_path = Path(vector_index_path)
        self.device = device

        # Load vector index
        self._load_vector_index(encoder_model)

        # Initialize dimensional filter if warehouse path provided
        self.dimensional_filter = None
        if stindex_warehouse_path:
            self.stindex_warehouse_path = Path(stindex_warehouse_path)
            self.dimensional_filter = DimensionalFilter(stindex_warehouse_path)

        # Load chunk texts for returning full results
        self._chunk_texts = None

        logger.info(f"HybridRetriever initialized: "
                   f"vector_index={self.vector_index_path}, "
                   f"dimensional_filter={'enabled' if self.dimensional_filter else 'disabled'}")

    def _load_vector_index(self, encoder_model: Optional[str] = None) -> None:
        """Load FAISS index and related components."""
        try:
            import faiss
        except ImportError:
            raise ImportError("faiss-cpu is required: pip install faiss-cpu")

        # Load config
        config_path = self.vector_index_path / "index_config.json"
        with open(config_path, "r") as f:
            self.index_config = json.load(f)

        # Load FAISS index
        index_path = self.vector_index_path / "faiss_index.bin"
        self.faiss_index = faiss.read_index(str(index_path))
        logger.info(f"Loaded FAISS index with {self.faiss_index.ntotal} vectors")

        # Load ID mapping
        id_mapping_path = self.vector_index_path / "id_mapping.json"
        with open(id_mapping_path, "r") as f:
            self.id_mapping = json.load(f)

        # Create reverse mapping (doc_id -> faiss_idx)
        self.reverse_id_mapping = {
            doc_id: idx for idx, doc_id in enumerate(self.id_mapping)
        }

        # Initialize encoder
        model_name = encoder_model or self.index_config.get("model_name", "BAAI/bge-m3")
        self._init_encoder(model_name)

    def _init_encoder(self, model_name: str) -> None:
        """Initialize sentence transformer encoder."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required: pip install sentence-transformers"
            )

        logger.info(f"Loading encoder: {model_name}")
        self.encoder = SentenceTransformer(model_name, device=self.device)

    def _load_chunk_texts(self) -> Dict[str, str]:
        """Load chunk texts from metadata file."""
        if self._chunk_texts is not None:
            return self._chunk_texts

        self._chunk_texts = {}

        # Try metadata file in vector index
        metadata_path = self.vector_index_path / "chunks_metadata.jsonl"
        if metadata_path.exists():
            import jsonlines
            with jsonlines.open(metadata_path, "r") as reader:
                for chunk in reader:
                    doc_id = chunk.get("doc_id", "")
                    text = chunk.get("text", "")
                    self._chunk_texts[doc_id] = text

        # Fall back to enriched chunks in warehouse
        if not self._chunk_texts and self.stindex_warehouse_path:
            enriched_path = self.stindex_warehouse_path / "chunks_enriched.jsonl"
            if enriched_path.exists():
                import jsonlines
                with jsonlines.open(enriched_path, "r") as reader:
                    for chunk in reader:
                        doc_id = chunk.get("doc_id", "")
                        text = chunk.get("text", "")
                        self._chunk_texts[doc_id] = text

        return self._chunk_texts

    def encode_query(self, query: str) -> np.ndarray:
        """Encode query text to embedding."""
        embedding = self.encoder.encode(
            query,
            normalize_embeddings=self.index_config.get("normalize_embeddings", True),
        )
        return embedding.astype(np.float32).reshape(1, -1)

    def search(
        self,
        query: str,
        k: int = 10,
        temporal_filter: Optional[Union[TemporalFilter, Dict[str, Any]]] = None,
        spatial_filter: Optional[Union[SpatialFilter, Dict[str, Any]]] = None,
        dimension_filters: Optional[List[Union[DimensionFilter, Dict[str, Any]]]] = None,
        filter_mode: str = "post",
        expand_factor: int = 3,
    ) -> RetrievalResponse:
        """
        Search for relevant chunks.

        Args:
            query: Query text
            k: Number of results to return
            temporal_filter: Temporal filter criteria
            spatial_filter: Spatial filter criteria
            dimension_filters: List of generic dimension filters (e.g., drug, procedure)
            filter_mode: 'pre' (filter first) or 'post' (search first then filter)
            expand_factor: For post-filtering, retrieve k*expand_factor candidates

        Returns:
            RetrievalResponse with results
        """
        # Check if filtering is needed
        has_filters = (
            temporal_filter is not None or
            spatial_filter is not None or
            dimension_filters is not None
        )

        if not has_filters:
            # Pure vector search
            return self._vector_search(query, k)

        if self.dimensional_filter is None:
            logger.warning("Dimensional filter not available, falling back to vector search")
            return self._vector_search(query, k)

        if filter_mode == "pre":
            return self._pre_filter_search(query, k, temporal_filter, spatial_filter, dimension_filters)
        else:
            return self._post_filter_search(query, k, temporal_filter, spatial_filter, dimension_filters, expand_factor)

    def _vector_search(self, query: str, k: int) -> RetrievalResponse:
        """Pure vector similarity search."""
        query_embedding = self.encode_query(query)

        # Search FAISS
        scores, indices = self.faiss_index.search(query_embedding, k)

        # Build results
        results = []
        chunk_texts = self._load_chunk_texts()

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue

            doc_id = self.id_mapping[idx]
            text = chunk_texts.get(doc_id, "")

            results.append(RetrievalResult(
                doc_id=doc_id,
                text=text,
                score=float(score),
                vector_score=float(score),
            ))

        return RetrievalResponse(
            results=results,
            query=query,
            total_candidates=self.faiss_index.ntotal,
        )

    def _pre_filter_search(
        self,
        query: str,
        k: int,
        temporal_filter: Optional[Union[TemporalFilter, Dict[str, Any]]],
        spatial_filter: Optional[Union[SpatialFilter, Dict[str, Any]]],
        dimension_filters: Optional[List[Union[DimensionFilter, Dict[str, Any]]]] = None,
    ) -> RetrievalResponse:
        """Filter first, then search within filtered set."""
        # Apply dimensional filters
        filter_result = self.dimensional_filter.filter(
            temporal_filter=temporal_filter,
            spatial_filter=spatial_filter,
            dimension_filters=dimension_filters,
        )

        filtered_ids = filter_result.doc_ids

        if not filtered_ids:
            logger.warning("No chunks match the filters")
            return RetrievalResponse(
                results=[],
                query=query,
                total_candidates=self.faiss_index.ntotal,
                filtered_candidates=0,
                filter_stats=filter_result.filter_stats,
            )

        # Get FAISS indices for filtered chunks
        filtered_faiss_indices = [
            self.reverse_id_mapping[doc_id]
            for doc_id in filtered_ids
            if doc_id in self.reverse_id_mapping
        ]

        if not filtered_faiss_indices:
            logger.warning("No filtered chunks found in vector index")
            return RetrievalResponse(
                results=[],
                query=query,
                total_candidates=self.faiss_index.ntotal,
                filtered_candidates=len(filtered_ids),
                filter_stats=filter_result.filter_stats,
            )

        # Create subset selector
        query_embedding = self.encode_query(query)

        # For small filtered sets, do direct computation
        if len(filtered_faiss_indices) <= 1000:
            results = self._search_subset(query_embedding, filtered_faiss_indices, k)
        else:
            # For larger sets, search all then filter
            search_k = min(k * 10, self.faiss_index.ntotal)
            scores, indices = self.faiss_index.search(query_embedding, search_k)

            filtered_set = set(filtered_faiss_indices)
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx in filtered_set:
                    results.append((score, idx))
                    if len(results) >= k:
                        break

        # Build response
        chunk_texts = self._load_chunk_texts()
        retrieval_results = []

        for score, idx in results:
            doc_id = self.id_mapping[idx]
            text = chunk_texts.get(doc_id, "")

            retrieval_results.append(RetrievalResult(
                doc_id=doc_id,
                text=text,
                score=float(score),
                vector_score=float(score),
            ))

        return RetrievalResponse(
            results=retrieval_results,
            query=query,
            total_candidates=self.faiss_index.ntotal,
            filtered_candidates=len(filtered_ids),
            filter_stats=filter_result.filter_stats,
        )

    def _search_subset(
        self,
        query_embedding: np.ndarray,
        indices: List[int],
        k: int,
    ) -> List[Tuple[float, int]]:
        """Search within a subset of vectors by direct computation."""
        import faiss

        # Reconstruct vectors for the subset
        vectors = np.zeros((len(indices), query_embedding.shape[1]), dtype=np.float32)
        for i, idx in enumerate(indices):
            vectors[i] = self.faiss_index.reconstruct(idx)

        # Compute similarities
        similarities = np.dot(vectors, query_embedding.T).flatten()

        # Get top-k
        top_k_indices = np.argsort(similarities)[::-1][:k]

        results = [
            (float(similarities[i]), indices[i])
            for i in top_k_indices
        ]

        return results

    def _post_filter_search(
        self,
        query: str,
        k: int,
        temporal_filter: Optional[Union[TemporalFilter, Dict[str, Any]]],
        spatial_filter: Optional[Union[SpatialFilter, Dict[str, Any]]],
        dimension_filters: Optional[List[Union[DimensionFilter, Dict[str, Any]]]] = None,
        expand_factor: int = 3,
    ) -> RetrievalResponse:
        """Search first, then filter results."""
        # Apply dimensional filters to get allowed IDs
        filter_result = self.dimensional_filter.filter(
            temporal_filter=temporal_filter,
            spatial_filter=spatial_filter,
            dimension_filters=dimension_filters,
        )
        filtered_ids = filter_result.doc_ids

        # Search with expanded k
        search_k = k * expand_factor
        query_embedding = self.encode_query(query)
        scores, indices = self.faiss_index.search(query_embedding, search_k)

        # Filter and build results
        chunk_texts = self._load_chunk_texts()
        results = []

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue

            doc_id = self.id_mapping[idx]

            # Apply filter
            if doc_id not in filtered_ids:
                continue

            text = chunk_texts.get(doc_id, "")
            results.append(RetrievalResult(
                doc_id=doc_id,
                text=text,
                score=float(score),
                vector_score=float(score),
            ))

            if len(results) >= k:
                break

        return RetrievalResponse(
            results=results,
            query=query,
            total_candidates=self.faiss_index.ntotal,
            filtered_candidates=len(filtered_ids),
            filter_stats=filter_result.filter_stats,
        )

    def search_batch(
        self,
        queries: List[str],
        k: int = 10,
        temporal_filter: Optional[Union[TemporalFilter, Dict[str, Any]]] = None,
        spatial_filter: Optional[Union[SpatialFilter, Dict[str, Any]]] = None,
    ) -> List[RetrievalResponse]:
        """
        Search for multiple queries in batch.

        Args:
            queries: List of query texts
            k: Number of results per query
            temporal_filter: Shared temporal filter for all queries
            spatial_filter: Shared spatial filter for all queries

        Returns:
            List of RetrievalResponse, one per query
        """
        # For now, process sequentially
        # TODO: Optimize with batch encoding and parallel filtering
        return [
            self.search(query, k, temporal_filter, spatial_filter)
            for query in queries
        ]

    def get_chunk_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Get full chunk data by ID."""
        chunk_texts = self._load_chunk_texts()
        text = chunk_texts.get(doc_id)

        if text is None:
            return None

        return {
            "doc_id": doc_id,
            "text": text,
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get retriever statistics."""
        return {
            "vector_index_path": str(self.vector_index_path),
            "num_vectors": self.faiss_index.ntotal,
            "embedding_dim": self.index_config.get("embedding_dim"),
            "encoder_model": self.index_config.get("model_name"),
            "dimensional_filter_enabled": self.dimensional_filter is not None,
            "temporal_keys": (
                len(self.dimensional_filter.temporal_index)
                if self.dimensional_filter else 0
            ),
            "spatial_keys": (
                len(self.dimensional_filter.spatial_index)
                if self.dimensional_filter else 0
            ),
            "generic_dimensions": (
                self.dimensional_filter.get_available_dimensions()
                if self.dimensional_filter else []
            ),
        }
