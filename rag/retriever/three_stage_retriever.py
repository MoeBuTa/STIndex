"""
Three-Stage Hybrid Retriever with Dimensional Pre-Filtering.

Implements:
1. Stage 0 (optional): Dimensional Pre-Filtering using FAISS IDSelector
2. Stage 1: Dense Retrieval (FAISS IVF) → 100 candidates
3. Stage 2: Sparse Reranking (BGE-M3 lexical) → 30 candidates
4. Stage 3: Cross-Encoder Reranking (BGE-reranker-v2-m3) → top-k

Supports two dimensional retrieval modes:
- Pre-filtering: Filter FAISS search to only dimension-matched docs (faster, exact)
- Early fusion: Boost dimension-matched docs in ranking (softer, more recall)

Usage:
    from rag.retriever.three_stage_retriever import ThreeStageRetriever

    retriever = ThreeStageRetriever(
        index_path="data/indices/medcorp_bgem3",
        dimension_index_path="data/indices/medcorp/indexes",
        device="cuda",
    )

    # Basic retrieval
    results = retriever.retrieve("What causes diabetes?", k=10)

    # With dimensional pre-filtering (recommended)
    results = retriever.retrieve(
        "What causes diabetes?",
        k=10,
        dimension_filters=[{"dimension": "drug", "values": ["insulin"]}],
        dimension_mode="prefilter",  # or "boost" for early fusion
    )
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from loguru import logger


@dataclass
class RetrievalResult:
    """Single retrieval result with scores from each stage."""

    doc_id: str
    title: str
    text: str
    dense_score: float = 0.0
    sparse_score: float = 0.0
    hybrid_score: float = 0.0
    rerank_score: float = 0.0
    final_score: float = 0.0
    rank: int = 0
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "text": self.text,
            "dense_score": self.dense_score,
            "sparse_score": self.sparse_score,
            "hybrid_score": self.hybrid_score,
            "rerank_score": self.rerank_score,
            "final_score": self.final_score,
            "rank": self.rank,
            "metadata": self.metadata,
        }


class ThreeStageRetriever:
    """
    Three-stage hybrid retriever with dimensional pre-filtering.

    Stages:
    0. (Optional) Dimensional pre-filter: Get FAISS indices matching dimensions
    1. Dense: FAISS search with BGE-M3 embeddings (optionally filtered)
    2. Sparse: Hybrid reranking with BGE-M3 lexical weights
    3. Rerank: Cross-encoder reranking with BGE-reranker-v2-m3

    Dimension modes:
    - "prefilter": FAISS searches only within dimension-matched docs
    - "boost": Dimension-matched docs get score boost during ranking
    - "postfilter": Legacy mode, filters after Stage 3
    """

    def __init__(
        self,
        index_path: str,
        dimension_index_path: Optional[str] = None,
        encoder_model: str = "BAAI/bge-m3",
        reranker_model: str = "BAAI/bge-reranker-v2-m3",
        device: str = "cuda",
        load_sparse: bool = True,
        load_reranker: bool = True,
        load_dimensions: bool = True,
    ):
        """
        Initialize three-stage retriever.

        Args:
            index_path: Path to BGE-M3 index directory
            dimension_index_path: Path to dimension index directory
            encoder_model: BGE-M3 encoder model name
            reranker_model: Cross-encoder model name
            device: Device for models (cuda/cpu)
            load_sparse: Whether to load sparse matrix
            load_reranker: Whether to load reranker model
            load_dimensions: Whether to load dimension indexes
        """
        self.index_path = Path(index_path)
        self.device = device

        # Load FAISS index
        logger.info(f"Loading FAISS index from {self.index_path}...")
        self._load_faiss_index()

        # Load metadata
        logger.info("Loading passages metadata...")
        self._load_metadata()

        # Load encoder
        logger.info(f"Loading BGE-M3 encoder on {device}...")
        self._load_encoder(encoder_model)

        # Load sparse matrix (optional)
        self.sparse_matrix = None
        if load_sparse:
            self._load_sparse_matrix()

        # Load reranker (optional, lazy)
        self.reranker = None
        self.reranker_model = reranker_model
        if load_reranker:
            self._load_reranker()

        # Load dimension indexes (optional)
        # dimension_index: doc_id based (for postfilter/boost)
        # dimension_index_faiss: faiss_idx based (for prefilter)
        self.dimension_index = None
        self.dimension_index_faiss = None
        if load_dimensions and dimension_index_path:
            self._load_dimension_indexes(Path(dimension_index_path))

        logger.info("ThreeStageRetriever initialized")
        logger.info(f"  Index: {self.index.ntotal:,} vectors")
        logger.info(f"  Passages: {len(self.passages):,}")
        if self.sparse_matrix is not None:
            logger.info(f"  Sparse: {self.sparse_matrix.shape}")
        if self.dimension_index:
            logger.info(f"  Dimensions: {len(self.dimension_index)}")
        if self.dimension_index_faiss:
            logger.info(f"  Dimensions (FAISS-indexed): {len(self.dimension_index_faiss)}")

    def _load_faiss_index(self):
        """Load FAISS index."""
        import faiss

        index_file = self.index_path / "faiss_index.bin"
        if not index_file.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_file}")

        self.index = faiss.read_index(str(index_file))

        # Load config
        config_file = self.index_path / "index_config.json"
        if config_file.exists():
            with open(config_file) as f:
                self.config = json.load(f)
        else:
            self.config = {}

    def _load_metadata(self):
        """Load passages metadata."""
        metadata_file = self.index_path / "passages_metadata.jsonl"
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_file}")

        self.passages = []
        self.doc_id_to_idx = {}

        with open(metadata_file) as f:
            for idx, line in enumerate(f):
                doc = json.loads(line)
                self.passages.append(doc)
                doc_id = doc.get("doc_id", str(idx))
                self.doc_id_to_idx[doc_id] = idx

    def _load_encoder(self, model_name: str):
        """Load BGE-M3 encoder."""
        try:
            from FlagEmbedding import BGEM3FlagModel
        except ImportError:
            raise ImportError("FlagEmbedding not installed. Install with: pip install FlagEmbedding")

        self.encoder = BGEM3FlagModel(
            model_name,
            use_fp16=True,
            device=self.device,
        )

    def _load_sparse_matrix(self):
        """Load precomputed sparse matrix."""
        import scipy.sparse as sp

        sparse_file = self.index_path / "sparse_weights.npz"
        if sparse_file.exists():
            logger.info(f"Loading sparse matrix from {sparse_file}...")
            self.sparse_matrix = sp.load_npz(str(sparse_file))
            logger.info(f"  Sparse matrix: {self.sparse_matrix.shape}, nnz={self.sparse_matrix.nnz:,}")
        else:
            logger.warning(f"Sparse matrix not found: {sparse_file}")
            logger.warning("  Stage 2 will use dynamic sparse encoding (slower)")

    def _load_reranker(self):
        """Load cross-encoder reranker."""
        try:
            from FlagEmbedding import FlagReranker
        except ImportError:
            logger.warning("FlagEmbedding FlagReranker not available, using transformers")
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder(self.reranker_model, device=self.device)
            return

        logger.info(f"Loading reranker: {self.reranker_model}...")
        self.reranker = FlagReranker(
            self.reranker_model,
            use_fp16=True,
            device=self.device,
        )

    def _load_dimension_indexes(self, dimension_index_path: Path):
        """Load dimension indexes for filtering."""
        # Load doc_id based index (for postfilter/boost mode)
        dimension_index_file = dimension_index_path / "dimension_index.json"
        if dimension_index_file.exists():
            logger.info(f"Loading dimension index from {dimension_index_file}...")
            with open(dimension_index_file) as f:
                self.dimension_index = json.load(f)
            logger.info(f"  Loaded {len(self.dimension_index)} dimensions (doc_id based)")

        # Load faiss_idx based index (for prefilter mode)
        dimension_index_faiss_file = dimension_index_path / "dimension_index_faiss.json"
        if dimension_index_faiss_file.exists():
            logger.info(f"Loading FAISS-indexed dimension index from {dimension_index_faiss_file}...")
            with open(dimension_index_faiss_file) as f:
                self.dimension_index_faiss = json.load(f)
            logger.info(f"  Loaded {len(self.dimension_index_faiss)} dimensions (faiss_idx based)")
        else:
            logger.warning(f"FAISS-indexed dimension index not found: {dimension_index_faiss_file}")
            logger.warning("  Prefilter mode will not be available. Run convert_dimension_index_to_faiss_idx.py")

    def _encode_query(self, query: str) -> Tuple[np.ndarray, Dict[int, float]]:
        """Encode query to get dense and sparse representations."""
        outputs = self.encoder.encode(
            [query],
            batch_size=1,
            max_length=512,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )

        dense_vec = outputs["dense_vecs"][0]
        sparse_weights = outputs["lexical_weights"][0]

        return dense_vec, sparse_weights

    def _get_faiss_indices_for_dimension_filters(
        self,
        dimension_filters: List[Dict[str, Any]],
        hierarchical: bool = True,
    ) -> Optional[np.ndarray]:
        """
        Get FAISS indices matching dimension filters.

        Args:
            dimension_filters: List of filter dicts with 'dimension' and 'values' keys
            hierarchical: If True, match at any hierarchy level

        Returns:
            numpy array of matching FAISS indices, or None if no filters/no matches
        """
        if not dimension_filters or not self.dimension_index_faiss:
            return None

        result_sets = []

        for filter_spec in dimension_filters:
            dim_name = filter_spec.get("dimension", "")
            values = filter_spec.get("values", [])

            if not dim_name or not values:
                continue

            # Normalize dimension name
            dim_name_normalized = dim_name.strip().lower().replace(" ", "_")

            if dim_name_normalized not in self.dimension_index_faiss:
                logger.warning(f"Dimension '{dim_name}' not found in FAISS index")
                continue

            dim_data = self.dimension_index_faiss[dim_name_normalized]
            labels_index = dim_data.get("labels", {})
            paths_index = dim_data.get("paths", {})

            # Collect matching faiss_indices for this dimension (OR across values)
            dim_matches = set()
            for value in values:
                value_normalized = value.strip().lower()

                # Match in labels index
                if value_normalized in labels_index:
                    dim_matches.update(labels_index[value_normalized])

                # If hierarchical, also match in paths
                if hierarchical:
                    for path, faiss_indices in paths_index.items():
                        if value_normalized in path:
                            dim_matches.update(faiss_indices)

            if dim_matches:
                result_sets.append(dim_matches)

        # AND across dimensions (intersection)
        if not result_sets:
            return None

        result = result_sets[0]
        for s in result_sets[1:]:
            result = result.intersection(s)

        if not result:
            return None

        # Convert to sorted numpy array
        return np.array(sorted(result), dtype=np.int64)

    def _stage1_dense_prefiltered(
        self,
        query_vec: np.ndarray,
        allowed_indices: np.ndarray,
        k: int = 100,
        nprobe: int = 256,
    ) -> List[Tuple[int, float]]:
        """
        Stage 1: Dense retrieval with FAISS ID pre-filtering.

        Uses faiss.IDSelectorBatch to restrict search to allowed_indices only.

        Args:
            query_vec: Query embedding
            allowed_indices: Array of allowed FAISS indices
            k: Number of candidates to retrieve
            nprobe: Number of IVF clusters to search

        Returns:
            List of (doc_idx, dense_score) tuples
        """
        import faiss

        # Set nprobe for IVF index
        if hasattr(self.index, "nprobe"):
            self.index.nprobe = nprobe

        # Create ID selector for allowed indices
        selector = faiss.IDSelectorBatch(allowed_indices)

        # Search with ID selector
        query_vec = query_vec.astype(np.float32).reshape(1, -1)

        # Adjust k to not exceed allowed indices
        k_actual = min(k, len(allowed_indices))

        # Use search_with_selector for FAISS index
        params = faiss.SearchParametersIVF()
        params.nprobe = nprobe
        params.sel = selector

        scores, indices = self.index.search(query_vec, k_actual, params=params)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:  # Valid index
                results.append((int(idx), float(score)))

        logger.debug(f"Prefiltered search: {len(allowed_indices)} allowed → {len(results)} results")
        return results

    def _stage1_dense(
        self,
        query_vec: np.ndarray,
        k: int = 100,
        nprobe: int = 256,
    ) -> List[Tuple[int, float]]:
        """
        Stage 1: Dense retrieval using FAISS.

        Returns:
            List of (doc_idx, dense_score) tuples
        """
        # Set nprobe for IVF index
        if hasattr(self.index, "nprobe"):
            self.index.nprobe = nprobe

        # Search
        query_vec = query_vec.astype(np.float32).reshape(1, -1)
        scores, indices = self.index.search(query_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:  # Valid index
                results.append((int(idx), float(score)))

        return results

    def _stage2_sparse(
        self,
        query_sparse: Dict[int, float],
        candidates: List[Tuple[int, float]],
        k: int = 30,
        alpha: float = 0.65,
    ) -> List[Tuple[int, float, float, float]]:
        """
        Stage 2: Sparse reranking with hybrid score.

        Hybrid formula: score = alpha * dense_norm + (1-alpha) * sparse_norm

        Returns:
            List of (doc_idx, dense_score, sparse_score, hybrid_score) tuples
        """
        if not candidates:
            return []

        doc_indices = [idx for idx, _ in candidates]
        dense_scores = np.array([score for _, score in candidates])

        # Normalize dense scores to [0, 1]
        if len(dense_scores) > 1:
            dense_min, dense_max = dense_scores.min(), dense_scores.max()
            if dense_max > dense_min:
                dense_norm = (dense_scores - dense_min) / (dense_max - dense_min)
            else:
                dense_norm = np.ones_like(dense_scores)
        else:
            dense_norm = np.ones(len(dense_scores))

        # Compute sparse scores
        if self.sparse_matrix is not None:
            # Use precomputed sparse matrix
            # Note: BGE-M3 returns string keys, convert to int for sparse matrix indexing
            query_token_ids = [int(tid) for tid in query_sparse.keys()]
            query_weights = np.array([query_sparse[tid] for tid in query_sparse.keys()])

            # Get sparse vectors for candidates
            doc_sparse = self.sparse_matrix[doc_indices][:, query_token_ids].toarray()
            sparse_scores = (doc_sparse * query_weights).sum(axis=1)
        else:
            # Dynamic sparse computation (slower)
            sparse_scores = np.zeros(len(candidates))
            logger.warning("Dynamic sparse computation not implemented, using zero scores")

        # Normalize sparse scores to [0, 1]
        if len(sparse_scores) > 1:
            sparse_min, sparse_max = sparse_scores.min(), sparse_scores.max()
            if sparse_max > sparse_min:
                sparse_norm = (sparse_scores - sparse_min) / (sparse_max - sparse_min)
            else:
                sparse_norm = np.ones_like(sparse_scores)
        else:
            sparse_norm = np.ones(len(sparse_scores))

        # Compute hybrid scores
        hybrid_scores = alpha * dense_norm + (1 - alpha) * sparse_norm

        # Sort by hybrid score and return top-k
        results = [
            (doc_indices[i], float(dense_scores[i]), float(sparse_scores[i]), float(hybrid_scores[i]))
            for i in range(len(candidates))
        ]
        results.sort(key=lambda x: x[3], reverse=True)

        return results[:k]

    def _stage3_rerank(
        self,
        query: str,
        candidates: List[Tuple[int, float, float, float]],
        k: int = 10,
    ) -> List[Tuple[int, float, float, float, float]]:
        """
        Stage 3: Cross-encoder reranking.

        Returns:
            List of (doc_idx, dense_score, sparse_score, hybrid_score, rerank_score) tuples
        """
        if not candidates or self.reranker is None:
            return [(idx, ds, ss, hs, hs) for idx, ds, ss, hs in candidates[:k]]

        # Build query-document pairs
        pairs = []
        for doc_idx, _, _, _ in candidates:
            doc = self.passages[doc_idx]
            full_text = f"{doc.get('title', '')}. {doc.get('text', '')}"[:512]
            pairs.append([query, full_text])

        # Compute rerank scores
        try:
            if hasattr(self.reranker, "compute_score"):
                # FlagReranker
                rerank_scores = self.reranker.compute_score(pairs, normalize=True)
            else:
                # CrossEncoder
                rerank_scores = self.reranker.predict(pairs)
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return [(idx, ds, ss, hs, hs) for idx, ds, ss, hs in candidates[:k]]

        # Combine with hybrid scores
        results = [
            (
                candidates[i][0],  # doc_idx
                candidates[i][1],  # dense_score
                candidates[i][2],  # sparse_score
                candidates[i][3],  # hybrid_score
                float(rerank_scores[i]),  # rerank_score
            )
            for i in range(len(candidates))
        ]

        # Sort by rerank score
        results.sort(key=lambda x: x[4], reverse=True)

        return results[:k]

    def _apply_dimension_filters(
        self,
        candidates: List[Tuple],
        dimension_filters: List[Dict[str, Any]],
        hierarchical: bool = True,
    ) -> List[Tuple]:
        """
        Apply dimensional filtering to candidates.

        Args:
            candidates: List of candidate tuples (doc_idx, ...)
            dimension_filters: List of filter dicts with 'dimension' and 'values' keys
            hierarchical: If True, match at any hierarchy level

        Returns:
            Filtered list of candidates
        """
        if not dimension_filters or not self.dimension_index:
            return candidates

        # Get allowed doc_ids from dimension filters
        allowed_doc_ids = self._get_allowed_doc_ids(dimension_filters, hierarchical)

        if not allowed_doc_ids:
            logger.warning("Dimension filter matched 0 documents")
            return []

        # Filter candidates
        filtered = []
        for candidate in candidates:
            doc_idx = candidate[0]
            doc_id = self.passages[doc_idx].get("doc_id", str(doc_idx))
            if doc_id in allowed_doc_ids:
                filtered.append(candidate)

        logger.debug(f"Dimension filter: {len(candidates)} → {len(filtered)} candidates")
        return filtered

    def _get_allowed_doc_ids(
        self,
        dimension_filters: List[Dict[str, Any]],
        hierarchical: bool = True,
    ) -> Set[str]:
        """Get set of doc_ids that match all dimension filters."""
        result_sets = []

        for filter_spec in dimension_filters:
            dim_name = filter_spec.get("dimension", "")
            values = filter_spec.get("values", [])

            if not dim_name or not values:
                continue

            # Normalize dimension name
            dim_name_normalized = dim_name.strip().lower().replace(" ", "_")

            if dim_name_normalized not in self.dimension_index:
                logger.warning(f"Dimension '{dim_name}' not found in index")
                continue

            dim_data = self.dimension_index[dim_name_normalized]
            labels_index = dim_data.get("labels", {})
            paths_index = dim_data.get("paths", {})

            # Collect matching doc_ids for this dimension (OR across values)
            dim_matches = set()
            for value in values:
                value_normalized = value.strip().lower()

                # Match in labels index
                if value_normalized in labels_index:
                    dim_matches.update(labels_index[value_normalized])

                # If hierarchical, also match in paths
                if hierarchical:
                    for path, doc_ids in paths_index.items():
                        if value_normalized in path:
                            dim_matches.update(doc_ids)

            if dim_matches:
                result_sets.append(dim_matches)

        # AND across dimensions (intersection)
        if not result_sets:
            return set()

        result = result_sets[0]
        for s in result_sets[1:]:
            result = result.intersection(s)

        return result

    def retrieve(
        self,
        query: str,
        k: int = 10,
        nprobe: int = 256,
        stage1_k: int = 100,
        stage2_k: int = 30,
        hybrid_alpha: float = 0.65,
        use_sparse: bool = True,
        use_reranker: bool = True,
        dimension_filters: Optional[List[Dict[str, Any]]] = None,
        hierarchical: bool = True,
        dimension_mode: str = "prefilter",
        dim_boost_weight: float = 0.3,
    ) -> List[RetrievalResult]:
        """
        Retrieve documents using three-stage hybrid retrieval.

        Args:
            query: Query string
            k: Number of final results to return
            nprobe: FAISS search clusters
            stage1_k: Candidates after Stage 1 (dense)
            stage2_k: Candidates after Stage 2 (sparse)
            hybrid_alpha: Dense weight in hybrid score (0-1)
            use_sparse: Whether to use Stage 2
            use_reranker: Whether to use Stage 3
            dimension_filters: Optional dimension filters
            hierarchical: Whether to use hierarchical matching for filters
            dimension_mode: How to apply dimension filters:
                - "prefilter": Filter FAISS search to dimension-matched docs (recommended)
                - "boost": Boost dimension-matched docs in ranking
                - "postfilter": Legacy mode, filter after Stage 3
            dim_boost_weight: Weight for dimension boost (0-1, only for mode="boost")

        Returns:
            List of RetrievalResult objects
        """
        # Encode query
        query_vec, query_sparse = self._encode_query(query)

        # Get dimension-matched FAISS indices (for prefilter/boost modes)
        dim_matched_indices = None
        dim_matched_set = None
        if dimension_filters and dimension_mode in ("prefilter", "boost"):
            dim_matched_indices = self._get_faiss_indices_for_dimension_filters(
                dimension_filters, hierarchical
            )
            if dim_matched_indices is not None:
                dim_matched_set = set(dim_matched_indices.tolist())
                logger.info(f"Dimension filter matched {len(dim_matched_indices):,} documents")

        # Stage 1: Dense retrieval
        if dimension_mode == "prefilter" and dim_matched_indices is not None:
            # Pre-filtered FAISS search
            stage1_results = self._stage1_dense_prefiltered(
                query_vec, dim_matched_indices, k=stage1_k, nprobe=nprobe
            )
            if not stage1_results:
                logger.warning("Prefiltered search returned 0 results, falling back to full search")
                stage1_results = self._stage1_dense(query_vec, k=stage1_k, nprobe=nprobe)
        else:
            # Standard dense search
            stage1_results = self._stage1_dense(query_vec, k=stage1_k, nprobe=nprobe)

        if not stage1_results:
            return []

        # Stage 2: Sparse reranking (optional)
        if use_sparse and (self.sparse_matrix is not None):
            stage2_results = self._stage2_sparse(
                query_sparse, stage1_results, k=stage2_k, alpha=hybrid_alpha
            )
        else:
            # Skip sparse, use dense scores
            stage2_results = [
                (idx, score, 0.0, score) for idx, score in stage1_results[:stage2_k]
            ]

        # Apply dimension boost if mode="boost"
        if dimension_mode == "boost" and dim_matched_set:
            boosted_results = []
            for doc_idx, dense, sparse, hybrid in stage2_results:
                dim_match = 1.0 if doc_idx in dim_matched_set else 0.0
                boosted_score = hybrid + dim_boost_weight * dim_match
                boosted_results.append((doc_idx, dense, sparse, boosted_score))
            boosted_results.sort(key=lambda x: x[3], reverse=True)
            stage2_results = boosted_results

        # Stage 3: Cross-encoder reranking (optional)
        if use_reranker and self.reranker is not None:
            # For postfilter mode, get more candidates before filtering
            stage3_k = k * 3 if (dimension_mode == "postfilter" and dimension_filters) else k
            stage3_results = self._stage3_rerank(query, stage2_results, k=stage3_k)
        else:
            # Skip reranking, use hybrid scores
            stage3_results = [
                (idx, ds, ss, hs, hs) for idx, ds, ss, hs in stage2_results
            ]

        # Stage 4: Dimensional post-filtering (only for mode="postfilter")
        if dimension_mode == "postfilter" and dimension_filters:
            stage3_results = self._apply_dimension_filters(
                stage3_results, dimension_filters, hierarchical
            )

        # Build final results
        results = []
        for rank, (doc_idx, dense, sparse, hybrid, rerank) in enumerate(stage3_results[:k]):
            doc = self.passages[doc_idx]
            result = RetrievalResult(
                doc_id=doc.get("doc_id", str(doc_idx)),
                title=doc.get("title", ""),
                text=doc.get("text", ""),
                dense_score=dense,
                sparse_score=sparse,
                hybrid_score=hybrid,
                rerank_score=rerank,
                final_score=rerank,
                rank=rank + 1,
                metadata=doc.get("metadata", {}),
            )
            results.append(result)

        return results

    def get_dimension_stats(self, dimension: str) -> Dict[str, int]:
        """Get statistics for a specific dimension."""
        if not self.dimension_index:
            return {}

        dim_name_normalized = dimension.strip().lower().replace(" ", "_")
        if dim_name_normalized not in self.dimension_index:
            return {}

        dim_data = self.dimension_index[dim_name_normalized]
        labels_index = dim_data.get("labels", {})

        # Sort by count
        sorted_labels = sorted(
            [(label, len(doc_ids)) for label, doc_ids in labels_index.items()],
            key=lambda x: x[1],
            reverse=True,
        )

        return {label: count for label, count in sorted_labels[:50]}

    def get_stats(self) -> Dict[str, Any]:
        """Get retriever statistics."""
        stats = {
            "index_size": self.index.ntotal,
            "num_passages": len(self.passages),
            "embedding_dim": self.config.get("embedding_dimension", 1024),
            "encoder_model": self.config.get("encoder_model", "BAAI/bge-m3"),
            "has_sparse": self.sparse_matrix is not None,
            "has_reranker": self.reranker is not None,
            "has_dimensions": self.dimension_index is not None,
        }

        if self.sparse_matrix is not None:
            stats["sparse_shape"] = self.sparse_matrix.shape
            stats["sparse_nnz"] = self.sparse_matrix.nnz

        if self.dimension_index:
            stats["num_dimensions"] = len(self.dimension_index)

        return stats
