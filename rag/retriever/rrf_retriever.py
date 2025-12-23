"""
RRF (Reciprocal Rank Fusion) Retriever for MedRAG.

Implements RRF-4: Combines 4 retrievers using Reciprocal Rank Fusion
- BM25 (lexical)
- Contriever (general-domain semantic)
- SPECTER (scientific-domain semantic)
- MedCPT (biomedical-domain semantic)

Supports hierarchical dimensional filtering for dimension-aware retrieval.

Based on MedRAG: https://github.com/Teddy-XiongGZ/MedRAG

Usage:
    from rag.retriever.rrf_retriever import RRFRetriever

    retriever = RRFRetriever(
        corpus_path="data/original/medcorp/train.jsonl",
        indices_dir="data/indices/medcorp",
        k=32,
        rrf_k=60
    )

    # Basic retrieval
    results = retriever.retrieve("What causes fever?", k=10)

    # Retrieval with dimensional filtering
    results = retriever.retrieve(
        "What causes fever?",
        k=10,
        dimension_filters=[{"dimension": "symptom", "values": ["fever"]}]
    )
"""

import json
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, Any

import faiss
from loguru import logger
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


@dataclass
class RetrievalResult:
    """Single retrieval result."""
    doc_id: str
    title: str
    contents: str
    score: float
    rank: int
    source_corpus: str = ""
    metadata: Dict = field(default_factory=dict)


class RRFRetriever:
    """
    RRF-4 Retriever combining 4 retrievers with Reciprocal Rank Fusion.

    Fusion formula: score = sum(1 / (rrf_k + rank_i + 1))

    Supports hierarchical dimensional filtering for dimension-aware retrieval.
    """

    def __init__(
        self,
        corpus_path: str,
        indices_dir: str,
        use_bm25: bool = True,
        use_contriever: bool = True,
        use_specter: bool = True,
        use_medcpt: bool = True,
        rrf_k: int = 60,
        device: str = "cpu",
        use_dimensions: bool = False
    ):
        """
        Initialize RRF retriever.

        Args:
            corpus_path: Path to corpus JSONL file
            indices_dir: Directory containing FAISS indices
            use_bm25: Whether to include BM25 retriever
            use_contriever: Whether to include Contriever
            use_specter: Whether to include SPECTER
            use_medcpt: Whether to include MedCPT
            rrf_k: RRF smoothing parameter (default 60)
            device: Device for neural models
            use_dimensions: Whether to load dimensional indexes for filtering
        """
        self.corpus_path = Path(corpus_path)
        self.indices_dir = Path(indices_dir)
        self.rrf_k = rrf_k
        self.device = device

        # Load corpus
        logger.info(f"Loading corpus from {self.corpus_path}")
        self.documents = self._load_corpus()
        logger.info(f"Loaded {len(self.documents)} documents")

        # Build doc_id -> index mapping for filtering
        self.doc_id_to_idx = {
            doc.get("doc_id", ""): idx
            for idx, doc in enumerate(self.documents)
        }

        # Initialize retrievers
        self.retrievers = {}

        if use_bm25:
            logger.info("Initializing BM25 retriever...")
            self._init_bm25()

        if use_contriever:
            logger.info("Initializing Contriever...")
            self._init_dense_retriever("contriever", "facebook/contriever")

        if use_specter:
            logger.info("Initializing SPECTER...")
            self._init_dense_retriever("specter", "allenai/specter")

        if use_medcpt:
            logger.info("Initializing MedCPT...")
            self._init_dense_retriever("medcpt", "ncbi/MedCPT-Query-Encoder")

        # Load dimensional indexes if requested
        self.dimension_index = None
        self.chunks_metadata = None
        if use_dimensions:
            self._load_dimension_indexes()

        logger.info(f"✓ RRF-{len(self.retrievers)} retriever ready")
        logger.info(f"  Active retrievers: {', '.join(self.retrievers.keys())}")
        if self.dimension_index:
            logger.info(f"  Dimensional filtering: enabled ({len(self.dimension_index)} dimensions)")

    def _load_dimension_indexes(self):
        """Load dimensional indexes for filtering."""
        dimension_index_path = self.indices_dir / "indexes" / "dimension_index.json"
        chunks_metadata_path = self.indices_dir / "chunks_metadata.jsonl"

        if dimension_index_path.exists():
            logger.info(f"Loading dimension index from {dimension_index_path}")
            with open(dimension_index_path) as f:
                self.dimension_index = json.load(f)
            logger.info(f"  Loaded {len(self.dimension_index)} dimensions")
        else:
            logger.warning(f"Dimension index not found: {dimension_index_path}")

        if chunks_metadata_path.exists():
            logger.info(f"Loading chunks metadata from {chunks_metadata_path}")
            self.chunks_metadata = {}
            with open(chunks_metadata_path) as f:
                for line in f:
                    doc = json.loads(line)
                    doc_id = doc.get("doc_id", doc.get("chunk_id", ""))
                    self.chunks_metadata[doc_id] = doc
            logger.info(f"  Loaded {len(self.chunks_metadata)} chunks")

    def _load_corpus(self) -> List[Dict]:
        """Load corpus from JSONL file."""
        documents = []
        with open(self.corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    documents.append(json.loads(line))
        return documents

    def _init_bm25(self):
        """Initialize BM25 retriever."""
        # Tokenize documents
        logger.info("  Tokenizing documents for BM25...")
        tokenized_docs = []
        for doc in self.documents:
            text = doc.get("contents", "")
            tokens = text.lower().split()
            tokenized_docs.append(tokens)

        # Create BM25 index
        logger.info("  Building BM25 index...")
        bm25 = BM25Okapi(tokenized_docs)

        self.retrievers["bm25"] = {
            "index": bm25,
            "tokenized_docs": tokenized_docs
        }
        logger.info("  ✓ BM25 ready")

    def _init_dense_retriever(self, name: str, model_name: str):
        """
        Initialize dense retriever from FAISS index.

        Args:
            name: Retriever name (contriever, specter, medcpt)
            model_name: HuggingFace model name for query encoding
        """
        # Load FAISS index
        index_file = self.indices_dir / f"{name}_index.faiss"
        if not index_file.exists():
            logger.warning(f"  ✗ Index not found: {index_file}")
            logger.warning(f"  Skipping {name}")
            return

        logger.info(f"  Loading FAISS index: {index_file}")
        index = faiss.read_index(str(index_file))

        # Load config
        config_file = self.indices_dir / f"{name}_config.json"
        with open(config_file, "r") as f:
            config = json.load(f)

        # Load encoder model
        logger.info(f"  Loading encoder: {model_name}")
        encoder = SentenceTransformer(model_name, device=self.device)

        self.retrievers[name] = {
            "index": index,
            "encoder": encoder,
            "config": config
        }
        logger.info(f"  ✓ {name.upper()} ready ({index.ntotal:,} docs)")

    def retrieve(
        self,
        query: str,
        k: int = 32,
        retrieval_k: int = 100,
        dimension_filters: Optional[List[Dict[str, Any]]] = None,
        filter_mode: str = "post",
        hierarchical: bool = True
    ) -> List[RetrievalResult]:
        """
        Retrieve documents using RRF fusion with optional dimensional filtering.

        Args:
            query: Query string
            k: Number of final results to return
            retrieval_k: Number of results to retrieve from each retriever before fusion
            dimension_filters: List of dimension filter dicts, e.g.:
                [{"dimension": "symptom", "values": ["fever", "cough"]},
                 {"dimension": "drug", "values": ["aspirin"]}]
                Multiple values in same dimension are OR'd, multiple dimensions are AND'd
            filter_mode: "post" (filter after RRF) or "pre" (filter before retrieval)
            hierarchical: If True, query at any level matches all hierarchy levels

        Returns:
            List of RetrievalResult objects sorted by RRF score
        """
        # Get allowed doc_ids if filtering
        allowed_doc_ids = None
        if dimension_filters and self.dimension_index:
            allowed_doc_ids = self._apply_dimension_filters(
                dimension_filters, hierarchical
            )
            logger.debug(f"Dimension filter matched {len(allowed_doc_ids)} documents")

            if not allowed_doc_ids:
                logger.warning("Dimension filter matched 0 documents, returning empty results")
                return []

            # For pre-filter mode, we need to expand retrieval_k to ensure enough candidates
            if filter_mode == "pre" and allowed_doc_ids:
                # Estimate expansion factor based on filter selectivity
                selectivity = len(allowed_doc_ids) / len(self.documents)
                if selectivity > 0:
                    retrieval_k = min(int(retrieval_k / selectivity), len(self.documents))

        # Retrieve from each retriever
        all_rankings = {}

        for retriever_name, retriever_data in self.retrievers.items():
            if retriever_name == "bm25":
                rankings = self._retrieve_bm25(query, retrieval_k, retriever_data)
            else:
                rankings = self._retrieve_dense(query, retrieval_k, retriever_data)

            all_rankings[retriever_name] = rankings

        # Fuse rankings with RRF
        fused_results = self._fuse_rrf(all_rankings, k * 10 if allowed_doc_ids else k)

        # Apply post-filtering if needed
        if allowed_doc_ids and filter_mode == "post":
            fused_results = [
                r for r in fused_results
                if r.doc_id in allowed_doc_ids
            ]
            # Re-rank after filtering
            for i, result in enumerate(fused_results):
                result.rank = i + 1

        return fused_results[:k]

    def _apply_dimension_filters(
        self,
        dimension_filters: List[Dict[str, Any]],
        hierarchical: bool = True
    ) -> Set[str]:
        """
        Apply dimension filters and return matching doc_ids.

        Args:
            dimension_filters: List of filter dicts with 'dimension' and 'values' keys
            hierarchical: If True, match at any hierarchy level

        Returns:
            Set of matching doc_ids (intersection of all dimension filters)
        """
        if not dimension_filters or not self.dimension_index:
            return set()

        # Start with all documents
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

    def get_dimension_stats(self, dimension: str) -> Dict[str, int]:
        """
        Get statistics for a specific dimension.

        Args:
            dimension: Dimension name (e.g., "symptom", "drug")

        Returns:
            Dict with top labels and their document counts
        """
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
            reverse=True
        )

        return {label: count for label, count in sorted_labels[:50]}

    def _retrieve_bm25(
        self,
        query: str,
        k: int,
        retriever_data: Dict
    ) -> List[Tuple[int, float]]:
        """
        Retrieve using BM25.

        Returns:
            List of (doc_idx, score) tuples
        """
        bm25 = retriever_data["index"]
        tokenized_query = query.lower().split()

        scores = bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:k]

        rankings = [(int(idx), float(scores[idx])) for idx in top_indices]
        return rankings

    def _retrieve_dense(
        self,
        query: str,
        k: int,
        retriever_data: Dict
    ) -> List[Tuple[int, float]]:
        """
        Retrieve using dense retriever (Contriever/SPECTER/MedCPT).

        Returns:
            List of (doc_idx, score) tuples
        """
        index = retriever_data["index"]
        encoder = retriever_data["encoder"]

        # Encode query
        query_embedding = encoder.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype(np.float32)

        # Search
        scores, indices = index.search(query_embedding, k)

        rankings = [
            (int(indices[0][i]), float(scores[0][i]))
            for i in range(len(indices[0]))
            if indices[0][i] >= 0  # Filter out invalid indices
        ]

        return rankings

    def _fuse_rrf(
        self,
        all_rankings: Dict[str, List[Tuple[int, float]]],
        k: int
    ) -> List[RetrievalResult]:
        """
        Fuse multiple rankings using Reciprocal Rank Fusion.

        RRF formula: score(doc) = sum_{r in rankings} 1 / (rrf_k + rank_r(doc))

        Args:
            all_rankings: Dict of {retriever_name: [(doc_idx, score), ...]}
            k: Number of results to return

        Returns:
            List of RetrievalResult objects
        """
        # Compute RRF scores
        rrf_scores = defaultdict(float)

        for retriever_name, rankings in all_rankings.items():
            for rank, (doc_idx, _) in enumerate(rankings):
                rrf_scores[doc_idx] += 1.0 / (self.rrf_k + rank + 1)

        # Sort by RRF score
        sorted_docs = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:k]

        # Build results
        results = []
        for rank, (doc_idx, rrf_score) in enumerate(sorted_docs):
            doc = self.documents[doc_idx]

            result = RetrievalResult(
                doc_id=doc.get("doc_id", ""),
                title=doc.get("title", ""),
                contents=doc.get("contents", ""),
                score=rrf_score,
                rank=rank + 1,
                source_corpus=doc.get("metadata", {}).get("source_corpus", ""),
                metadata=doc.get("metadata", {})
            )
            results.append(result)

        return results

    def get_stats(self) -> Dict:
        """Get retriever statistics."""
        stats = {
            "num_documents": len(self.documents),
            "num_retrievers": len(self.retrievers),
            "active_retrievers": list(self.retrievers.keys()),
            "rrf_k": self.rrf_k
        }

        for name, data in self.retrievers.items():
            if name == "bm25":
                stats[f"{name}_vocab_size"] = len(set(
                    token for doc in data["tokenized_docs"] for token in doc
                ))
            else:
                stats[f"{name}_index_size"] = data["index"].ntotal

        return stats
