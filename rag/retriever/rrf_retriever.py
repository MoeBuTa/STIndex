"""
RRF (Reciprocal Rank Fusion) Retriever for MedRAG.

Implements RRF-4: Combines 4 retrievers using Reciprocal Rank Fusion
- BM25 (lexical)
- Contriever (general-domain semantic)
- SPECTER (scientific-domain semantic)
- MedCPT (biomedical-domain semantic)

Based on MedRAG: https://github.com/Teddy-XiongGZ/MedRAG

Usage:
    from rag.retriever.rrf_retriever import RRFRetriever

    retriever = RRFRetriever(
        corpus_path="data/original/medcorp/train.jsonl",
        indices_dir="data/indices/medcorp",
        k=32,
        rrf_k=60
    )

    results = retriever.retrieve("What causes fever?", k=10)
"""

import json
import numpy as np
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
        device: str = "cpu"
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
        """
        self.corpus_path = Path(corpus_path)
        self.indices_dir = Path(indices_dir)
        self.rrf_k = rrf_k
        self.device = device

        # Load corpus
        logger.info(f"Loading corpus from {self.corpus_path}")
        self.documents = self._load_corpus()
        logger.info(f"Loaded {len(self.documents)} documents")

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

        logger.info(f"✓ RRF-{len(self.retrievers)} retriever ready")
        logger.info(f"  Active retrievers: {', '.join(self.retrievers.keys())}")

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
        retrieval_k: int = 100
    ) -> List[RetrievalResult]:
        """
        Retrieve documents using RRF fusion.

        Args:
            query: Query string
            k: Number of final results to return
            retrieval_k: Number of results to retrieve from each retriever before fusion

        Returns:
            List of RetrievalResult objects sorted by RRF score
        """
        # Retrieve from each retriever
        all_rankings = {}

        for retriever_name, retriever_data in self.retrievers.items():
            if retriever_name == "bm25":
                rankings = self._retrieve_bm25(query, retrieval_k, retriever_data)
            else:
                rankings = self._retrieve_dense(query, retrieval_k, retriever_data)

            all_rankings[retriever_name] = rankings

        # Fuse rankings with RRF
        fused_results = self._fuse_rrf(all_rankings, k)

        return fused_results

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
