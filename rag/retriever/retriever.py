"""
RAG Retriever Implementation.

High-level retriever for RAG pipelines with multi-hop support
and intelligent context formatting.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from loguru import logger

# Import STIndex HybridRetriever
try:
    from stindex.retrieval import HybridRetriever, TemporalFilter, SpatialFilter
except ImportError as e:
    logger.warning(f"stindex.retrieval not available: {e}")
    HybridRetriever = None
    TemporalFilter = None
    SpatialFilter = None


@dataclass
class RAGRetrieverConfig:
    """Configuration for RAG retriever."""
    # Vector index settings
    vector_index_path: str = "data/vector/rag"
    warehouse_path: Optional[str] = "data/warehouse/rag"
    encoder_model: Optional[str] = None  # Use index default
    device: str = "cpu"

    # Retrieval settings
    default_k: int = 5
    max_context_length: int = 4000  # Max chars for context
    chunk_separator: str = "\n\n"
    include_metadata: bool = True

    # Multi-hop settings
    max_hops: int = 2
    documents_per_hop: int = 3

    # Filtering settings
    enable_dimensional_filter: bool = True
    filter_mode: str = "post"  # 'pre' or 'post'


@dataclass
class RetrievedChunk:
    """A single retrieved chunk."""
    doc_id: str
    text: str
    score: float
    document_id: Optional[str] = None
    doc_title: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "text": self.text,
            "score": self.score,
            "document_id": self.document_id,
            "doc_title": self.doc_title,
            **self.metadata,
        }


@dataclass
class RetrievalContext:
    """Result of retrieval operation."""
    query: str
    chunks: List[RetrievedChunk]
    total_candidates: int
    filtered_candidates: Optional[int] = None
    hops: int = 1
    retrieval_stats: Dict[str, Any] = field(default_factory=dict)

    def get_context_text(self, max_length: Optional[int] = None, separator: str = "\n\n") -> str:
        """Get concatenated context text for LLM input."""
        texts = []
        total_length = 0

        for chunk in self.chunks:
            text = chunk.text
            if max_length and total_length + len(text) > max_length:
                # Truncate to fit
                remaining = max_length - total_length - len(separator)
                if remaining > 100:  # Only add if meaningful
                    texts.append(text[:remaining] + "...")
                break

            texts.append(text)
            total_length += len(text) + len(separator)

        return separator.join(texts)

    def get_formatted_context(
        self,
        include_titles: bool = True,
        include_scores: bool = False,
        max_length: Optional[int] = None,
    ) -> str:
        """Get formatted context with document markers."""
        formatted_chunks = []
        total_length = 0

        for i, chunk in enumerate(self.chunks, 1):
            # Build chunk header
            header_parts = [f"[Document {i}]"]
            if include_titles and chunk.doc_title:
                header_parts.append(f" {chunk.doc_title}")
            if include_scores:
                header_parts.append(f" (score: {chunk.score:.3f})")

            header = "".join(header_parts)
            chunk_text = f"{header}\n{chunk.text}"

            if max_length and total_length + len(chunk_text) > max_length:
                remaining = max_length - total_length - 10
                if remaining > 100:
                    formatted_chunks.append(chunk_text[:remaining] + "...")
                break

            formatted_chunks.append(chunk_text)
            total_length += len(chunk_text) + 2

        return "\n\n".join(formatted_chunks)

    def get_doc_ids(self) -> List[str]:
        """Get list of doc IDs from chunks."""
        return [c.doc_id for c in self.chunks]

    def get_unique_document_ids(self) -> List[str]:
        """Get unique parent document IDs."""
        return list(set(c.document_id for c in self.chunks if c.document_id))


class RAGRetriever:
    """
    High-level RAG retriever with multi-hop support.

    Wraps STIndex HybridRetriever with additional RAG-specific features.
    """

    def __init__(
        self,
        config: Optional[RAGRetrieverConfig] = None,
        vector_index_path: Optional[str] = None,
        warehouse_path: Optional[str] = None,
    ):
        """
        Initialize RAG retriever.

        Args:
            config: Full configuration object
            vector_index_path: Override vector index path
            warehouse_path: Override warehouse path
        """
        self.config = config or RAGRetrieverConfig()

        # Override paths if provided
        if vector_index_path:
            self.config.vector_index_path = vector_index_path
        if warehouse_path:
            self.config.warehouse_path = warehouse_path

        # Initialize hybrid retriever
        self._init_retriever()

        # Load chunk metadata for doc_id and title
        self._chunk_metadata = None

        logger.info(f"RAGRetriever initialized: vector_index={self.config.vector_index_path}")

    def _init_retriever(self) -> None:
        """Initialize the underlying HybridRetriever."""
        if HybridRetriever is None:
            raise ImportError(
                "stindex.retrieval is required. "
                "Install with: pip install stindex"
            )

        self.hybrid_retriever = HybridRetriever(
            vector_index_path=self.config.vector_index_path,
            stindex_warehouse_path=self.config.warehouse_path,
            encoder_model=self.config.encoder_model,
            device=self.config.device,
        )

    def _load_chunk_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Load chunk metadata for doc_id and title mapping."""
        if self._chunk_metadata is not None:
            return self._chunk_metadata

        self._chunk_metadata = {}

        # Try to load from vector index metadata
        metadata_path = Path(self.config.vector_index_path) / "chunks_metadata.jsonl"
        if metadata_path.exists():
            import jsonlines
            with jsonlines.open(metadata_path, "r") as reader:
                for chunk in reader:
                    doc_id = chunk.get("doc_id", "")

                    # Get dimensions (new unified format)
                    dimensions = chunk.get("dimensions", {})

                    # Build metadata dict
                    meta = {
                        "doc_id": chunk.get("doc_id"),
                        "doc_title": chunk.get("doc_title"),
                        "temporal_normalized": chunk.get("temporal_normalized"),
                        "spatial_text": chunk.get("spatial_text"),
                        "dimensions": dimensions,
                    }

                    # Extract flat labels for backward compatibility
                    # NEW format: dimensions["temporal"] = [["2022", "Q1", "2022-03"]]
                    # OLD format: temporal_labels = ["2022", "Q1", "2022-03"]
                    if dimensions.get("temporal"):
                        meta["temporal_labels"] = dimensions["temporal"][0] if dimensions["temporal"] else []
                    else:
                        meta["temporal_labels"] = chunk.get("temporal_labels", [])

                    if dimensions.get("spatial"):
                        meta["spatial_labels"] = dimensions["spatial"][0] if dimensions["spatial"] else []
                    else:
                        meta["spatial_labels"] = chunk.get("spatial_labels", [])

                    self._chunk_metadata[doc_id] = meta

        return self._chunk_metadata

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        temporal_filter: Optional[Union[TemporalFilter, Dict[str, Any]]] = None,
        spatial_filter: Optional[Union[SpatialFilter, Dict[str, Any]]] = None,
    ) -> RetrievalContext:
        """
        Retrieve relevant chunks for a query.

        Args:
            query: Query text
            k: Number of chunks to retrieve
            temporal_filter: Optional temporal filter
            spatial_filter: Optional spatial filter

        Returns:
            RetrievalContext with retrieved chunks
        """
        k = k or self.config.default_k

        # Use hybrid retriever
        response = self.hybrid_retriever.search(
            query=query,
            k=k,
            temporal_filter=temporal_filter,
            spatial_filter=spatial_filter,
            filter_mode=self.config.filter_mode,
        )

        # Load metadata
        metadata = self._load_chunk_metadata()

        # Convert to RetrievedChunk objects
        chunks = []
        for result in response.results:
            chunk_meta = metadata.get(result.doc_id, {})
            chunks.append(RetrievedChunk(
                doc_id=result.doc_id,
                text=result.text,
                score=result.score,
                document_id=chunk_meta.get("doc_id"),
                doc_title=chunk_meta.get("doc_title"),
                metadata={
                    "temporal_normalized": chunk_meta.get("temporal_normalized"),
                    "spatial_text": chunk_meta.get("spatial_text"),
                    "temporal_labels": chunk_meta.get("temporal_labels", []),
                    "spatial_labels": chunk_meta.get("spatial_labels", []),
                    "dimensions": chunk_meta.get("dimensions", {}),
                },
            ))

        return RetrievalContext(
            query=query,
            chunks=chunks,
            total_candidates=response.total_candidates,
            filtered_candidates=response.filtered_candidates,
            hops=1,
            retrieval_stats={
                "filter_stats": response.filter_stats,
            },
        )

    def retrieve_multihop(
        self,
        question: str,
        max_hops: Optional[int] = None,
        documents_per_hop: Optional[int] = None,
        temporal_filter: Optional[Union[TemporalFilter, Dict[str, Any]]] = None,
        spatial_filter: Optional[Union[SpatialFilter, Dict[str, Any]]] = None,
    ) -> RetrievalContext:
        """
        Multi-hop retrieval for complex questions.

        Iteratively retrieves documents, extracts entities, and
        queries for related documents.

        Args:
            question: Complex question requiring multi-hop reasoning
            max_hops: Maximum retrieval hops
            documents_per_hop: Documents to retrieve per hop
            temporal_filter: Optional temporal filter
            spatial_filter: Optional spatial filter

        Returns:
            RetrievalContext with all retrieved chunks
        """
        max_hops = max_hops or self.config.max_hops
        documents_per_hop = documents_per_hop or self.config.documents_per_hop

        all_chunks = []
        all_doc_ids = set()
        queries = [question]

        for hop in range(max_hops):
            hop_chunks = []

            for query in queries:
                # Retrieve for this query
                context = self.retrieve(
                    query=query,
                    k=documents_per_hop,
                    temporal_filter=temporal_filter,
                    spatial_filter=spatial_filter,
                )

                # Add new chunks (deduplication)
                for chunk in context.chunks:
                    if chunk.doc_id not in all_doc_ids:
                        all_doc_ids.add(chunk.doc_id)
                        all_chunks.append(chunk)
                        hop_chunks.append(chunk)

            if not hop_chunks:
                break  # No new information

            # Generate follow-up queries from retrieved content
            # Simple heuristic: extract titles as potential queries
            if hop < max_hops - 1:
                queries = self._generate_followup_queries(question, hop_chunks)
                if not queries:
                    break

        return RetrievalContext(
            query=question,
            chunks=all_chunks,
            total_candidates=self.hybrid_retriever.faiss_index.ntotal,
            hops=hop + 1,
            retrieval_stats={
                "num_hops": hop + 1,
                "unique_chunks": len(all_chunks),
            },
        )

    def _generate_followup_queries(
        self,
        original_question: str,
        chunks: List[RetrievedChunk],
    ) -> List[str]:
        """
        Generate follow-up queries from retrieved chunks.

        Simple heuristic: use document titles as potential queries.
        For more sophisticated query generation, use LLM.
        """
        queries = []

        # Extract unique titles
        seen_titles = set()
        for chunk in chunks:
            if chunk.doc_title and chunk.doc_title not in seen_titles:
                seen_titles.add(chunk.doc_title)
                queries.append(chunk.doc_title)

        # Limit number of follow-up queries
        return queries[:3]

    def retrieve_with_context(
        self,
        query: str,
        k: Optional[int] = None,
        max_context_length: Optional[int] = None,
    ) -> Tuple[str, RetrievalContext]:
        """
        Retrieve and return formatted context string.

        Convenience method for simple RAG pipelines.

        Args:
            query: Query text
            k: Number of chunks to retrieve
            max_context_length: Max context length in characters

        Returns:
            Tuple of (formatted_context, RetrievalContext)
        """
        context = self.retrieve(query, k)
        max_length = max_context_length or self.config.max_context_length

        formatted = context.get_formatted_context(
            include_titles=self.config.include_metadata,
            max_length=max_length,
        )

        return formatted, context

    def get_stats(self) -> Dict[str, Any]:
        """Get retriever statistics."""
        stats = self.hybrid_retriever.get_stats()
        stats.update({
            "config": {
                "default_k": self.config.default_k,
                "max_context_length": self.config.max_context_length,
                "max_hops": self.config.max_hops,
                "filter_mode": self.config.filter_mode,
            }
        })
        return stats


class SimpleRetriever:
    """
    Simple retriever without STIndex HybridRetriever dependency.

    Uses FAISS directly for basic vector search.
    """

    def __init__(
        self,
        vector_index_path: str,
        encoder_model: str = "BAAI/bge-m3",
        device: str = "cpu",
    ):
        """Initialize simple retriever."""
        self.vector_index_path = Path(vector_index_path)
        self.device = device

        # Load components
        self._load_index()
        self._init_encoder(encoder_model)

    def _load_index(self) -> None:
        """Load FAISS index and metadata."""
        import faiss

        # Load index
        index_path = self.vector_index_path / "faiss_index.bin"
        self.index = faiss.read_index(str(index_path))

        # Load ID mapping
        id_mapping_path = self.vector_index_path / "id_mapping.json"
        with open(id_mapping_path, "r") as f:
            self.id_mapping = json.load(f)

        # Load texts
        self.texts = {}
        metadata_path = self.vector_index_path / "chunks_metadata.jsonl"
        if metadata_path.exists():
            import jsonlines
            with jsonlines.open(metadata_path, "r") as reader:
                for chunk in reader:
                    doc_id = chunk.get("doc_id", chunk.get("chunk_id", ""))
                    self.texts[doc_id] = chunk.get("text", "")

    def _init_encoder(self, model_name: str) -> None:
        """Initialize encoder."""
        from sentence_transformers import SentenceTransformer
        self.encoder = SentenceTransformer(model_name, device=self.device)

    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve top-k chunks for query."""
        import numpy as np

        # Encode query
        query_embedding = self.encoder.encode(
            query,
            normalize_embeddings=True,
        ).astype(np.float32).reshape(1, -1)

        # Search
        scores, indices = self.index.search(query_embedding, k)

        # Build results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            doc_id = self.id_mapping[idx]
            results.append({
                "doc_id": doc_id,
                "text": self.texts.get(doc_id, ""),
                "score": float(score),
            })

        return results
