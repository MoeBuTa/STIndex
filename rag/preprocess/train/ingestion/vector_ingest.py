#!/usr/bin/env python3
"""
FAISS CPU Vector Ingestion Module.

This module encodes corpus documents into dense vectors using a sentence transformer
and indexes them in FAISS for efficient similarity search.

Optimized for:
- CPU-only inference (no GPU required)
- Closed-domain corpus (pre-processed multi-hop QA datasets)
- Memory-efficient batch processing with checkpointing

Usage:
    python -m rag.preprocess.train.ingestion.vector_ingest \
        --input data/corpus/documents.jsonl \
        --output data/vector/rag \
        --config rag/cfg/ingestion.yaml
"""

import argparse
import json
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import jsonlines
import numpy as np
from loguru import logger
from tqdm import tqdm


class VectorIngester:
    """
    Ingest corpus documents into FAISS vector index.

    Supports:
    - Multiple encoder models (BGE-M3, all-MiniLM, etc.)
    - CPU-optimized encoding with batching
    - IVF index for efficient approximate search
    - Flat index for exact search (smaller corpora)
    - Checkpointing for resumable ingestion
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        model_name: str = "BAAI/bge-m3",
        output_dir: str = "data/vector/rag",
        batch_size: int = 32,
        use_gpu: bool = False,
        index_type: str = "flat",
        normalize_embeddings: bool = True,
    ):
        """
        Initialize vector ingester.

        Args:
            config_path: Path to ingestion config YAML
            model_name: HuggingFace model name for encoding
            output_dir: Output directory for index and metadata
            batch_size: Encoding batch size (smaller for CPU)
            use_gpu: Use GPU for encoding (default False for CPU)
            index_type: FAISS index type: 'flat', 'ivf', 'hnsw'
            normalize_embeddings: Normalize embeddings to unit length
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load config if provided
        if config_path:
            self._load_config(config_path)
        else:
            self.model_name = model_name
            self.batch_size = batch_size
            self.use_gpu = use_gpu
            self.index_type = index_type
            self.normalize_embeddings = normalize_embeddings

        # Initialize encoder
        self.encoder = None
        self.embedding_dim = None

        # Statistics
        self.stats = {
            "total_documents": 0,
            "encoded_documents": 0,
            "failed_documents": 0,
            "index_type": self.index_type,
            "model_name": self.model_name,
            "start_time": None,
            "end_time": None,
        }

    def _load_config(self, config_path: str) -> None:
        """Load configuration from YAML file."""
        import yaml

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        encoder_cfg = config.get("encoder", {})
        index_cfg = config.get("index", {})

        self.model_name = encoder_cfg.get("model_name", "BAAI/bge-m3")
        self.batch_size = encoder_cfg.get("batch_size", 32)
        self.use_gpu = encoder_cfg.get("device", "cpu") == "cuda"
        self.normalize_embeddings = encoder_cfg.get("normalize_embeddings", True)
        self.max_length = encoder_cfg.get("max_length", 512)

        self.index_type = index_cfg.get("index_type", "flat")
        self.nlist = index_cfg.get("ivf", {}).get("nlist", 100)
        self.nprobe = index_cfg.get("ivf", {}).get("nprobe", 10)

    def _init_encoder(self) -> None:
        """Initialize sentence transformer encoder."""
        if self.encoder is not None:
            return

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers is required: pip install sentence-transformers"
            )

        logger.info(f"Loading encoder model: {self.model_name}")

        device = "cuda" if self.use_gpu else "cpu"
        self.encoder = SentenceTransformer(self.model_name, device=device)
        self.embedding_dim = self.encoder.get_sentence_embedding_dimension()

        logger.info(f"Encoder loaded. Embedding dimension: {self.embedding_dim}")

    def _init_index(self, dimension: int, num_vectors: Optional[int] = None) -> Any:
        """Initialize FAISS index."""
        try:
            import faiss
        except ImportError:
            raise ImportError("faiss-cpu is required: pip install faiss-cpu")

        logger.info(f"Creating FAISS {self.index_type} index with dim={dimension}")

        if self.index_type == "flat":
            # Exact search (best for small corpora < 100K)
            index = faiss.IndexFlatIP(dimension)  # Inner product for normalized vectors

        elif self.index_type == "ivf":
            # Approximate search with clustering (good for 100K-10M)
            nlist = min(self.nlist, num_vectors // 40 + 1) if num_vectors else self.nlist
            quantizer = faiss.IndexFlatIP(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)

        elif self.index_type == "hnsw":
            # HNSW graph-based search
            index = faiss.IndexHNSWFlat(dimension, 32, faiss.METRIC_INNER_PRODUCT)

        else:
            raise ValueError(f"Unknown index type: {self.index_type}")

        return index

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings."""
        self._init_encoder()

        embeddings = self.encoder.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=self.normalize_embeddings,
        )

        return embeddings.astype(np.float32)

    def load_documents(
        self,
        input_path: str,
        limit: Optional[int] = None,
        text_field: str = "contents",
        id_field: str = "doc_id",
    ) -> Tuple[List[str], List[str], List[Dict]]:
        """
        Load documents from JSONL file.

        Returns:
            Tuple of (texts, ids, metadata)
        """
        texts = []
        ids = []
        metadata = []

        with jsonlines.open(input_path, "r") as reader:
            for i, doc in enumerate(reader):
                if limit and i >= limit:
                    break

                text = doc.get(text_field, "")
                if not text:
                    continue

                texts.append(text)
                ids.append(doc.get(id_field, f"doc_{i}"))
                metadata.append({
                    k: v for k, v in doc.items()
                    # Include all fields including text for retrieval
                })

        return texts, ids, metadata

    def ingest(
        self,
        input_path: str,
        limit: Optional[int] = None,
        text_field: str = "contents",
        id_field: str = "doc_id",
        checkpoint_interval: int = 10000,
        resume: bool = True,
    ) -> Dict[str, Any]:
        """
        Ingest corpus into FAISS index.

        Args:
            input_path: Path to input documents JSONL
            limit: Limit number of documents to process
            text_field: Field name for text content
            id_field: Field name for document ID
            checkpoint_interval: Save checkpoint every N documents
            resume: Resume from checkpoint if available

        Returns:
            Ingestion statistics
        """
        self.stats["start_time"] = datetime.now().isoformat()

        # Load documents
        logger.info(f"Loading documents from {input_path}")
        texts, ids, metadata = self.load_documents(input_path, limit, text_field, id_field)
        self.stats["total_documents"] = len(texts)
        logger.info(f"Loaded {len(texts)} documents")

        if len(texts) == 0:
            logger.warning("No documents to ingest")
            return self.stats

        # Check for checkpoint
        checkpoint_path = self.output_dir / "checkpoint.pkl"
        start_idx = 0
        embeddings_list = []

        if resume and checkpoint_path.exists():
            logger.info("Loading checkpoint...")
            with open(checkpoint_path, "rb") as f:
                checkpoint = pickle.load(f)
            start_idx = checkpoint["processed"]
            embeddings_list = checkpoint["embeddings"]
            logger.info(f"Resumed from checkpoint: {start_idx} chunks already encoded")

        # Initialize encoder
        self._init_encoder()

        # Encode in batches
        logger.info(f"Encoding documents (batch_size={self.batch_size})...")
        for i in tqdm(range(start_idx, len(texts), self.batch_size), desc="Encoding"):
            batch_texts = texts[i:i + self.batch_size]

            try:
                batch_embeddings = self.encode_texts(batch_texts)
                embeddings_list.append(batch_embeddings)
                self.stats["encoded_documents"] += len(batch_texts)
            except Exception as e:
                logger.warning(f"Encoding failed for batch {i}: {e}")
                self.stats["failed_documents"] += len(batch_texts)
                continue

            # Save checkpoint
            if (i + self.batch_size) % checkpoint_interval == 0:
                logger.info(f"Saving checkpoint at {i + self.batch_size}")
                with open(checkpoint_path, "wb") as f:
                    pickle.dump({
                        "processed": i + self.batch_size,
                        "embeddings": embeddings_list,
                    }, f)

        # Concatenate all embeddings
        logger.info("Concatenating embeddings...")
        all_embeddings = np.vstack(embeddings_list)

        # Create and populate index
        logger.info(f"Building FAISS index ({self.index_type})...")
        index = self._init_index(self.embedding_dim, len(all_embeddings))

        # Train index if needed (IVF)
        if self.index_type == "ivf" and not index.is_trained:
            logger.info("Training IVF index...")
            index.train(all_embeddings)

        # Add vectors
        logger.info("Adding vectors to index...")
        index.add(all_embeddings)

        # Save index
        self._save_index(index, ids, metadata)

        # Clean up checkpoint
        if checkpoint_path.exists():
            checkpoint_path.unlink()

        self.stats["end_time"] = datetime.now().isoformat()

        # Save stats
        stats_path = self.output_dir / "ingestion_stats.json"
        with open(stats_path, "w") as f:
            json.dump(self.stats, f, indent=2)

        logger.success(f"Vector ingestion complete! {self.stats['encoded_documents']} documents indexed")
        return self.stats

    def _save_index(
        self,
        index: Any,
        ids: List[str],
        metadata: List[Dict],
    ) -> None:
        """Save FAISS index and metadata."""
        import faiss

        # Save FAISS index
        index_path = self.output_dir / "faiss_index.bin"
        faiss.write_index(index, str(index_path))
        logger.info(f"Saved FAISS index to {index_path}")

        # Save ID mapping (int index -> doc_id)
        id_mapping_path = self.output_dir / "id_mapping.json"
        with open(id_mapping_path, "w") as f:
            json.dump(ids, f)
        logger.info(f"Saved ID mapping to {id_mapping_path}")

        # Save metadata
        metadata_path = self.output_dir / "chunks_metadata.jsonl"
        with jsonlines.open(metadata_path, "w") as writer:
            for i, meta in enumerate(metadata):
                meta["faiss_idx"] = i
                meta["doc_id"] = ids[i]
                writer.write(meta)
        logger.info(f"Saved metadata to {metadata_path}")

        # Save index config
        config_path = self.output_dir / "index_config.json"
        config = {
            "index_type": self.index_type,
            "embedding_dim": self.embedding_dim,
            "model_name": self.model_name,
            "normalize_embeddings": self.normalize_embeddings,
            "num_vectors": len(ids),
            "created_at": datetime.now().isoformat(),
        }
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        logger.info(f"Saved index config to {config_path}")


class VectorIndexLoader:
    """
    Load FAISS index for retrieval.

    Usage:
        loader = VectorIndexLoader("data/vector/rag")
        results = loader.search("What is the capital of France?", k=10)
    """

    def __init__(self, index_dir: str):
        """
        Load FAISS index and metadata.

        Args:
            index_dir: Directory containing index files
        """
        self.index_dir = Path(index_dir)

        # Load config
        config_path = self.index_dir / "index_config.json"
        with open(config_path, "r") as f:
            self.config = json.load(f)

        # Load FAISS index
        self._load_index()

        # Load ID mapping
        self._load_id_mapping()

        # Initialize encoder (lazy)
        self.encoder = None

    def _load_index(self) -> None:
        """Load FAISS index from disk."""
        import faiss

        index_path = self.index_dir / "faiss_index.bin"
        self.index = faiss.read_index(str(index_path))
        logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")

    def _load_id_mapping(self) -> None:
        """Load ID mapping."""
        id_mapping_path = self.index_dir / "id_mapping.json"
        with open(id_mapping_path, "r") as f:
            self.id_mapping = json.load(f)

    def _init_encoder(self) -> None:
        """Initialize encoder for query encoding."""
        if self.encoder is not None:
            return

        from sentence_transformers import SentenceTransformer

        self.encoder = SentenceTransformer(
            self.config["model_name"],
            device="cpu"
        )

    def encode_query(self, query: str) -> np.ndarray:
        """Encode query to embedding."""
        self._init_encoder()

        embedding = self.encoder.encode(
            query,
            normalize_embeddings=self.config.get("normalize_embeddings", True),
        )

        return embedding.astype(np.float32).reshape(1, -1)

    def search(
        self,
        query: str,
        k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks.

        Args:
            query: Query text
            k: Number of results to return

        Returns:
            List of results with doc_id and score
        """
        query_embedding = self.encode_query(query)

        # Search FAISS
        scores, indices = self.index.search(query_embedding, k)

        # Map indices to doc IDs
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:  # FAISS returns -1 for not found
                continue

            results.append({
                "doc_id": self.id_mapping[idx],
                "score": float(score),
                "faiss_idx": int(idx),
            })

        return results

    def search_batch(
        self,
        queries: List[str],
        k: int = 10,
    ) -> List[List[Dict[str, Any]]]:
        """
        Search for multiple queries in batch.

        Args:
            queries: List of query texts
            k: Number of results per query

        Returns:
            List of results for each query
        """
        self._init_encoder()

        # Encode all queries
        query_embeddings = self.encoder.encode(
            queries,
            normalize_embeddings=self.config.get("normalize_embeddings", True),
        ).astype(np.float32)

        # Batch search
        scores, indices = self.index.search(query_embeddings, k)

        # Map to results
        all_results = []
        for query_scores, query_indices in zip(scores, indices):
            results = []
            for score, idx in zip(query_scores, query_indices):
                if idx < 0:
                    continue
                results.append({
                    "doc_id": self.id_mapping[idx],
                    "score": float(score),
                    "faiss_idx": int(idx),
                })
            all_results.append(results)

        return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Ingest corpus into FAISS vector index"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/corpus/documents.jsonl",
        help="Input documents JSONL file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/vector/rag",
        help="Output directory for index",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Config file path (optional)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="BAAI/bge-m3",
        help="Encoder model name",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Encoding batch size (smaller for CPU)",
    )
    parser.add_argument(
        "--index-type",
        type=str,
        default="ivf",
        choices=["flat", "ivf", "hnsw"],
        help="FAISS index type",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of documents to process",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=10000,
        help="Checkpoint interval",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Don't resume from checkpoint",
    )

    args = parser.parse_args()

    # Create ingester
    if args.config:
        ingester = VectorIngester(config_path=args.config)
    else:
        ingester = VectorIngester(
            model_name=args.model,
            output_dir=args.output,
            batch_size=args.batch_size,
            index_type=args.index_type,
            use_gpu=False,  # CPU by default
        )

    # Override output dir if specified
    if args.output:
        ingester.output_dir = Path(args.output)
        ingester.output_dir.mkdir(parents=True, exist_ok=True)

    # Run ingestion
    stats = ingester.ingest(
        input_path=args.input,
        limit=args.limit,
        checkpoint_interval=args.checkpoint_interval,
        resume=not args.no_resume,
    )

    print(f"\n=== Vector Ingestion Complete ===")
    print(f"Total documents: {stats['total_documents']}")
    print(f"Encoded: {stats['encoded_documents']}")
    print(f"Failed: {stats['failed_documents']}")
    print(f"Index type: {stats['index_type']}")
    print(f"Model: {stats['model_name']}")


if __name__ == "__main__":
    main()
