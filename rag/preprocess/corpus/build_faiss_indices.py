"""
Build FAISS indices from computed embeddings for RRF-4 retrieval.

Creates FAISS indices for:
- Contriever (inner product)
- SPECTER (L2 norm)
- MedCPT (inner product)

Usage:
    python -m rag.preprocess.corpus.build_faiss_indices \\
        --embeddings data/embeddings/medcorp \\
        --output data/indices/medcorp
"""

import argparse
import json
import numpy as np
from pathlib import Path

import faiss
from loguru import logger


def build_faiss_index(
    embeddings: np.ndarray,
    metric: str = "inner_product",
    use_hnsw: bool = False,
    hnsw_m: int = 32
) -> faiss.Index:
    """
    Build FAISS index from embeddings.

    Args:
        embeddings: Embedding matrix (num_docs x embedding_dim)
        metric: Distance metric ('inner_product' or 'l2')
        use_hnsw: Whether to use HNSW for approximate search
        hnsw_m: HNSW graph connectivity parameter

    Returns:
        FAISS index
    """
    num_docs, embedding_dim = embeddings.shape
    logger.info(f"Building FAISS index:")
    logger.info(f"  Documents: {num_docs:,}")
    logger.info(f"  Embedding dim: {embedding_dim}")
    logger.info(f"  Metric: {metric}")
    logger.info(f"  HNSW: {use_hnsw}")

    # Ensure float32 for FAISS
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)

    # Create index based on metric
    if metric == "inner_product":
        if use_hnsw:
            # HNSW with inner product
            index = faiss.IndexHNSWFlat(embedding_dim, hnsw_m, faiss.METRIC_INNER_PRODUCT)
        else:
            # Flat index with inner product
            index = faiss.IndexFlatIP(embedding_dim)
    elif metric == "l2":
        if use_hnsw:
            # HNSW with L2
            index = faiss.IndexHNSWFlat(embedding_dim, hnsw_m, faiss.METRIC_L2)
        else:
            # Flat index with L2
            index = faiss.IndexFlatL2(embedding_dim)
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    # Add vectors to index
    logger.info("Adding vectors to index...")
    index.add(embeddings)
    logger.info(f"✓ Index built: {index.ntotal} vectors")

    return index


def main():
    parser = argparse.ArgumentParser(
        description="Build FAISS indices from computed embeddings"
    )
    parser.add_argument(
        "--embeddings",
        type=str,
        required=True,
        help="Directory containing embeddings (output of compute_embeddings.py)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for FAISS indices"
    )
    parser.add_argument(
        "--use-hnsw",
        action="store_true",
        help="Use HNSW for approximate nearest neighbor search"
    )
    parser.add_argument(
        "--hnsw-m",
        type=int,
        default=32,
        help="HNSW graph connectivity parameter"
    )
    args = parser.parse_args()

    embeddings_dir = Path(args.embeddings)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("Building FAISS Indices for RRF-4")
    logger.info("=" * 80)
    logger.info(f"Embeddings: {embeddings_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"HNSW: {args.use_hnsw}")
    logger.info("=" * 80)

    # Find all embedding files
    retriever_types = []
    for config_file in embeddings_dir.glob("*_config.json"):
        retriever_type = config_file.stem.replace("_config", "")
        retriever_types.append(retriever_type)

    logger.info(f"Found embeddings for: {', '.join(retriever_types)}")

    # Build index for each retriever
    for retriever_type in retriever_types:
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"Building index for: {retriever_type}")
        logger.info("=" * 80)

        # Load config
        config_file = embeddings_dir / f"{retriever_type}_config.json"
        with open(config_file, "r") as f:
            config = json.load(f)

        metric = config["metric"]
        logger.info(f"Config: {config}")

        # Load embeddings
        embeddings_file = embeddings_dir / f"{retriever_type}_embeddings.npy"
        logger.info(f"Loading embeddings from {embeddings_file}")
        embeddings = np.load(embeddings_file)
        logger.info(f"  Loaded shape: {embeddings.shape}")

        # Build FAISS index
        index = build_faiss_index(
            embeddings,
            metric=metric,
            use_hnsw=args.use_hnsw,
            hnsw_m=args.hnsw_m
        )

        # Save index
        index_file = output_dir / f"{retriever_type}_index.faiss"
        logger.info(f"Saving index to {index_file}")
        faiss.write_index(index, str(index_file))
        logger.info(f"✓ Saved: {index_file}")
        logger.info(f"  Size: {index_file.stat().st_size / 1024 / 1024:.1f} MB")

        # Copy metadata and config to output dir
        import shutil
        metadata_src = embeddings_dir / f"{retriever_type}_metadata.jsonl"
        metadata_dst = output_dir / f"{retriever_type}_metadata.jsonl"
        shutil.copy(metadata_src, metadata_dst)
        logger.info(f"✓ Copied metadata: {metadata_dst}")

        config_dst = output_dir / f"{retriever_type}_config.json"
        shutil.copy(config_file, config_dst)
        logger.info(f"✓ Copied config: {config_dst}")

    logger.info("")
    logger.info("=" * 80)
    logger.info("✓ All FAISS indices built successfully!")
    logger.info("=" * 80)
    logger.info(f"Output directory: {output_dir}")
    logger.info("")
    logger.info("Index files:")
    for retriever_type in retriever_types:
        index_file = output_dir / f"{retriever_type}_index.faiss"
        logger.info(f"  - {index_file.name}")
    logger.info("")
    logger.info("Next step: Build BM25 index and test RRF-4 retriever")


if __name__ == "__main__":
    main()
