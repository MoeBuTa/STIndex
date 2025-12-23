#!/usr/bin/env python3
"""
Build BGE-M3 indices for MedCorp.

Creates:
1. FAISS IVF index with BGE-M3 dense embeddings (1024-dim)
2. Sparse weights matrix (CSR format) for BGE-M3 lexical matching
3. Passages metadata file

Usage:
    # Full build (requires GPU, ~2-4 hours for 483K docs)
    python -m scripts.rag.build_bgem3_index

    # Test mode (100 docs)
    python -m scripts.rag.build_bgem3_index --test --limit 100

    # Resume from checkpoint
    python -m scripts.rag.build_bgem3_index --resume

    # Merge checkpoints only
    python -m scripts.rag.build_bgem3_index --merge-only
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def load_corpus(corpus_path: Path, limit: Optional[int] = None) -> List[Dict]:
    """Load corpus from JSONL file."""
    documents = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            if line.strip():
                documents.append(json.loads(line))
    return documents


def init_encoder(device: str = "cuda", use_fp16: bool = True):
    """Initialize BGE-M3 encoder."""
    try:
        from FlagEmbedding import BGEM3FlagModel
    except ImportError:
        logger.error("FlagEmbedding not installed. Install with: pip install FlagEmbedding")
        sys.exit(1)

    logger.info(f"Loading BGE-M3 encoder on {device}...")
    model = BGEM3FlagModel(
        "BAAI/bge-m3",
        use_fp16=use_fp16,
        device=device,
    )
    logger.info("BGE-M3 encoder loaded")
    return model


def encode_batch(
    model,
    texts: List[str],
    batch_size: int = 64,
    max_length: int = 512,
) -> Tuple[np.ndarray, List[Dict]]:
    """
    Encode texts with BGE-M3 to get dense and sparse representations.

    Returns:
        dense_embeddings: np.ndarray of shape (n_texts, 1024)
        sparse_weights: List of dicts mapping token_id -> weight
    """
    # BGE-M3 encode returns dict with 'dense_vecs' and 'lexical_weights'
    outputs = model.encode(
        texts,
        batch_size=batch_size,
        max_length=max_length,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )

    dense_embeddings = outputs["dense_vecs"]
    sparse_weights = outputs["lexical_weights"]

    return dense_embeddings, sparse_weights


def build_faiss_index(
    embeddings: np.ndarray,
    index_type: str = "ivf",
    nlist: int = 4096,
    use_gpu: bool = True,
    gpu_id: int = 0,
):
    """
    Build FAISS index from embeddings.

    Args:
        embeddings: Dense embeddings (n_docs, dim)
        index_type: "flat" or "ivf"
        nlist: Number of clusters for IVF
        use_gpu: Whether to use GPU for training
        gpu_id: GPU device ID

    Returns:
        FAISS index
    """
    import faiss

    n_docs, dim = embeddings.shape
    logger.info(f"Building FAISS {index_type} index: {n_docs:,} docs, {dim}-dim")

    if index_type == "flat":
        # Exact search (slower but accurate)
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings.astype(np.float32))
    elif index_type == "ivf":
        # IVF index for fast approximate search
        # Adjust nlist based on corpus size
        nlist = min(nlist, n_docs // 39)  # Rule of thumb: sqrt(n_docs)
        nlist = max(nlist, 1)

        logger.info(f"  nlist={nlist}")

        # Create quantizer and index
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)

        # Train on GPU if available
        if use_gpu and faiss.get_num_gpus() > 0:
            logger.info(f"  Training on GPU {gpu_id}...")
            res = faiss.StandardGpuResources()
            gpu_index = faiss.index_cpu_to_gpu(res, gpu_id, index)

            # Train on sample
            sample_size = min(n_docs, 100000)
            sample_indices = np.random.choice(n_docs, sample_size, replace=False)
            train_data = embeddings[sample_indices].astype(np.float32)
            gpu_index.train(train_data)

            # Copy trained index back to CPU
            index = faiss.index_gpu_to_cpu(gpu_index)
        else:
            logger.info("  Training on CPU...")
            sample_size = min(n_docs, 100000)
            sample_indices = np.random.choice(n_docs, sample_size, replace=False)
            train_data = embeddings[sample_indices].astype(np.float32)
            index.train(train_data)

        # Add all vectors
        logger.info("  Adding vectors to index...")
        index.add(embeddings.astype(np.float32))

        # Set default nprobe
        index.nprobe = min(256, nlist)
    else:
        raise ValueError(f"Unknown index type: {index_type}")

    logger.info(f"  Index built: {index.ntotal:,} vectors")
    return index


def sparse_weights_to_csr(
    sparse_weights: List[Dict[int, float]],
    vocab_size: int = 250002,  # BGE-M3 tokenizer vocab size
) -> "scipy.sparse.csr_matrix":
    """
    Convert list of sparse weight dicts to CSR matrix.

    Args:
        sparse_weights: List of dicts mapping token_id -> weight
        vocab_size: Vocabulary size

    Returns:
        CSR matrix of shape (n_docs, vocab_size)
    """
    import scipy.sparse as sp

    n_docs = len(sparse_weights)
    logger.info(f"Converting {n_docs:,} sparse vectors to CSR matrix...")

    # Build COO format first (faster construction)
    rows = []
    cols = []
    data = []

    for doc_idx, weights in enumerate(tqdm(sparse_weights, desc="Building sparse matrix")):
        for token_id, weight in weights.items():
            rows.append(doc_idx)
            cols.append(token_id)
            data.append(weight)

    # Create COO and convert to CSR
    coo = sp.coo_matrix(
        (data, (rows, cols)),
        shape=(n_docs, vocab_size),
        dtype=np.float32,
    )
    csr = coo.tocsr()

    logger.info(f"  CSR matrix: {csr.shape}, nnz={csr.nnz:,}")
    return csr


def convert_sparse_to_serializable(sparse_weights: List[Dict]) -> List[Dict]:
    """Convert sparse weights with float16/numpy values to JSON-serializable format."""
    serializable = []
    for weights in sparse_weights:
        converted = {}
        for token_id, weight in weights.items():
            # Convert numpy/float16 to Python float
            if hasattr(weight, 'item'):
                weight = weight.item()
            converted[int(token_id)] = float(weight)
        serializable.append(converted)
    return serializable


def save_checkpoint(
    checkpoint_dir: Path,
    checkpoint_idx: int,
    dense_embeddings: np.ndarray,
    sparse_weights: List[Dict],
    doc_ids: List[str],
):
    """Save encoding checkpoint."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    np.save(checkpoint_dir / f"dense_{checkpoint_idx}.npy", dense_embeddings)

    # Convert float16 to float for JSON serialization
    serializable_sparse = convert_sparse_to_serializable(sparse_weights)
    with open(checkpoint_dir / f"sparse_{checkpoint_idx}.json", "w") as f:
        json.dump(serializable_sparse, f)

    with open(checkpoint_dir / f"doc_ids_{checkpoint_idx}.json", "w") as f:
        json.dump(doc_ids, f)

    logger.info(f"  Saved checkpoint {checkpoint_idx}")


def load_checkpoints(checkpoint_dir: Path) -> Tuple[np.ndarray, List[Dict], List[str]]:
    """Load and merge all checkpoints."""
    dense_files = sorted(checkpoint_dir.glob("dense_*.npy"))
    sparse_files = sorted(checkpoint_dir.glob("sparse_*.json"))
    doc_id_files = sorted(checkpoint_dir.glob("doc_ids_*.json"))

    if not dense_files:
        return None, None, None

    logger.info(f"Loading {len(dense_files)} checkpoints...")

    all_dense = []
    all_sparse = []
    all_doc_ids = []

    for dense_f, sparse_f, doc_id_f in zip(dense_files, sparse_files, doc_id_files):
        all_dense.append(np.load(dense_f))
        with open(sparse_f) as f:
            all_sparse.extend(json.load(f))
        with open(doc_id_f) as f:
            all_doc_ids.extend(json.load(f))

    dense_embeddings = np.vstack(all_dense)
    logger.info(f"  Loaded {dense_embeddings.shape[0]:,} embeddings")

    return dense_embeddings, all_sparse, all_doc_ids


def main():
    parser = argparse.ArgumentParser(description="Build BGE-M3 indices for MedCorp")
    parser.add_argument(
        "--corpus",
        default="data/original/medcorp/train.jsonl",
        help="Path to corpus JSONL file",
    )
    parser.add_argument(
        "--output-dir",
        default="data/indices/medcorp_bgem3",
        help="Output directory for indices",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Encoding batch size",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=50000,
        help="Save checkpoint every N documents",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of documents (for testing)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode (limit to 100 docs)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoints",
    )
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Only merge existing checkpoints",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device for encoding (cuda/cpu)",
    )
    parser.add_argument(
        "--index-type",
        default="ivf",
        choices=["flat", "ivf"],
        help="FAISS index type",
    )
    parser.add_argument(
        "--nlist",
        type=int,
        default=4096,
        help="Number of IVF clusters",
    )

    args = parser.parse_args()

    if args.test:
        args.limit = args.limit or 100

    corpus_path = Path(args.corpus)
    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("BGE-M3 Index Builder")
    logger.info("=" * 60)
    logger.info(f"Corpus: {corpus_path}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Device: {args.device}")
    if args.limit:
        logger.info(f"Limit: {args.limit} documents")

    # Check if we're just merging
    if args.merge_only:
        dense_embeddings, sparse_weights, doc_ids = load_checkpoints(checkpoint_dir)
        if dense_embeddings is None:
            logger.error("No checkpoints found to merge")
            return 1
    else:
        # Load corpus
        logger.info("\n1. Loading corpus...")
        documents = load_corpus(corpus_path, limit=args.limit)
        logger.info(f"   Loaded {len(documents):,} documents")

        # Check for existing checkpoints
        start_idx = 0
        all_dense = []
        all_sparse = []
        all_doc_ids = []

        if args.resume and checkpoint_dir.exists():
            dense_embeddings, sparse_weights, doc_ids = load_checkpoints(checkpoint_dir)
            if dense_embeddings is not None:
                all_dense.append(dense_embeddings)
                all_sparse.extend(sparse_weights)
                all_doc_ids.extend(doc_ids)
                start_idx = len(doc_ids)
                logger.info(f"   Resuming from document {start_idx:,}")

        if start_idx < len(documents):
            # Initialize encoder
            logger.info("\n2. Initializing BGE-M3 encoder...")
            model = init_encoder(device=args.device)

            # Encode documents
            logger.info("\n3. Encoding documents...")

            checkpoint_dense = []
            checkpoint_sparse = []
            checkpoint_doc_ids = []
            checkpoint_idx = start_idx // args.checkpoint_interval

            for i in tqdm(range(start_idx, len(documents), args.batch_size), desc="Encoding"):
                batch_docs = documents[i : i + args.batch_size]

                # Prepare texts: title + contents
                texts = [
                    f"{doc.get('title', '')}. {doc.get('contents', '')}"
                    for doc in batch_docs
                ]

                # Encode batch
                dense, sparse = encode_batch(model, texts, batch_size=len(texts))

                checkpoint_dense.append(dense)
                checkpoint_sparse.extend(sparse)
                checkpoint_doc_ids.extend([doc.get("doc_id", str(i + j)) for j, doc in enumerate(batch_docs)])

                # Save checkpoint
                if len(checkpoint_doc_ids) >= args.checkpoint_interval:
                    dense_array = np.vstack(checkpoint_dense)
                    save_checkpoint(
                        checkpoint_dir,
                        checkpoint_idx,
                        dense_array,
                        checkpoint_sparse,
                        checkpoint_doc_ids,
                    )
                    all_dense.append(dense_array)
                    all_sparse.extend(checkpoint_sparse)
                    all_doc_ids.extend(checkpoint_doc_ids)

                    checkpoint_dense = []
                    checkpoint_sparse = []
                    checkpoint_doc_ids = []
                    checkpoint_idx += 1

            # Save remaining
            if checkpoint_dense:
                dense_array = np.vstack(checkpoint_dense)
                save_checkpoint(
                    checkpoint_dir,
                    checkpoint_idx,
                    dense_array,
                    checkpoint_sparse,
                    checkpoint_doc_ids,
                )
                all_dense.append(dense_array)
                all_sparse.extend(checkpoint_sparse)
                all_doc_ids.extend(checkpoint_doc_ids)

        # Merge all embeddings
        dense_embeddings = np.vstack(all_dense) if all_dense else None
        sparse_weights = all_sparse
        doc_ids = all_doc_ids

    if dense_embeddings is None:
        logger.error("No embeddings to process")
        return 1

    # Build FAISS index
    logger.info("\n4. Building FAISS index...")
    index = build_faiss_index(
        dense_embeddings,
        index_type=args.index_type,
        nlist=args.nlist,
        use_gpu=args.device == "cuda",
    )

    # Save FAISS index
    import faiss
    faiss_path = output_dir / "faiss_index.bin"
    faiss.write_index(index, str(faiss_path))
    logger.info(f"   Saved FAISS index: {faiss_path}")

    # Build and save sparse matrix
    logger.info("\n5. Building sparse matrix...")
    sparse_matrix = sparse_weights_to_csr(sparse_weights)

    import scipy.sparse as sp
    sparse_path = output_dir / "sparse_weights.npz"
    sp.save_npz(str(sparse_path), sparse_matrix)
    logger.info(f"   Saved sparse matrix: {sparse_path}")

    # Save metadata
    logger.info("\n6. Saving metadata...")
    metadata_path = output_dir / "passages_metadata.jsonl"

    # Reload corpus if needed (for full metadata)
    if not args.merge_only:
        with open(metadata_path, "w") as f:
            for i, doc in enumerate(documents):
                metadata = {
                    "doc_id": doc.get("doc_id", str(i)),
                    "title": doc.get("title", ""),
                    "text": doc.get("contents", ""),
                    "metadata": doc.get("metadata", {}),
                }
                f.write(json.dumps(metadata) + "\n")
    else:
        # Use doc_ids from checkpoints
        with open(metadata_path, "w") as f:
            for doc_id in doc_ids:
                f.write(json.dumps({"doc_id": doc_id}) + "\n")

    logger.info(f"   Saved metadata: {metadata_path}")

    # Save config
    config = {
        "encoder_model": "BAAI/bge-m3",
        "encoder_type": "bgem3",
        "embedding_dimension": 1024,
        "index_type": args.index_type,
        "metric_type": "ip",
        "normalize_embeddings": True,
        "num_passages": len(doc_ids),
        "index_params": {
            "nlist": args.nlist,
            "nprobe": 256,
        },
        "sparse_enabled": True,
        "sparse_vocab_size": 250002,
    }

    config_path = output_dir / "index_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    logger.info(f"   Saved config: {config_path}")

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("Build Complete!")
    logger.info("=" * 60)
    logger.info(f"Documents indexed: {len(doc_ids):,}")
    logger.info(f"Dense index size: {faiss_path.stat().st_size / (1024**3):.2f} GB")
    logger.info(f"Sparse matrix size: {sparse_path.stat().st_size / (1024**3):.2f} GB")
    logger.info(f"Output directory: {output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
