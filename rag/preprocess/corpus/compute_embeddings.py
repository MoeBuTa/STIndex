"""
Compute embeddings for MedCorp corpus using multiple retrievers.

For RRF-4 retriever, we need embeddings from:
1. facebook/contriever (general-domain semantic)
2. allenai/specter (scientific-domain)
3. ncbi/MedCPT-Query-Encoder (biomedical-domain)

Usage:
    # Compute all embeddings for MedCorp
    python -m rag.preprocess.corpus.compute_embeddings \\
        --corpus data/original/medcorp/train.jsonl \\
        --output data/embeddings/medcorp \\
        --models contriever specter medcpt

    # Compute only for specific retriever
    python -m rag.preprocess.corpus.compute_embeddings \\
        --corpus data/original/medcorp/train.jsonl \\
        --output data/embeddings/medcorp \\
        --models medcpt
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import List, Dict

import torch
from loguru import logger
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


# Model configurations from MedRAG
RETRIEVER_MODELS = {
    "contriever": {
        "model_name": "facebook/contriever",
        "sep": ". ",  # Separator for title and content
        "metric": "inner_product"
    },
    "specter": {
        "model_name": "allenai/specter",
        "sep": " [SEP] ",  # Uses sep_token
        "metric": "l2"
    },
    "medcpt": {
        "model_name": "ncbi/MedCPT-Query-Encoder",
        "sep": " ",  # Keeps as list [title, content]
        "metric": "inner_product"
    }
}


def load_corpus(corpus_path: str) -> List[Dict]:
    """
    Load corpus from JSONL file.

    Args:
        corpus_path: Path to corpus JSONL file

    Returns:
        List of document dictionaries
    """
    logger.info(f"Loading corpus from {corpus_path}")
    documents = []

    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                documents.append(json.loads(line))

    logger.info(f"Loaded {len(documents)} documents")
    return documents


def prepare_texts_for_embedding(
    documents: List[Dict],
    retriever_type: str
) -> List[str]:
    """
    Prepare document texts according to retriever requirements.

    Different retrievers have different text preprocessing:
    - Contriever: "title. content"
    - SPECTER: "title [SEP] content"
    - MedCPT: "title content" (or list format)

    Args:
        documents: List of document dictionaries
        retriever_type: One of 'contriever', 'specter', 'medcpt'

    Returns:
        List of formatted text strings
    """
    config = RETRIEVER_MODELS[retriever_type]
    sep = config["sep"]
    texts = []

    for doc in documents:
        title = doc.get("title", "")
        content = doc.get("contents", "")

        # Combine according to retriever requirements
        if title and content:
            text = f"{title}{sep}{content}"
        elif content:
            text = content
        else:
            text = title or ""

        texts.append(text)

    return texts


def compute_embeddings(
    texts: List[str],
    model_name: str,
    batch_size: int = 32,
    device: str = "cuda"
) -> np.ndarray:
    """
    Compute embeddings for texts using sentence transformer.

    Args:
        texts: List of text strings
        model_name: HuggingFace model name
        batch_size: Batch size for encoding
        device: Device to use ('cuda' or 'cpu')

    Returns:
        Numpy array of embeddings (num_docs x embedding_dim)
    """
    logger.info(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name, device=device)

    logger.info(f"Computing embeddings for {len(texts)} documents...")
    logger.info(f"  Batch size: {batch_size}")
    logger.info(f"  Device: {device}")

    # Encode with progress bar
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True  # L2 normalization
    )

    logger.info(f"✓ Computed embeddings: shape={embeddings.shape}")
    return embeddings


def save_embeddings(
    embeddings: np.ndarray,
    documents: List[Dict],
    retriever_type: str,
    output_dir: str
) -> None:
    """
    Save embeddings and metadata to disk.

    Output format (compatible with MedRAG):
    - {retriever_type}_embeddings.npy: Embedding matrix
    - {retriever_type}_metadata.jsonl: Document metadata (doc_id, title)
    - {retriever_type}_config.json: Configuration

    Args:
        embeddings: Embedding matrix (num_docs x embedding_dim)
        documents: Original documents
        retriever_type: Retriever name
        output_dir: Output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    config = RETRIEVER_MODELS[retriever_type]

    # Save embeddings
    embeddings_file = output_path / f"{retriever_type}_embeddings.npy"
    np.save(embeddings_file, embeddings)
    logger.info(f"✓ Saved embeddings: {embeddings_file}")
    logger.info(f"  Size: {embeddings_file.stat().st_size / 1024 / 1024:.1f} MB")

    # Save metadata
    metadata_file = output_path / f"{retriever_type}_metadata.jsonl"
    with open(metadata_file, "w", encoding="utf-8") as f:
        for idx, doc in enumerate(documents):
            metadata = {
                "index": idx,
                "doc_id": doc.get("doc_id", ""),
                "title": doc.get("title", ""),
                "source_corpus": doc.get("metadata", {}).get("source_corpus", ""),
                "original_id": doc.get("metadata", {}).get("original_id", "")
            }
            f.write(json.dumps(metadata) + "\n")
    logger.info(f"✓ Saved metadata: {metadata_file}")

    # Save configuration
    config_file = output_path / f"{retriever_type}_config.json"
    config_data = {
        "retriever_type": retriever_type,
        "model_name": config["model_name"],
        "metric": config["metric"],
        "num_documents": len(documents),
        "embedding_dim": embeddings.shape[1],
        "text_separator": config["sep"]
    }
    with open(config_file, "w") as f:
        json.dump(config_data, f, indent=2)
    logger.info(f"✓ Saved config: {config_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Compute embeddings for MedCorp corpus"
    )
    parser.add_argument(
        "--corpus",
        type=str,
        required=True,
        help="Path to corpus JSONL file"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for embeddings"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(RETRIEVER_MODELS.keys()),
        default=["contriever", "specter", "medcpt"],
        help="Retriever models to compute embeddings for"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for encoding"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda or cpu)"
    )
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("MedCorp Embedding Computation for RRF-4")
    logger.info("=" * 80)
    logger.info(f"Corpus: {args.corpus}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Models: {', '.join(args.models)}")
    logger.info(f"Device: {args.device}")
    logger.info("=" * 80)

    # Load corpus
    documents = load_corpus(args.corpus)

    # Compute embeddings for each model
    for retriever_type in args.models:
        logger.info("")
        logger.info("=" * 80)
        logger.info(f"Computing embeddings for: {retriever_type}")
        logger.info(f"  Model: {RETRIEVER_MODELS[retriever_type]['model_name']}")
        logger.info("=" * 80)

        # Prepare texts
        texts = prepare_texts_for_embedding(documents, retriever_type)

        # Compute embeddings
        embeddings = compute_embeddings(
            texts,
            RETRIEVER_MODELS[retriever_type]["model_name"],
            batch_size=args.batch_size,
            device=args.device
        )

        # Save to disk
        save_embeddings(embeddings, documents, retriever_type, args.output)

    logger.info("")
    logger.info("=" * 80)
    logger.info("✓ All embeddings computed successfully!")
    logger.info("=" * 80)
    logger.info(f"Output directory: {args.output}")
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Build FAISS indices from embeddings")
    logger.info("  2. Build BM25 index for lexical retrieval")
    logger.info("  3. Test RRF-4 retriever on MIRAGE questions")


if __name__ == "__main__":
    main()
