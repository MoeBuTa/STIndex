"""
Semantic clustering of questions using sentence embeddings + KMeans.

Clusters questions to discover domain-specific dimensional schemas.
Uses SentenceTransformer for embeddings and KMeans for clustering.
"""

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger

try:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    SKLEARN_AVAILABLE = True
except ImportError:
    logger.warning("scikit-learn not installed. Install with: pip install scikit-learn")
    SKLEARN_AVAILABLE = False


class QuestionClusterer:
    """
    Semantic clustering of questions using sentence embeddings + KMeans.

    Features:
    - Embedding caching (avoid recomputing on subsequent runs)
    - Progress logging with loguru
    - Multiple output formats (CSV, JSON)
    - Cluster quality metrics (silhouette score, inertia)
    - Sample selection for schema discovery

    Follows patterns from:
    - EventClusterAnalyzer (clustering.py) - sklearn import, error handling
    - AnalysisDataExporter (export.py) - file I/O, output directory management

    Example:
        clusterer = QuestionClusterer()

        result = clusterer.cluster_questions_from_file(
            questions_file='data/original/mirage/train.jsonl',
            output_dir='data/schema_discovery/clusters',
            n_clusters=10,
            dataset_name='mirage'
        )

        # Output files:
        # - cluster_assignments.csv: All questions with cluster IDs
        # - cluster_analysis.json: Statistics and metadata
        # - cluster_samples.json: Representative samples for schema discovery
    """

    def __init__(
        self,
        model_name: str = 'all-MiniLM-L6-v2',
        cache_dir: str = 'data/questions/clustering/embeddings'
    ):
        """
        Initialize clusterer with embedding model.

        Args:
            model_name: SentenceTransformer model name
            cache_dir: Directory to cache embeddings
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for clustering")

        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Lazy loading - don't load model until first use
        self._encoder = None

    @property
    def encoder(self):
        """Lazy load SentenceTransformer model."""
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading SentenceTransformer model: {self.model_name}")
                self._encoder = SentenceTransformer(self.model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers not installed. "
                    "Install with: pip install sentence-transformers"
                )
        return self._encoder

    def cluster_questions_from_file(
        self,
        questions_file: str,
        output_dir: str,
        n_clusters: int = 10,
        dataset_name: str = 'mirage',
        n_samples_per_cluster: int = 20,
        force_recompute: bool = False
    ) -> Dict:
        """
        Load questions from JSONL, cluster, and save results.

        Args:
            questions_file: Path to JSONL file (e.g., 'data/original/mirage/train.jsonl')
            output_dir: Directory for output files (e.g., 'data/schema_discovery/clusters')
            n_clusters: Number of clusters (default: 10)
            dataset_name: Dataset identifier for caching
            n_samples_per_cluster: Number of representative samples per cluster
            force_recompute: Force recompute embeddings even if cache exists

        Returns:
            Dict with cluster statistics and file paths

        Output Files:
            - cluster_assignments.csv: All questions with cluster IDs
            - cluster_analysis.json: Statistics and metadata
            - cluster_samples.json: Representative samples for schema discovery
        """
        logger.info(f"Clustering questions from: {questions_file}")

        # Load questions from JSONL
        question_data = self._load_questions_from_file(questions_file)
        questions = [item['question'] for item in question_data]
        logger.info(f"Loaded {len(questions)} questions")

        # Compute or load embeddings
        embeddings = self.compute_embeddings(
            questions=questions,
            dataset_name=dataset_name,
            force_recompute=force_recompute
        )

        # Perform clustering
        logger.info(f"Clustering into {n_clusters} clusters...")
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)

        # Compute distances to centroids
        distances = self._compute_distances_to_centroids(embeddings, kmeans)

        # Organize results by cluster
        clusters = self._organize_clusters(
            question_data=question_data,
            questions=questions,
            labels=labels,
            distances=distances,
            centroids=kmeans.cluster_centers_
        )

        # Select representative samples for each cluster
        cluster_samples = {}
        for cluster_id, cluster_info in clusters.items():
            cluster_samples[cluster_id] = self._select_representative_samples(
                questions=cluster_info['questions'],
                embeddings=embeddings[cluster_info['question_indices']],
                centroid=cluster_info['centroid'],
                n_samples=n_samples_per_cluster
            )

        # Compute quality metrics
        quality_metrics = self._compute_cluster_quality_metrics(embeddings, labels)

        # Prepare metadata
        metadata = {
            'dataset_name': dataset_name,
            'n_questions': len(questions),
            'n_clusters': n_clusters,
            'model_name': self.model_name,
            'embedding_dim': embeddings.shape[1],
            'timestamp': datetime.now().isoformat()
        }

        # Save results
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self._save_cluster_assignments(output_dir, question_data, labels, distances)
        self._save_cluster_analysis(output_dir, clusters, metadata, quality_metrics)
        self._save_cluster_samples(output_dir, cluster_samples)

        logger.info(f"✓ Clustering complete. Results saved to: {output_dir}")
        logger.info(f"  Silhouette score: {quality_metrics['silhouette_score']:.3f}")
        logger.info(f"  Inertia: {quality_metrics['inertia']:.2f}")

        return {
            'metadata': metadata,
            'quality_metrics': quality_metrics,
            'clusters': clusters,
            'cluster_samples': cluster_samples,
            'output_files': {
                'assignments': str(output_dir / 'cluster_assignments.csv'),
                'analysis': str(output_dir / 'cluster_analysis.json'),
                'samples': str(output_dir / 'cluster_samples.json')
            }
        }

    def compute_embeddings(
        self,
        questions: List[str],
        dataset_name: str,
        force_recompute: bool = False
    ) -> np.ndarray:
        """
        Compute or load cached embeddings.

        Caching Strategy:
        - Cache directory: data/questions/clustering/embeddings/
        - Cache file: {dataset_name}_embeddings.npy
        - Metadata: {dataset_name}_embeddings_meta.json
        - Validates: model name, question count match before loading
        - Auto-recomputes if cache invalid or corrupted

        Args:
            questions: List of question texts
            dataset_name: Dataset identifier (e.g., 'mirage')
            force_recompute: Force recompute even if cache exists

        Returns:
            numpy array of embeddings (n_questions, embedding_dim)
        """
        cache_file = self.cache_dir / f"{dataset_name}_embeddings.npy"
        meta_file = self.cache_dir / f"{dataset_name}_embeddings_meta.json"

        # Try to load from cache
        if not force_recompute and cache_file.exists() and meta_file.exists():
            try:
                # Load and validate metadata
                with open(meta_file, 'r') as f:
                    meta = json.load(f)

                if (meta['model_name'] == self.model_name and
                    meta['n_questions'] == len(questions)):
                    logger.info(f"Loading cached embeddings from: {cache_file}")
                    embeddings = np.load(cache_file)

                    # Validate shape
                    if embeddings.shape[0] == len(questions):
                        logger.info(f"✓ Loaded {embeddings.shape[0]} embeddings from cache")
                        return embeddings
                    else:
                        logger.warning("Cache shape mismatch, recomputing...")
                else:
                    logger.warning("Cache metadata mismatch, recomputing...")
            except Exception as e:
                logger.warning(f"Error loading cache: {e}, recomputing...")

        # Compute embeddings
        logger.info(f"Computing embeddings for {len(questions)} questions...")
        logger.info(f"Using model: {self.model_name}")

        embeddings = self.encoder.encode(
            questions,
            show_progress_bar=True,
            convert_to_numpy=True,
            batch_size=32
        )

        # Save to cache
        logger.info(f"Saving embeddings to cache: {cache_file}")
        np.save(cache_file, embeddings)

        # Save metadata
        meta = {
            'model_name': self.model_name,
            'n_questions': len(questions),
            'embedding_dim': embeddings.shape[1],
            'timestamp': datetime.now().isoformat()
        }
        with open(meta_file, 'w') as f:
            json.dump(meta, f, indent=2)

        logger.info(f"✓ Computed embeddings: {embeddings.shape}")
        return embeddings

    def _load_questions_from_file(self, questions_file: str) -> List[Dict]:
        """Load questions from JSONL file."""
        import jsonlines

        question_data = []
        with jsonlines.open(questions_file) as reader:
            for item in reader:
                question_data.append(item)

        return question_data

    def _compute_distances_to_centroids(
        self,
        embeddings: np.ndarray,
        kmeans: KMeans
    ) -> np.ndarray:
        """Compute Euclidean distance from each point to its cluster centroid."""
        centroids = kmeans.cluster_centers_
        labels = kmeans.labels_

        distances = np.zeros(len(embeddings))
        for i, (embedding, label) in enumerate(zip(embeddings, labels)):
            distances[i] = np.linalg.norm(embedding - centroids[label])

        return distances

    def _organize_clusters(
        self,
        question_data: List[Dict],
        questions: List[str],
        labels: np.ndarray,
        distances: np.ndarray,
        centroids: np.ndarray
    ) -> Dict:
        """Organize questions by cluster with metadata."""
        clusters = {}

        for cluster_id in range(len(centroids)):
            cluster_mask = labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]

            # Get source distribution if metadata available
            source_dist = {}
            if question_data and 'metadata' in question_data[0]:
                sources = [
                    question_data[i].get('metadata', {}).get('source_dataset', 'unknown')
                    for i in cluster_indices
                ]
                source_dist = {s: sources.count(s) for s in set(sources)}

            clusters[cluster_id] = {
                'size': int(cluster_mask.sum()),
                'question_indices': cluster_indices.tolist(),
                'questions': [questions[i] for i in cluster_indices],
                'centroid': centroids[cluster_id].tolist(),
                'centroid_norm': float(np.linalg.norm(centroids[cluster_id])),
                'avg_distance_to_centroid': float(distances[cluster_mask].mean()),
                'source_distribution': source_dist
            }

        return clusters

    def _select_representative_samples(
        self,
        questions: List[str],
        embeddings: np.ndarray,
        centroid: np.ndarray,
        n_samples: int
    ) -> List[str]:
        """
        Select questions closest to cluster centroid.

        Args:
            questions: All questions in cluster
            embeddings: Embeddings for questions in cluster
            centroid: Cluster centroid
            n_samples: Number of samples to select

        Returns:
            List of representative question texts
        """
        # Compute distances to centroid
        if isinstance(centroid, list):
            centroid = np.array(centroid)

        distances = np.linalg.norm(embeddings - centroid, axis=1)

        # Get indices of closest questions
        n_samples = min(n_samples, len(questions))
        closest_indices = np.argsort(distances)[:n_samples]

        return [questions[i] for i in closest_indices]

    def _compute_cluster_quality_metrics(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray
    ) -> Dict:
        """
        Compute clustering quality metrics.

        Metrics:
        - Silhouette score: [-1, 1], higher is better
        - Inertia: Sum of squared distances to centroids, lower is better

        Args:
            embeddings: Question embeddings
            labels: Cluster assignments

        Returns:
            Dict with quality metrics
        """
        # Silhouette score (may take a while for large datasets)
        if len(embeddings) > 10000:
            logger.info("Large dataset, using sample for silhouette score...")
            sample_size = 10000
            indices = np.random.choice(len(embeddings), sample_size, replace=False)
            sil_score = silhouette_score(embeddings[indices], labels[indices])
        else:
            sil_score = silhouette_score(embeddings, labels)

        # Inertia (from KMeans)
        kmeans_temp = KMeans(n_clusters=len(np.unique(labels)), random_state=42, n_init=10)
        kmeans_temp.fit(embeddings)
        inertia = kmeans_temp.inertia_

        # Average cluster size
        cluster_sizes = [np.sum(labels == i) for i in np.unique(labels)]
        avg_size = np.mean(cluster_sizes)

        return {
            'silhouette_score': float(sil_score),
            'inertia': float(inertia),
            'avg_cluster_size': float(avg_size),
            'min_cluster_size': int(min(cluster_sizes)),
            'max_cluster_size': int(max(cluster_sizes))
        }

    def _save_cluster_assignments(
        self,
        output_dir: Path,
        question_data: List[Dict],
        labels: np.ndarray,
        distances: np.ndarray
    ):
        """Save CSV with all question-cluster assignments."""
        assignments_file = output_dir / 'cluster_assignments.csv'

        with open(assignments_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'question_id',
                'cluster_id',
                'question',
                'distance_to_centroid',
                'source_dataset'
            ])

            for i, (item, label, distance) in enumerate(zip(question_data, labels, distances)):
                writer.writerow([
                    item.get('id', f'q_{i}'),
                    int(label),
                    item['question'][:500],  # Truncate long questions
                    f"{distance:.4f}",
                    item.get('metadata', {}).get('source_dataset', 'unknown')
                ])

        logger.info(f"Saved cluster assignments: {assignments_file}")

    def _save_cluster_analysis(
        self,
        output_dir: Path,
        clusters: Dict,
        metadata: Dict,
        quality_metrics: Dict
    ):
        """Save JSON with cluster statistics and metadata."""
        analysis_file = output_dir / 'cluster_analysis.json'

        # Prepare cluster info (remove full question lists for size)
        clusters_info = {}
        for cluster_id, info in clusters.items():
            clusters_info[str(cluster_id)] = {
                'size': info['size'],
                'centroid_norm': info['centroid_norm'],
                'avg_distance_to_centroid': info['avg_distance_to_centroid'],
                'source_distribution': info['source_distribution']
            }

        analysis = {
            'metadata': metadata,
            'quality_metrics': quality_metrics,
            'clusters': clusters_info
        }

        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)

        logger.info(f"Saved cluster analysis: {analysis_file}")

    def _save_cluster_samples(
        self,
        output_dir: Path,
        cluster_samples: Dict[int, List[str]]
    ):
        """Save JSON with representative samples for schema discovery."""
        samples_file = output_dir / 'cluster_samples.json'

        # Convert int keys to strings for JSON
        samples_json = {str(k): v for k, v in cluster_samples.items()}

        with open(samples_file, 'w') as f:
            json.dump(samples_json, f, indent=2)

        logger.info(f"Saved cluster samples: {samples_file}")
