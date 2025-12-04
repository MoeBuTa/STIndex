"""
Test script for QuestionClusterer implementation.

This script tests the question clustering functionality with the MIRAGE dataset.
"""

from stindex.schema_discovery import QuestionClusterer
from pathlib import Path
import json


def main():
    print("=" * 80)
    print("Testing QuestionClusterer with MIRAGE Dataset")
    print("=" * 80)

    # Initialize clusterer
    print("\n1. Initializing QuestionClusterer...")
    clusterer = QuestionClusterer(
        model_name='all-MiniLM-L6-v2',
        cache_dir='data/questions/clustering/embeddings'
    )
    print("✓ Clusterer initialized")

    # Run clustering
    print("\n2. Clustering questions from MIRAGE dataset...")
    print("   This will take ~35 seconds on first run (computing embeddings)")
    print("   Subsequent runs will be faster (~3-5 seconds) due to caching")

    result = clusterer.cluster_questions_from_file(
        questions_file='data/original/mirage/train.jsonl',
        output_dir='data/schema_discovery/clusters',
        n_clusters=10,
        dataset_name='mirage',
        n_samples_per_cluster=20,
        force_recompute=False
    )

    # Verify outputs
    print("\n3. Verifying outputs...")

    output_dir = Path('data/schema_discovery/clusters')

    # Check files exist
    assignments_file = output_dir / 'cluster_assignments.csv'
    analysis_file = output_dir / 'cluster_analysis.json'
    samples_file = output_dir / 'cluster_samples.json'

    assert assignments_file.exists(), "cluster_assignments.csv not found"
    assert analysis_file.exists(), "cluster_analysis.json not found"
    assert samples_file.exists(), "cluster_samples.json not found"
    print("✓ All output files exist")

    # Verify assignments file
    with open(assignments_file, 'r') as f:
        lines = f.readlines()
        n_rows = len(lines) - 1  # Exclude header
        print(f"✓ cluster_assignments.csv: {n_rows} questions")

    # Verify analysis file
    with open(analysis_file, 'r') as f:
        analysis = json.load(f)
        print(f"✓ cluster_analysis.json loaded")
        print(f"   - Dataset: {analysis['metadata']['dataset_name']}")
        print(f"   - Questions: {analysis['metadata']['n_questions']}")
        print(f"   - Clusters: {analysis['metadata']['n_clusters']}")
        print(f"   - Model: {analysis['metadata']['model_name']}")
        print(f"   - Embedding dim: {analysis['metadata']['embedding_dim']}")
        print(f"   - Silhouette score: {analysis['quality_metrics']['silhouette_score']:.3f}")
        print(f"   - Inertia: {analysis['quality_metrics']['inertia']:.2f}")
        print(f"   - Avg cluster size: {analysis['quality_metrics']['avg_cluster_size']:.1f}")
        print(f"   - Min cluster size: {analysis['quality_metrics']['min_cluster_size']}")
        print(f"   - Max cluster size: {analysis['quality_metrics']['max_cluster_size']}")

    # Verify samples file
    with open(samples_file, 'r') as f:
        samples = json.load(f)
        print(f"✓ cluster_samples.json loaded")
        print(f"   - Number of clusters: {len(samples)}")
        for cluster_id, questions in samples.items():
            print(f"   - Cluster {cluster_id}: {len(questions)} samples")

    # Display sample questions from first cluster
    print("\n4. Sample questions from Cluster 0:")
    print("-" * 80)
    cluster_0_samples = samples['0'][:5]  # First 5 questions
    for i, question in enumerate(cluster_0_samples, 1):
        # Truncate long questions
        truncated = question[:150] + "..." if len(question) > 150 else question
        print(f"{i}. {truncated}")

    print("\n" + "=" * 80)
    print("✓ QuestionClusterer test completed successfully!")
    print("=" * 80)

    # Return result for inspection
    return result


if __name__ == "__main__":
    result = main()
