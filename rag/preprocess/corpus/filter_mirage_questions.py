"""
Filter MIRAGE questions to textbook-answerable subset.

Filters MIRAGE dataset to only include MedQA questions, which are derived
from 18 English medical textbooks used by USMLE students.

Usage:
    python -m rag.preprocess.corpus.filter_mirage_questions \\
        --input data/original/mirage/train.jsonl \\
        --output data/evaluation/filtered_mirage_questions.jsonl
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any
from loguru import logger


def load_mirage_questions(input_file: str) -> List[Dict[str, Any]]:
    """
    Load MIRAGE questions from JSONL file.

    Args:
        input_file: Path to MIRAGE train.jsonl file

    Returns:
        List of question dictionaries
    """
    logger.info(f"Loading MIRAGE questions from {input_file}")
    questions = []

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                questions.append(json.loads(line))

    logger.info(f"Loaded {len(questions)} questions")
    return questions


def filter_textbook_questions(questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Filter questions to only include MedQA (textbook-derived) questions.

    MedQA questions are identified by metadata['source_dataset'] == 'medqa'.
    MedQA is derived from 18 English medical textbooks:
    - Gray's Anatomy
    - Harrison's Principles of Internal Medicine
    - Current Medical Diagnosis & Treatment
    - And 15 others

    Args:
        questions: List of all MIRAGE questions

    Returns:
        List of filtered MedQA questions
    """
    logger.info("Filtering for MedQA (textbook) questions...")

    filtered = []
    source_counts = {}

    for question in questions:
        metadata = question.get("metadata", {})
        source_dataset = metadata.get("source_dataset", "")

        # Track source distribution
        source_counts[source_dataset] = source_counts.get(source_dataset, 0) + 1

        # Filter for MedQA
        if source_dataset == "medqa":
            filtered.append(question)

    logger.info(f"Source dataset distribution:")
    for source, count in sorted(source_counts.items()):
        logger.info(f"  {source}: {count} questions")

    logger.info(f"✓ Filtered to {len(filtered)} MedQA questions ({len(filtered)/len(questions)*100:.1f}%)")

    return filtered


def save_filtered_questions(
    questions: List[Dict[str, Any]],
    output_file: str
):
    """
    Save filtered questions to JSONL file.

    Args:
        questions: List of filtered questions
        output_file: Output file path
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving filtered questions to {output_file}")

    with open(output_file, "w", encoding="utf-8") as f:
        for question in questions:
            f.write(json.dumps(question) + "\n")

    logger.success(f"✓ Saved {len(questions)} filtered questions")


def main():
    parser = argparse.ArgumentParser(
        description="Filter MIRAGE questions to textbook-answerable subset"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/original/mirage/train.jsonl",
        help="Path to MIRAGE train.jsonl file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/evaluation/filtered_mirage_questions.jsonl",
        help="Output path for filtered questions"
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate filtering by sampling random questions"
    )

    args = parser.parse_args()

    # Load questions
    questions = load_mirage_questions(args.input)

    # Filter to textbook questions
    filtered_questions = filter_textbook_questions(questions)

    # Save filtered questions
    save_filtered_questions(filtered_questions, args.output)

    # Print summary
    print("\n" + "=" * 60)
    print("MIRAGE FILTERING SUMMARY")
    print("=" * 60)
    print(f"Total questions: {len(questions):,}")
    print(f"MedQA questions: {len(filtered_questions):,}")
    print(f"Filtered: {len(filtered_questions)/len(questions)*100:.1f}%")
    print(f"\nMedQA questions are derived from 18 medical textbooks")
    print(f"including Gray's Anatomy, Harrison's, CMDT, and others.")
    print(f"\nOutput: {args.output}")
    print("=" * 60 + "\n")

    # Validation sampling
    if args.validate:
        import random
        sample_size = min(20, len(filtered_questions))
        sample = random.sample(filtered_questions, sample_size)

        print("\n" + "=" * 60)
        print(f"VALIDATION SAMPLE ({sample_size} random questions)")
        print("=" * 60)
        for i, q in enumerate(sample[:5], 1):  # Show first 5
            print(f"\n{i}. {q['question'][:100]}...")
            print(f"   Source: {q['metadata']['source_dataset']}")
            print(f"   Answer: {q['metadata']['correct_option']}")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
