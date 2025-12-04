# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import os
import requests
from typing import Dict, List, Optional

import uuid

from data_process.utils.question_type import infer_question_type


# MIRAGE Medical Benchmark GitHub repository
MIRAGE_GITHUB_URL = "https://raw.githubusercontent.com/Teddy-XiongGZ/MIRAGE/main/benchmark.json"


def download_raw_data(raw_filepath: str) -> None:
    """Download MIRAGE benchmark data from GitHub repository"""
    print(f"Downloading MIRAGE benchmark from {MIRAGE_GITHUB_URL}")
    response = requests.get(MIRAGE_GITHUB_URL)
    response.raise_for_status()

    with open(raw_filepath, "wb") as fout:
        fout.write(response.content)

    print(f"Downloaded MIRAGE benchmark to {raw_filepath}")
    return


def load_raw_data(dataset_dir: str, split: str, filter_clinical_only: bool = True) -> List[dict]:
    """
    Load MIRAGE benchmark data

    Note: MIRAGE benchmark doesn't have train/dev/test splits in the original format.
    We treat the entire benchmark as 'train' for consistency with other datasets.

    Args:
        dataset_dir: Directory to save/cache data
        split: Split name (only 'train' supported)
        filter_clinical_only: If True, only load clinical exam questions (medqa, medmcqa, mmlu)
                             which are answerable by Textbooks + StatPearls corpus.
                             If False, load all questions including research QA.
    """
    raw_dir = os.path.join(dataset_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    raw_filepath = os.path.join(raw_dir, "benchmark.json")

    if not os.path.exists(raw_filepath):
        download_raw_data(raw_filepath)

    with open(raw_filepath, "r", encoding="utf-8") as fin:
        dataset = json.load(fin)

    # Clinical exam datasets (answered by Textbooks + StatPearls)
    CLINICAL_DATASETS = {"medqa", "medmcqa", "mmlu"}
    # Research QA datasets (answered by PubMed)
    RESEARCH_DATASETS = {"pubmedqa", "bioasq"}

    # MIRAGE benchmark structure:
    # {
    #   "medqa": {"0000": {question, options, answer}, "0001": {...}, ...},
    #   "medmcqa": {...},
    #   "pubmedqa": {...},
    #   ...
    # }
    # We flatten all datasets into a single list
    all_questions = []
    dataset_counts = {}
    filtered_out_counts = {}

    for dataset_name, questions_dict in dataset.items():
        count = 0
        filtered_count = 0

        for question_id, question_data in questions_dict.items():
            # Filter based on corpus availability
            if filter_clinical_only and dataset_name not in CLINICAL_DATASETS:
                filtered_count += 1
                continue

            # Add source dataset name and question ID to each question
            question_entry = question_data.copy()
            question_entry["source_dataset"] = dataset_name
            question_entry["original_question_id"] = question_id
            all_questions.append(question_entry)
            count += 1

        dataset_counts[dataset_name] = count
        if filtered_count > 0:
            filtered_out_counts[dataset_name] = filtered_count

    print(f"Loaded {len(all_questions)} questions from MIRAGE benchmark")
    if filter_clinical_only:
        print(f"  Filter: Clinical exams only (Textbooks + StatPearls corpus)")
    for dataset_name, count in dataset_counts.items():
        if count > 0:
            print(f"  ✓ {dataset_name}: {count} questions")
    if filtered_out_counts:
        print(f"\n  Filtered out (requires PubMed corpus):")
        for dataset_name, count in filtered_out_counts.items():
            print(f"  ✗ {dataset_name}: {count} questions")

    return all_questions


def format_raw_data(raw: dict) -> Optional[dict]:
    """
    Format MIRAGE question to match the pikerag protocol

    Input format (MIRAGE):
    {
        "question": "What is the most common cause of ...",
        "options": {"A": "Option A text", "B": "Option B text", ...},
        "answer": "A",
        "source_dataset": "medqa"  # Added during load_raw_data
    }

    Output format (pikerag protocol):
    {
        "id": "uuid",
        "question": "...",
        "answer_labels": ["Option A text"],
        "question_type": "multiple_choice",
        "metadata": {
            "original_id": "...",
            "source_dataset": "medqa",
            "options": {"A": "...", "B": "...", ...},
            "correct_option": "A"
        }
    }
    """
    # Extract answer text from options
    correct_option = raw.get("answer", "")
    options = raw.get("options", {})

    if not correct_option or correct_option not in options:
        print(f"Warning: Invalid answer '{correct_option}' for question: {raw.get('question', '')[:50]}...")
        return None

    answer_text = options[correct_option]
    answer_labels = [answer_text]

    # Infer question type (multiple choice)
    qtype = "multiple_choice"  # All MIRAGE questions are multiple choice

    # Format according to protocol
    formatted_data = {
        "id": uuid.uuid4().hex,
        "question": raw["question"],
        "answer_labels": answer_labels,
        "question_type": qtype,
        "metadata": {
            "original_id": raw.get("original_question_id", ""),
            "source_dataset": raw.get("source_dataset", "mirage"),
            "options": options,
            "correct_option": correct_option,
            # Note: MIRAGE doesn't provide retrieval contexts or supporting facts
            # These would need to be retrieved separately for RAG
            "retrieval_contexts": [],
            "supporting_facts": []
        }
    }

    return formatted_data
