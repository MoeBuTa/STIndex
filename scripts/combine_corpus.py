#!/usr/bin/env python3
"""Combine textbooks and StatPearls into single corpus."""

import json
from pathlib import Path
from tqdm import tqdm

# Paths
textbooks_file = "data/original/medcorp/train.jsonl"
statpearls_file = "data/original/medcorp/raw/statpearls_processed.jsonl"
output_file = "data/original/medcorp/combined_corpus.jsonl"

print("Combining MedCorp corpora...")
print(f"  Textbooks: {textbooks_file}")
print(f"  StatPearls: {statpearls_file}")
print(f"  Output: {output_file}")
print()

# Count first
print("Counting documents...")
with open(textbooks_file) as f:
    textbook_count = sum(1 for _ in f)
with open(statpearls_file) as f:
    statpearls_count = sum(1 for _ in f)

print(f"  Textbooks: {textbook_count:,} docs")
print(f"  StatPearls: {statpearls_count:,} docs")
print(f"  Total: {textbook_count + statpearls_count:,} docs")
print()

# Combine
Path(output_file).parent.mkdir(parents=True, exist_ok=True)

with open(output_file, 'w') as out:
    # Copy textbooks
    print("Copying textbooks...")
    with open(textbooks_file) as f:
        for line in tqdm(f, total=textbook_count, desc="Textbooks"):
            out.write(line)
    
    # Copy StatPearls
    print("Copying StatPearls...")
    with open(statpearls_file) as f:
        for line in tqdm(f, total=statpearls_count, desc="StatPearls"):
            out.write(line)

print()
print(f"✓ Combined corpus saved to: {output_file}")
print(f"  Total: {textbook_count + statpearls_count:,} documents")
