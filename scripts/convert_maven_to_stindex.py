#!/usr/bin/env python3
"""
Convert MAVEN event detection dataset to STIndex format.

Usage:
    python scripts/convert_maven_to_stindex.py --input data/input/maven/train.jsonl \
                                                 --output data/input/maven_stindex_train.jsonl \
                                                 --limit 100

    # Convert all splits
    python scripts/convert_maven_to_stindex.py --convert-all
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any, List
from loguru import logger


def convert_document(maven_doc: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a single MAVEN document to STIndex format.

    Args:
        maven_doc: Original MAVEN document with 'content', 'events', etc.

    Returns:
        STIndex-formatted document with 'id', 'title', 'text', 'metadata'
    """
    # Concatenate all sentences into full text
    full_text = ' '.join([sent['sentence'] for sent in maven_doc['content']])

    # Extract event types (if available)
    event_types = []
    if 'events' in maven_doc:
        event_types = list(set([evt['type'] for evt in maven_doc['events']]))

    # Create STIndex format
    stindex_doc = {
        'id': maven_doc['id'],
        'title': maven_doc['title'],
        'text': full_text,
        'source': 'maven',
        'metadata': {
            'num_sentences': len(maven_doc['content']),
            'num_tokens': sum([len(sent['tokens']) for sent in maven_doc['content']]),
            'original_event_types': event_types,
            'num_events': len(maven_doc.get('events', [])),
        }
    }

    return stindex_doc


def convert_file(
    input_path: Path,
    output_path: Path,
    limit: int = None,
    verbose: bool = True
) -> int:
    """
    Convert a MAVEN JSONL file to STIndex format.

    Args:
        input_path: Path to input MAVEN .jsonl file
        output_path: Path to output STIndex .jsonl file
        limit: Optional limit on number of documents to convert
        verbose: Whether to log progress

    Returns:
        Number of documents converted
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    converted_count = 0

    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:

        for i, line in enumerate(infile):
            if limit and i >= limit:
                break

            try:
                maven_doc = json.loads(line)
                stindex_doc = convert_document(maven_doc)

                # Write to output file
                outfile.write(json.dumps(stindex_doc, ensure_ascii=False) + '\n')
                converted_count += 1

                if verbose and (converted_count % 100 == 0):
                    logger.info(f"Converted {converted_count} documents...")

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse line {i+1}: {e}")
                continue
            except Exception as e:
                logger.error(f"Error converting document {i+1}: {e}")
                continue

    if verbose:
        logger.success(f"✓ Converted {converted_count} documents to {output_path}")

    return converted_count


def main():
    parser = argparse.ArgumentParser(
        description="Convert MAVEN dataset to STIndex format",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        '--input',
        type=Path,
        help='Input MAVEN .jsonl file'
    )

    parser.add_argument(
        '--output',
        type=Path,
        help='Output STIndex .jsonl file'
    )

    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of documents to convert (default: no limit)'
    )

    parser.add_argument(
        '--convert-all',
        action='store_true',
        help='Convert all splits (train, valid, test) with default output names'
    )

    parser.add_argument(
        '--base-dir',
        type=Path,
        default=Path('data/input/maven'),
        help='Base directory for MAVEN dataset (default: data/input/maven)'
    )

    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress logging'
    )

    args = parser.parse_args()

    # Configure logging
    if args.quiet:
        logger.remove()
        logger.add(lambda msg: None)  # Suppress output

    if args.convert_all:
        # Convert all splits
        splits = ['train', 'valid', 'test']
        total_converted = 0

        logger.info("Converting all MAVEN splits to STIndex format...")

        for split in splits:
            input_path = args.base_dir / f'{split}.jsonl'
            output_path = args.base_dir / f'stindex_{split}.jsonl'

            if not input_path.exists():
                logger.warning(f"Skipping {split}: file not found at {input_path}")
                continue

            logger.info(f"\n📄 Converting {split} split...")
            count = convert_file(
                input_path=input_path,
                output_path=output_path,
                limit=args.limit,
                verbose=not args.quiet
            )
            total_converted += count

        logger.success(f"\n✅ Total converted: {total_converted} documents")

    else:
        # Convert single file
        if not args.input or not args.output:
            parser.error("--input and --output are required unless using --convert-all")

        count = convert_file(
            input_path=args.input,
            output_path=args.output,
            limit=args.limit,
            verbose=not args.quiet
        )

        logger.success(f"✅ Converted {count} documents")


if __name__ == '__main__':
    main()
