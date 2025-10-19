"""
Distributed batch evaluation using Accelerate with config-based initialization.

Launch with:
    accelerate launch --config cfg/deepspeed_zero2.yaml eval/batch_evaluation_accelerate.py <dataset_path> [--config extract]

Features:
- Config-driven LLM initialization (uses cfg/extract.yml and provider configs)
- DeepSpeed ZeRO-2 support for memory efficiency
- Multi-GPU distributed processing
- Resume capability with checkpointing
- Compatible with eval/evaluation.py metrics
"""

import json
import time
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from datetime import datetime

# Add parent directory to path
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from rich.console import Console
from tqdm import tqdm
import torch

try:
    from accelerate import Accelerator
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    logger.error("Accelerate not installed. Install with: pip install accelerate deepspeed")
    sys.exit(1)

from stindex import STIndexExtractor
from stindex.agents.response.models import ExtractionResult
from stindex.agents.prompts.extraction import ExtractionPrompt
from eval.metrics import OverallMetrics


console = Console()


class EvaluationDataset(Dataset):
    """Dataset wrapper for evaluation samples"""

    def __init__(self, samples: List[Dict[str, Any]]):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    """Custom collate function that returns list of samples"""
    return batch


class AccelerateEvaluator:
    """Distributed batch evaluator using Accelerate"""

    def __init__(
        self,
        extractor: STIndexExtractor,
        output_dir: str = "data/output/eval_results",
        batch_size: int = 8,
        accelerator: Optional[Accelerator] = None,
    ):
        """
        Initialize accelerate evaluator.

        Args:
            extractor: STIndexExtractor instance (pre-configured)
            output_dir: Directory to save results
            batch_size: Number of samples per batch per GPU
            accelerator: Accelerator instance (creates one if None)
        """
        self.extractor = extractor
        self.llm = extractor.llm
        self.geocoder = extractor.geocoder
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.accelerator = accelerator or Accelerator()

        self.prompt_builder = ExtractionPrompt()
        self.metrics = OverallMetrics()

        # Only main process logs
        if self.accelerator.is_main_process:
            logger.info(f"Initialized on {self.accelerator.num_processes} GPUs")
            logger.info(f"Batch size per GPU: {batch_size}")
            logger.info(f"Total effective batch size: {batch_size * self.accelerator.num_processes}")

    def evaluate_dataset(
        self,
        dataset_path: str,
        resume: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluate on dataset with distributed batch processing.

        Args:
            dataset_path: Path to JSON dataset file
            resume: Whether to resume from existing results

        Returns:
            Dictionary with overall metrics
        """
        if self.accelerator.is_main_process:
            logger.info(f"Loading dataset from {dataset_path}")

        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)

        # Load existing results if resuming
        existing_completions, evaluated_ids = self._load_existing_results()
        if resume and existing_completions:
            if self.accelerator.is_main_process:
                logger.info(f"Resuming: {len(evaluated_ids)} entries already processed")
            # Filter out completed samples
            dataset = [s for s in dataset if s.get("id", "unknown") not in evaluated_ids]

        # Create dataloader
        eval_dataset = EvaluationDataset(dataset)
        dataloader = DataLoader(
            eval_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

        # Prepare with Accelerate (distributes across GPUs)
        dataloader = self.accelerator.prepare(dataloader)

        # Process batches
        all_results = []
        if self.accelerator.is_main_process:
            pbar = tqdm(total=len(dataset), desc="Evaluating")
        else:
            pbar = None

        for batch in dataloader:
            batch_results = self._process_batch(batch)
            all_results.extend(batch_results)

            # Save individual results immediately
            for result in batch_results:
                self._save_individual_result(result)

            if pbar:
                pbar.update(len(batch) * self.accelerator.num_processes)

        if pbar:
            pbar.close()

        # Gather results from all processes
        self.accelerator.wait_for_everyone()
        all_results = self._gather_results(all_results)

        # Main process calculates metrics and saves
        if self.accelerator.is_main_process:
            # Merge with existing results if resuming
            if resume and existing_completions:
                all_results.extend(list(existing_completions.values()))

            final_metrics = self._calculate_metrics(all_results)
            self._save_detailed_results(all_results)
            self._save_metrics_summary(final_metrics)

            return final_metrics

        return {}

    def _process_batch(self, batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process a batch of samples with the LLM.

        Args:
            batch: List of dataset entries

        Returns:
            List of evaluation results
        """
        # Build messages for batch
        messages_batch = []
        for sample in batch:
            messages = self.prompt_builder.build_messages(sample["text"])
            messages_batch.append(messages)

        # Batch extraction with LLM
        start_time = time.time()
        try:
            extraction_results = self.llm.generate_batch(
                messages_batch=messages_batch,
                response_model=ExtractionResult,
                max_tokens=2048,
            )
            processing_time = time.time() - start_time
            success = True
        except Exception as e:
            logger.error(f"Batch extraction failed on rank {self.accelerator.process_index}: {e}")
            processing_time = time.time() - start_time
            extraction_results = [None] * len(batch)
            success = False

        # Post-process and geocode
        results = []
        for sample, extraction_result in zip(batch, extraction_results):
            if extraction_result:
                # Add character positions
                text = sample["text"]
                for entity in extraction_result.temporal_mentions:
                    if entity.text in text:
                        start_char = text.find(entity.text)
                        entity.start_char = start_char
                        entity.end_char = start_char + len(entity.text)

                for entity in extraction_result.spatial_mentions:
                    if entity.text in text:
                        start_char = text.find(entity.text)
                        entity.start_char = start_char
                        entity.end_char = start_char + len(entity.text)

                # Geocode spatial mentions
                for entity in extraction_result.spatial_mentions:
                    parent_region = getattr(entity, 'parent_region', None)
                    coords = self.geocoder.get_coordinates(
                        entity.text,
                        context=parent_region
                    )
                    if coords:
                        entity.latitude = coords[0]
                        entity.longitude = coords[1]

            # Create result entry
            result = {
                "id": sample.get("id", "unknown"),
                "text": sample["text"],
                "ground_truth": sample["ground_truth"],
                "extraction_result": extraction_result,
                "success": success and extraction_result is not None,
                "processing_time": processing_time / len(batch),
                "process_rank": self.accelerator.process_index,
            }
            results.append(result)

        return results

    def _load_existing_results(self) -> tuple[Dict[str, Any], Set[str]]:
        """
        Load existing results from individual files.

        Returns:
            Tuple of (completions dict, set of evaluated IDs)
        """
        results_dir = self.output_dir / "individual_results"
        if not results_dir.exists():
            return {}, set()

        completions = {}
        evaluated_ids = set()

        try:
            for result_file in results_dir.glob("result_*.json"):
                with open(result_file, 'r', encoding='utf-8') as f:
                    result = json.load(f)
                    sample_id = result["id"]
                    completions[sample_id] = result
                    evaluated_ids.add(sample_id)

            return completions, evaluated_ids
        except Exception as e:
            logger.warning(f"Failed to load existing results: {e}")
            return {}, set()

    def _save_individual_result(self, result: Dict[str, Any]):
        """Save individual result immediately for fault tolerance"""
        results_dir = self.output_dir / "individual_results"
        results_dir.mkdir(parents=True, exist_ok=True)

        sample_id = result["id"]
        output_file = results_dir / f"result_{sample_id}.json"

        # Convert to serializable format
        serializable_result = {
            "id": result["id"],
            "text": result["text"],
            "ground_truth": result["ground_truth"],
            "success": result["success"],
            "processing_time": result["processing_time"],
            "process_rank": result.get("process_rank", 0),
        }

        if result["extraction_result"]:
            serializable_result["temporal_entities"] = [
                {
                    "text": e.text,
                    "normalized": e.normalized,
                    "temporal_type": str(e.temporal_type) if hasattr(e, 'temporal_type') else None,
                    "start_char": getattr(e, 'start_char', None),
                    "end_char": getattr(e, 'end_char', None),
                }
                for e in result["extraction_result"].temporal_mentions
            ]
            serializable_result["spatial_entities"] = [
                {
                    "text": e.text,
                    "location_type": str(e.location_type) if hasattr(e, 'location_type') else None,
                    "latitude": e.latitude,
                    "longitude": e.longitude,
                    "start_char": getattr(e, 'start_char', None),
                    "end_char": getattr(e, 'end_char', None),
                }
                for e in result["extraction_result"].spatial_mentions
            ]

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_result, f, indent=2, ensure_ascii=False)

    def _gather_results(self, local_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Gather results from all processes"""
        # Use Accelerate's gather to collect from all GPUs
        all_results = [local_results]
        if self.accelerator.num_processes > 1:
            all_results = self.accelerator.gather(all_results)

        # Flatten on main process
        if self.accelerator.is_main_process:
            flattened = []
            for results_list in all_results:
                flattened.extend(results_list)
            return flattened

        return local_results

    def _calculate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate evaluation metrics from results"""
        from eval.evaluation import STIndexEvaluator

        temp_evaluator = STIndexEvaluator()

        for result in results:
            if not result["extraction_result"]:
                continue

            # Convert back to entities for evaluation
            extraction_result = result["extraction_result"]

            # Evaluate temporal
            temporal_eval = temp_evaluator._evaluate_temporal(
                extraction_result.temporal_mentions if hasattr(extraction_result, 'temporal_mentions') else [],
                result["ground_truth"].get("temporal", [])
            )

            # Evaluate spatial
            spatial_eval = temp_evaluator._evaluate_spatial(
                extraction_result.spatial_mentions if hasattr(extraction_result, 'spatial_mentions') else [],
                result["ground_truth"].get("spatial", [])
            )

            # Update metrics
            temp_evaluator.metrics.total_documents += 1
            temp_evaluator.metrics.total_processing_time += result["processing_time"]
            if result["success"]:
                temp_evaluator.metrics.successful_extractions += 1

        return temp_evaluator.metrics.to_dict()

    def _save_detailed_results(self, results: List[Dict[str, Any]]):
        """Save detailed per-entry results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"accelerate_detailed_results_{timestamp}.json"

        # Convert to serializable format
        serializable_results = []
        for r in results:
            entry = {
                "id": r["id"],
                "text": r["text"],
                "ground_truth": r["ground_truth"],
                "success": r["success"],
                "processing_time": r["processing_time"],
            }

            if r.get("extraction_result") and hasattr(r["extraction_result"], 'temporal_mentions'):
                entry["temporal_entities"] = [
                    {
                        "text": e.text,
                        "normalized": e.normalized,
                        "temporal_type": str(e.temporal_type) if hasattr(e, 'temporal_type') else None,
                        "start_char": getattr(e, 'start_char', None),
                        "end_char": getattr(e, 'end_char', None),
                    }
                    for e in r["extraction_result"].temporal_mentions
                ]
                entry["spatial_entities"] = [
                    {
                        "text": e.text,
                        "location_type": str(e.location_type) if hasattr(e, 'location_type') else None,
                        "latitude": e.latitude,
                        "longitude": e.longitude,
                        "start_char": getattr(e, 'start_char', None),
                        "end_char": getattr(e, 'end_char', None),
                    }
                    for e in r["extraction_result"].spatial_mentions
                ]

            serializable_results.append(entry)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)

        logger.info(f"Detailed results saved to {output_file}")

    def _save_metrics_summary(self, metrics: Dict[str, Any]):
        """Save metrics summary with configuration details"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"accelerate_metrics_summary_{timestamp}.json"

        # Add configuration details to metrics
        metrics_with_config = {
            "config": self.extractor.config,
            "metrics": metrics
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(metrics_with_config, f, indent=2)

        logger.info(f"Metrics summary saved to {output_file}")
        console.print(f"\n[bold green]Distributed evaluation completed![/bold green]")
        console.print(f"Combined F1: {metrics['overall']['combined_f1']:.4f}")
        console.print(f"Success Rate: {metrics['overall']['success_rate']:.4f}")


def main():
    """Main entry point for accelerate launch"""
    import argparse

    parser = argparse.ArgumentParser(description="Distributed batch evaluation with Accelerate")
    parser.add_argument("dataset_path", type=str, help="Path to evaluation dataset JSON")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size per GPU")
    parser.add_argument("--output-dir", type=str, default="data/output/eval_results", help="Output directory")
    parser.add_argument("--config", type=str, default="extract", help="Config name to use (default: extract)")

    args = parser.parse_args()

    # Initialize accelerator
    accelerator = Accelerator()

    # Initialize STIndexExtractor (loads config automatically)
    if accelerator.is_main_process:
        logger.info(f"Initializing STIndexExtractor with config: {args.config}")

    extractor = STIndexExtractor(config_path=args.config)

    # Create evaluator
    evaluator = AccelerateEvaluator(
        extractor=extractor,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        accelerator=accelerator,
    )

    # Run evaluation
    metrics = evaluator.evaluate_dataset(args.dataset_path, resume=True)

    if accelerator.is_main_process:
        logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
