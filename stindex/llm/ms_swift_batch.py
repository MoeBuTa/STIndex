"""
Batch inference using swift.llm.infer_main() - no server deployment needed.

This module provides offline batch inference using MS-SWIFT's native infer_main()
function. Unlike the server-based MSSwiftLLM which uses InferClient + HTTP,
this module processes batches directly without starting a server.

Key features:
- Uses swift.llm.infer_main(InferArguments(...)) for batch processing
- Supports tensor parallelism for multi-GPU inference
- Auto-detects available GPUs
- No server overhead - pure batch processing

Usage:
    from stindex.llm.ms_swift_batch import MSSwiftBatchLLM

    llm = MSSwiftBatchLLM({
        'model_path': 'Qwen/Qwen3-30B-A3B-Instruct-2507',
        'tensor_parallel_size': 2,
        'gpu_memory_utilization': 0.7,
        'max_model_len': 32768,
        'temperature': 0.0,
        'max_tokens': 4096,
    })

    messages_batch = [[{"role": "user", "content": "Hello"}]]
    responses = llm.generate_batch(messages_batch)
"""

import os
import subprocess
import tempfile
import json
from pathlib import Path
from typing import Any, Dict, List
from loguru import logger

try:
    from swift.llm import infer_main, InferArguments
    SWIFT_AVAILABLE = True
except ImportError:
    SWIFT_AVAILABLE = False
    logger.warning("MS-SWIFT not installed. Install with: pip install ms-swift[all]")

from stindex.llm.response.models import LLMResponse, TokenUsage


class MSSwiftBatchLLM:
    """
    HuggingFace LLM provider using MS-SWIFT's native swift.llm.infer_main().

    Uses MS-SWIFT's batch inference API for offline processing:
    - swift.llm.infer_main() for batch generation
    - swift.llm.InferArguments for configuration
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize HuggingFace LLM provider with MS-SWIFT batch mode."""
        if not SWIFT_AVAILABLE:
            raise ImportError("MS-SWIFT not installed. Install with: pip install ms-swift[all]")

        self.config = config
        self.model_path = config.get("model_path", "Qwen/Qwen3-30B-A3B-Instruct-2507")
        self.model_name = config.get("model_name", self.model_path.split("/")[-1])

        # Tensor parallelism settings
        self.tensor_parallel_size = config.get("tensor_parallel_size", "auto")
        if self.tensor_parallel_size == "auto":
            self.tensor_parallel_size = self._auto_detect_gpus()

        # Generation parameters
        self.temperature = config.get("temperature", 0.0)
        self.max_tokens = config.get("max_tokens", 4096)
        self.seed = config.get("seed", None)
        self.top_p = config.get("top_p", 1.0)
        self.top_k = config.get("top_k", -1)

        # vLLM settings
        self.gpu_memory_utilization = config.get("gpu_memory_utilization", 0.7)
        self.max_model_len = config.get("max_model_len", 32768)
        self.max_num_seqs = config.get("max_num_seqs", 256)  # vLLM concurrent sequences
        self.dtype = config.get("dtype", "auto")
        self.use_hf = config.get("use_hf", True)
        self.infer_backend = config.get("infer_backend", "vllm")

        # Additional vLLM features (for memory optimization)
        self.enable_prefix_caching = config.get("enable_prefix_caching", None)  # None = vLLM default
        self.enable_chunked_prefill = config.get("enable_chunked_prefill", None)  # None = vLLM default
        self.enforce_eager = config.get("enforce_eager", False)  # Disable CUDA graphs

        logger.info(f"HuggingFace Batch LLM initialized with MS-SWIFT infer_main()")
        logger.info(f"  Model: {self.model_path}")
        logger.info(f"  Tensor Parallel Size: {self.tensor_parallel_size} GPUs")
        logger.info(f"  GPU Memory Utilization: {self.gpu_memory_utilization}")
        logger.info(f"  Max Model Length: {self.max_model_len}")
        logger.info(f"  Max Num Seqs: {self.max_num_seqs}")

    def _auto_detect_gpus(self) -> int:
        """
        Auto-detect available GPUs.

        Returns:
            Number of GPUs available
        """
        # Check CUDA_VISIBLE_DEVICES first
        cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_devices:
            gpu_ids = [d.strip() for d in cuda_devices.split(",") if d.strip()]
            logger.info(f"Auto-detected {len(gpu_ids)} GPUs from CUDA_VISIBLE_DEVICES: {gpu_ids}")
            return len(gpu_ids)

        # Fallback to nvidia-smi
        try:
            result = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                text=True
            )
            gpu_ids = result.strip().split("\n")
            logger.info(f"Auto-detected {len(gpu_ids)} GPUs from nvidia-smi")
            return len(gpu_ids)
        except Exception as e:
            logger.warning(f"Failed to auto-detect GPUs: {e}. Using 1 GPU.")
            return 1

    def _create_infer_arguments(self, dataset_path: str, result_dir: str) -> InferArguments:
        """
        Create InferArguments for MS-SWIFT.

        Args:
            dataset_path: Path to input JSONL file
            result_dir: Directory for output files

        Returns:
            InferArguments instance
        """
        # Build InferArguments
        args = InferArguments(
            model=self.model_path,
            dataset=dataset_path,
            infer_backend=self.infer_backend,
            use_hf=self.use_hf,
            temperature=self.temperature,
            max_new_tokens=self.max_tokens,  # MS-SWIFT uses max_new_tokens
            top_p=self.top_p,
            # vLLM settings
            vllm_tensor_parallel_size=self.tensor_parallel_size,
            vllm_gpu_memory_utilization=self.gpu_memory_utilization,
            vllm_max_model_len=self.max_model_len,
            vllm_max_num_seqs=self.max_num_seqs,  # Control concurrent sequences
            # vLLM memory optimization features
            vllm_enable_prefix_caching=self.enable_prefix_caching,
            vllm_enforce_eager=self.enforce_eager,
            # Output settings
            result_path=result_dir,
        )

        # Add optional parameters
        if self.seed is not None:
            args.seed = self.seed
        if self.top_k > 0:
            args.top_k = self.top_k

        return args

    def generate(self, messages: List[Dict[str, str]], max_retries: int = 3) -> LLMResponse:
        """
        Generate completion for a single message using batch mode.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            max_retries: Maximum number of retry attempts (not used, for API compatibility)

        Returns:
            LLMResponse with generated content
        """
        # Delegate to batch generation with single item
        responses = self.generate_batch([messages])
        return responses[0]

    def generate_batch(
        self,
        messages_batch: List[List[Dict[str, str]]],
        max_tokens: int = None,
        temperature: float = None,
    ) -> List[LLMResponse]:
        """
        Batch generation using infer_main().

        Args:
            messages_batch: List of message lists to process
            max_tokens: Override max_tokens (optional)
            temperature: Override temperature (optional)

        Returns:
            List of LLMResponse objects
        """
        # Temporarily override parameters
        old_max_tokens = self.max_tokens
        old_temperature = self.temperature

        if max_tokens is not None:
            self.max_tokens = max_tokens
        if temperature is not None:
            self.temperature = temperature

        try:
            # 1. Write messages to temp JSONL
            with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
                input_file = f.name
                for messages in messages_batch:
                    f.write(json.dumps({"messages": messages}) + "\n")

            # 2. Create temp result directory
            result_dir = tempfile.mkdtemp(prefix="ms_swift_batch_")

            # 3. Build InferArguments
            args = self._create_infer_arguments(input_file, result_dir)

            logger.info(f"Running MS-SWIFT batch inference on {len(messages_batch)} samples...")
            logger.debug(f"  Input: {input_file}")
            logger.debug(f"  Output: {result_dir}")

            # 4. Run inference
            try:
                infer_main(args)
            except Exception as e:
                logger.error(f"MS-SWIFT infer_main() failed: {e}")
                raise

            # 5. Read results
            # MS-SWIFT saves results in: result_dir/{model_name}/deploy_result/{timestamp}.jsonl
            # We need to find the most recent output file
            result_files = list(Path(result_dir).rglob("*.jsonl"))
            if not result_files:
                raise FileNotFoundError(f"No output files found in {result_dir}")

            # Sort by modification time and get most recent
            result_file = max(result_files, key=lambda p: p.stat().st_mtime)
            logger.debug(f"Reading results from: {result_file}")

            responses = []
            with open(result_file, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    responses.append(self._parse_response(data))

            # 6. Cleanup
            Path(input_file).unlink()
            # Keep result_dir for debugging if needed
            # Can be cleaned up later or by tempfile cleanup

            logger.info(f"✓ Batch inference complete: {len(responses)} responses")
            return responses

        finally:
            # Restore parameters
            self.max_tokens = old_max_tokens
            self.temperature = old_temperature

    def _parse_response(self, data: Dict[str, Any]) -> LLMResponse:
        """
        Parse MS-SWIFT output to LLMResponse.

        Args:
            data: Dictionary from MS-SWIFT output JSONL

        Returns:
            LLMResponse instance
        """
        # MS-SWIFT output format:
        # {
        #     "response": "...",
        #     "usage": {"prompt_tokens": ..., "completion_tokens": ..., "total_tokens": ...},
        #     "generation_info": {...}
        # }

        response_text = data.get("response", "")
        usage_dict = data.get("usage", {})

        return LLMResponse(
            model=self.model_name,
            input=data.get("messages", []),  # Original messages if available
            status="processed",
            content=response_text,
            usage=TokenUsage(
                prompt_tokens=usage_dict.get("prompt_tokens", 0),
                completion_tokens=usage_dict.get("completion_tokens", 0),
                total_tokens=usage_dict.get("total_tokens", 0),
            ),
            success=True,
        )
