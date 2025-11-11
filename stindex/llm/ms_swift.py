"""
HuggingFace LLM Provider via MS-SWIFT - Native InferClient Integration.

Directly uses MS-SWIFT native functions:
- swift.llm.run_deploy() for server deployment
- swift.llm.InferClient for inference
- swift.llm.RequestConfig for configuration

Simple, minimal adapter that extends STIndex's base patterns.
"""

from typing import Any, Dict, List
from loguru import logger

try:
    from swift.llm import (
        run_deploy,
        DeployArguments,
        InferClient,
        InferRequest,
        RequestConfig,
    )
    SWIFT_AVAILABLE = True
except ImportError:
    SWIFT_AVAILABLE = False
    logger.warning("MS-SWIFT not installed. Install with: pip install ms-swift")

from stindex.llm.response.models import LLMResponse, TokenUsage


class MSSwiftLLM:
    """
    HuggingFace LLM provider using MS-SWIFT's native swift.llm.InferClient.

    Uses MS-SWIFT's native Python client for inference:
    - swift.llm.InferClient for generation
    - swift.llm.RequestConfig for parameters
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize HuggingFace LLM provider with MS-SWIFT InferClient."""
        if not SWIFT_AVAILABLE:
            raise ImportError("MS-SWIFT not installed. Install with: pip install ms-swift")

        self.config = config
        self.model_name = config.get("model_name", "Qwen/Qwen3-8B")
        self.base_url = config.get("base_url", "http://localhost:8000")

        # Generation parameters
        self.temperature = config.get("temperature", 0.0)
        self.max_tokens = config.get("max_tokens", 2048)
        self.seed = config.get("seed", None)
        self.top_p = config.get("top_p", 1.0)
        self.top_k = config.get("top_k", -1)

        # Parse host and port from base_url
        import re
        match = re.match(r'https?://([^:]+):(\d+)', self.base_url)
        if match:
            host, port = match.groups()
        else:
            host, port = "localhost", "8000"

        # Create native MS-SWIFT InferClient
        self.client = InferClient(host=host, port=int(port), api_key='EMPTY')

        logger.info(f"HuggingFace LLM initialized with MS-SWIFT InferClient")
        logger.info(f"  Model: {self.model_name}")
        logger.info(f"  Base URL: {self.base_url}")

    def _create_request_config(self) -> 'RequestConfig':
        """Create RequestConfig for MS-SWIFT."""
        kwargs = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
        }
        if self.seed is not None:
            kwargs["seed"] = self.seed
        if self.top_k > 0:
            kwargs["top_k"] = self.top_k

        return RequestConfig(**kwargs)

    def generate(self, messages: List[Dict[str, str]], max_retries: int = 3) -> LLMResponse:
        """Generate completion using MS-SWIFT InferClient with retry logic."""
        import time

        last_error = None

        for attempt in range(max_retries):
            try:
                # Create request config
                request_config = self._create_request_config()

                # Create InferRequest
                request = InferRequest(messages=messages)

                # Call MS-SWIFT InferClient
                responses = self.client.infer(
                    [request],
                    request_config=request_config,
                    model=self.model_name,
                )

                # Extract response (first item in list)
                response = responses[0]

                # Extract content and usage
                content = response.choices[0].message.content
                usage = response.usage if hasattr(response, 'usage') else None

                return LLMResponse(
                    model=getattr(response, 'model', self.model_name),
                    input=messages,
                    status="processed",
                    content=content,
                    usage=TokenUsage(
                        prompt_tokens=getattr(usage, 'prompt_tokens', 0) if usage else 0,
                        completion_tokens=getattr(usage, 'completion_tokens', 0) if usage else 0,
                        total_tokens=getattr(usage, 'total_tokens', 0) if usage else 0,
                    ),
                    success=True,
                )
            except Exception as e:
                last_error = e
                error_str = str(e).lower()

                # Retry on mimetype errors (server warmup) or connection errors
                if ('mimetype' in error_str or 'connection' in error_str) and attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    logger.warning(f"MS-SWIFT server error (attempt {attempt + 1}/{max_retries}): {e}")
                    logger.info(f"Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"MS-SWIFT generation failed: {e}")
                    break

        return LLMResponse(
            model=self.model_name,
            input=messages,
            status="error",
            error_msg=str(last_error),
            success=False,
        )

    def generate_batch(
        self,
        messages_batch: List[List[Dict[str, str]]],
        max_tokens: int = None,
        temperature: float = None,
    ) -> List[LLMResponse]:
        """Batch generation with concurrent processing."""
        import concurrent.futures

        # Temporarily override parameters
        old_max_tokens = self.max_tokens
        old_temperature = self.temperature

        if max_tokens is not None:
            self.max_tokens = max_tokens
        if temperature is not None:
            self.temperature = temperature

        # Process in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(32, len(messages_batch))) as executor:
            results = list(executor.map(self.generate, messages_batch))

        # Restore parameters
        self.max_tokens = old_max_tokens
        self.temperature = old_temperature

        return results


def deploy_server(config: Dict[str, Any]) -> 'ContextManager[int]':
    """
    Deploy HuggingFace model server using MS-SWIFT's swift.llm.run_deploy().

    Args:
        config: Configuration dict from hf.yml

    Returns:
        Context manager yielding port number

    Example:
        >>> from stindex.utils.config import load_config_from_file
        >>> config = load_config_from_file("hf")
        >>> with deploy_server(config) as port:
        ...     print(f"Server on port {port}")
    """
    if not SWIFT_AVAILABLE:
        raise ImportError("MS-SWIFT not installed. Install with: pip install ms-swift")

    deployment = config.get("deployment", {})
    vllm_config = deployment.get("vllm", {})

    # Auto-detect GPUs if tensor_parallel_size is "auto"
    tensor_parallel_size = vllm_config.get("tensor_parallel_size", 1)
    if tensor_parallel_size == "auto":
        import subprocess
        try:
            # Get number of available GPUs
            gpu_count = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                text=True
            ).strip().split('\n')
            tensor_parallel_size = len(gpu_count) if gpu_count and gpu_count[0] else 1
            logger.info(f"Auto-detected {tensor_parallel_size} GPU(s) for tensor parallelism")
        except Exception as e:
            logger.warning(f"Failed to auto-detect GPUs: {e}. Using 1 GPU.")
            tensor_parallel_size = 1

    # Build DeployArguments with only supported parameters
    deploy_args = DeployArguments(
        model=deployment.get("model", "Qwen/Qwen3-8B"),
        infer_backend=deployment.get("infer_backend", "vllm"),
        host=deployment.get("host", "0.0.0.0"),
        port=deployment.get("port", 8000),
        verbose=deployment.get("verbose", True),
        log_interval=deployment.get("log_interval", 20),
        use_hf=deployment.get("use_hf", True),  # Use HuggingFace downloaded models
        vllm_tensor_parallel_size=tensor_parallel_size,
        vllm_gpu_memory_utilization=vllm_config.get("gpu_memory_utilization", 0.9),
        result_path=deployment.get("result_path", "data/output/result"),  # MS-SWIFT inference logs
    )

    # Add optional parameters if present
    if vllm_config.get("max_model_len"):
        deploy_args.max_model_len = vllm_config["max_model_len"]
    if deployment.get("ckpt_dir"):
        deploy_args.ckpt_dir = deployment["ckpt_dir"]
    if deployment.get("merge_lora"):
        deploy_args.merge_lora = True
    if deployment.get("lora_modules"):
        deploy_args.lora_modules = deployment["lora_modules"]

    # Note: dtype and trust_remote_code are handled automatically by vLLM
    # when using use_hf=True with HuggingFace models

    logger.info(f"Deploying HuggingFace model server: {deploy_args.model}")
    return run_deploy(deploy_args)
