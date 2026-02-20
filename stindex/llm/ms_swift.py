"""
HuggingFace LLM Provider via MS-SWIFT - Native InferClient Integration.

Uses MS-SWIFT's swift.llm.InferClient for inference against a deployed server.
"""

import re
import time
from typing import Any, Dict

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

from stindex.llm.base import LLMClient


def _parse_base_url(base_url: str):
    """Parse host and port from a base URL string."""
    match = re.match(r'https?://([^:]+):(\d+)', base_url)
    if match:
        host, port = match.groups()
        return host, int(port)
    return "localhost", 8001


class MSSwiftClient(LLMClient):
    """HuggingFace LLM client using MS-SWIFT's native InferClient."""

    def __init__(self, model, base_url="http://localhost:8001", temperature=0.0, max_tokens=4096):
        if not SWIFT_AVAILABLE:
            raise ImportError("MS-SWIFT not installed. Install with: pip install ms-swift")

        host, port = _parse_base_url(base_url)
        self.client = InferClient(host=host, port=port, api_key="EMPTY")
        self.model = model
        self.request_config = RequestConfig(temperature=temperature, max_tokens=max_tokens)

        logger.info(f"MSSwiftClient initialized: model={model}, base_url={base_url}")

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        request = InferRequest(messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ])
        for attempt in range(3):
            try:
                responses = self.client.infer(
                    [request], request_config=self.request_config, model=self.model
                )
                return responses[0].choices[0].message.content
            except Exception as e:
                if attempt == 2:
                    raise
                error_str = str(e).lower()
                wait_time = 2 ** attempt
                logger.warning(
                    f"MS-SWIFT server error (attempt {attempt + 1}/3): {e}. "
                    f"Retrying in {wait_time}s..."
                )
                time.sleep(wait_time)


def deploy_server(config: Dict[str, Any]):
    """
    Deploy HuggingFace model server using MS-SWIFT's swift.llm.run_deploy().

    Args:
        config: Configuration dict from hf.yml

    Returns:
        Context manager yielding port number
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
            gpu_count = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                text=True
            ).strip().split('\n')
            tensor_parallel_size = len(gpu_count) if gpu_count and gpu_count[0] else 1
            logger.info(f"Auto-detected {tensor_parallel_size} GPU(s) for tensor parallelism")
        except Exception as e:
            logger.warning(f"Failed to auto-detect GPUs: {e}. Using 1 GPU.")
            tensor_parallel_size = 1

    deploy_args = DeployArguments(
        model=deployment.get("model", "Qwen/Qwen3-8B"),
        infer_backend=deployment.get("infer_backend", "vllm"),
        host=deployment.get("host", "0.0.0.0"),
        port=deployment.get("port", 8000),
        verbose=deployment.get("verbose", True),
        log_interval=deployment.get("log_interval", 20),
        use_hf=deployment.get("use_hf", True),
        vllm_tensor_parallel_size=tensor_parallel_size,
        vllm_gpu_memory_utilization=vllm_config.get("gpu_memory_utilization", 0.9),
        result_path=deployment.get("result_path", "data/output/result"),
    )

    if vllm_config.get("max_model_len"):
        deploy_args.max_model_len = vllm_config["max_model_len"]
    if deployment.get("ckpt_dir"):
        deploy_args.ckpt_dir = deployment["ckpt_dir"]
    if deployment.get("merge_lora"):
        deploy_args.merge_lora = True
    if deployment.get("lora_modules"):
        deploy_args.lora_modules = deployment["lora_modules"]

    logger.info(f"Deploying HuggingFace model server: {deploy_args.model}")
    return run_deploy(deploy_args)
