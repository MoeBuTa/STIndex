"""
MS-SWIFT LLM Provider - Direct Integration.

Directly uses MS-SWIFT functions without wrapper classes:
- swift.llm.run_deploy() for server deployment
- swift.llm.inference_client() for inference
- swift.llm.XRequestConfig for configuration

Simple, minimal adapter that extends STIndex's base patterns.
"""

from typing import Any, Dict, List, Optional
from loguru import logger

try:
    from swift.llm import (
        run_deploy,
        DeployArguments,
        inference_client,
        XRequestConfig,
        get_model_list_client,
    )
    SWIFT_AVAILABLE = True
except ImportError:
    SWIFT_AVAILABLE = False
    logger.warning("MS-SWIFT not installed. Install with: pip install ms-swift")

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

from stindex.llm.response.models import LLMResponse, TokenUsage


class MSSwiftLLM:
    """
    MS-SWIFT LLM provider using native swift.llm functions.

    Directly calls:
    - swift.llm.inference_client() for generation
    - swift.llm.XRequestConfig for parameters
    - OpenAI SDK as fallback
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize MS-SWIFT LLM provider."""
        self.config = config
        self.model_name = config.get("model_name", "Qwen/Qwen3-8B")
        self.base_url = config.get("base_url", "http://localhost:8000")

        # Generation parameters
        self.temperature = config.get("temperature", 0.0)
        self.max_tokens = config.get("max_tokens", 2048)
        self.seed = config.get("seed", None)
        self.top_p = config.get("top_p", 1.0)
        self.top_k = config.get("top_k", -1)

        # Determine client mode
        self.use_swift = config.get("client_type", "auto") == "swift" or (
            config.get("client_type", "auto") == "auto" and SWIFT_AVAILABLE
        )

        if self.use_swift:
            if not SWIFT_AVAILABLE:
                raise ImportError("MS-SWIFT requested but not installed")
            logger.info(f"MS-SWIFT LLM using native swift.llm.inference_client")
        else:
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI SDK not installed")
            self.client = OpenAI(api_key='EMPTY', base_url=f"{self.base_url}/v1")
            logger.info(f"MS-SWIFT LLM using OpenAI SDK")

        logger.info(f"  Model: {self.model_name}")
        logger.info(f"  Base URL: {self.base_url}")

    def _create_request_config(self) -> Optional['XRequestConfig']:
        """Create XRequestConfig for MS-SWIFT."""
        if not SWIFT_AVAILABLE:
            return None

        kwargs = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
        }
        if self.seed is not None:
            kwargs["seed"] = self.seed
        if self.top_k > 0:
            kwargs["top_k"] = self.top_k

        return XRequestConfig(**kwargs)

    def generate(self, messages: List[Dict[str, str]]) -> LLMResponse:
        """Generate completion using MS-SWIFT or OpenAI SDK."""
        if self.use_swift:
            return self._generate_swift(messages)
        else:
            return self._generate_openai(messages)

    def _generate_swift(self, messages: List[Dict[str, str]]) -> LLMResponse:
        """Generate using swift.llm.inference_client()."""
        try:
            # Convert to query format
            query = messages[-1]['content'] if len(messages) == 1 else messages

            # Create request config
            request_config = self._create_request_config()

            # Call MS-SWIFT inference_client function
            response = inference_client(
                model_type=self.model_name,
                query=query,
                request_config=request_config,
                base_url=self.base_url,
            )

            # Extract response
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
            logger.error(f"MS-SWIFT inference_client failed: {e}")
            return LLMResponse(
                model=self.model_name,
                input=messages,
                status="error",
                error_msg=str(e),
                success=False,
            )

    def _generate_openai(self, messages: List[Dict[str, str]]) -> LLMResponse:
        """Generate using OpenAI SDK."""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                top_p=self.top_p,
                seed=self.seed,
            )

            return LLMResponse(
                model=response.model,
                input=messages,
                status="processed",
                content=response.choices[0].message.content,
                usage=TokenUsage(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    total_tokens=response.usage.total_tokens,
                ),
                success=True,
            )
        except Exception as e:
            logger.error(f"OpenAI SDK failed: {e}")
            return LLMResponse(
                model=self.model_name,
                input=messages,
                status="error",
                error_msg=str(e),
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
    Deploy MS-SWIFT server using swift.llm.run_deploy().

    Args:
        config: Configuration dict from ms_swift.yml

    Returns:
        Context manager yielding port number

    Example:
        >>> from stindex.utils.config import load_config_from_file
        >>> config = load_config_from_file("ms_swift")
        >>> with deploy_server(config) as port:
        ...     print(f"Server on port {port}")
    """
    if not SWIFT_AVAILABLE:
        raise ImportError("MS-SWIFT not installed. Install with: pip install ms-swift")

    deployment = config.get("deployment", {})
    vllm_config = deployment.get("vllm", {})

    # Build DeployArguments
    deploy_args = DeployArguments(
        model=deployment.get("model", "Qwen/Qwen3-8B"),
        infer_backend=deployment.get("infer_backend", "vllm"),
        host=deployment.get("host", "0.0.0.0"),
        port=deployment.get("port", 8000),
        verbose=deployment.get("verbose", True),
        log_interval=deployment.get("log_interval", 20),
        vllm_tensor_parallel_size=vllm_config.get("tensor_parallel_size", 1),
        vllm_gpu_memory_utilization=vllm_config.get("gpu_memory_utilization", 0.9),
        vllm_trust_remote_code=vllm_config.get("trust_remote_code", True),
        vllm_dtype=vllm_config.get("dtype", "auto"),
    )

    # Add optional parameters
    if vllm_config.get("max_model_len"):
        deploy_args.vllm_max_model_len = vllm_config["max_model_len"]
    if vllm_config.get("enable_lora"):
        deploy_args.vllm_enable_lora = True
    if deployment.get("ckpt_dir"):
        deploy_args.ckpt_dir = deployment["ckpt_dir"]

    logger.info(f"Deploying MS-SWIFT server: {deploy_args.model}")
    return run_deploy(deploy_args)
