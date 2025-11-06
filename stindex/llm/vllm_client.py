"""vLLM client that sends requests to vLLM router with OpenAI-compatible API."""

import os
from typing import Any, Dict, List
import time
import concurrent.futures

import requests
from loguru import logger

from stindex.llm.response.models import LLMResponse, TokenUsage


class VLLMClient:
    """vLLM client that connects to router for multi-model support."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize vLLM client.

        Args:
            config: Configuration dictionary with:
                - router_url: URL of the router server (default: http://localhost:8000)
                - model_name: Model ID to use for requests (REQUIRED)
                - temperature: Sampling temperature
                - max_tokens: Maximum tokens to generate
                - auto_start: Auto-start servers if not running (default: True)
        """
        self.config = config

        # Get router URL
        self.router_url = config.get(
            "router_url",
            os.getenv("STINDEX_VLLM_ROUTER_URL", "http://localhost:8000")
        )

        # Get model name (REQUIRED for router-based architecture)
        self.model_name = config.get("model_name")
        if not self.model_name:
            # If no model specified, check if router has models available
            logger.warning(
                "No model_name specified in config. You must provide --model argument at runtime.\n"
                "Example: stindex extract 'text' --model 'Qwen/Qwen3-8B'"
            )
            # Set a placeholder - will fail if used without runtime override
            self.model_name = "UNSPECIFIED"

        # Auto-start configuration
        self.auto_start = config.get("auto_start", True)

        logger.info(f"vLLM client initialized")
        logger.info(f"  Router URL: {self.router_url}")
        logger.info(f"  Model: {self.model_name}")
        logger.info(f"  Auto-start: {self.auto_start}")

        # Verify router is reachable (and auto-start if needed)
        self._verify_router()

    def _verify_router(self):
        """Verify router is reachable and supports the requested model."""
        try:
            # Check router health
            response = requests.get(f"{self.router_url}/", timeout=5)
            if response.status_code == 200:
                info = response.json()
                logger.info(f"✓ Connected to router: {self.router_url}")
                logger.debug(f"  Available models: {info.get('available_models', [])}")

                # Check if requested model is available
                available_models = info.get('available_models', [])
                if available_models and self.model_name not in available_models:
                    logger.warning(
                        f"Model '{self.model_name}' not found in router registry.\n"
                        f"  Available models: {available_models}\n"
                        f"  Check cfg/vllm.yml router.models configuration"
                    )
            else:
                logger.warning(f"Router returned status {response.status_code}")
        except Exception as e:
            if self.auto_start:
                logger.warning(f"Router not responding: {e}")
                logger.info("Auto-starting vLLM servers...")
                self._auto_start_servers()
            else:
                logger.error(f"Could not connect to vLLM router at {self.router_url}: {e}")
                logger.warning("Make sure the router is running:")
                logger.warning("  python -m stindex.server.router")
                logger.info("Or enable auto-start with 'auto_start: true' in config")

    def _auto_start_servers(self):
        """Auto-start vLLM backend servers and router."""
        try:
            from stindex.server.server_manager import ensure_vllm_servers_running

            logger.info("=" * 60)
            logger.info("Auto-starting vLLM servers (this may take 1-2 minutes)...")
            logger.info("=" * 60)

            success = ensure_vllm_servers_running()

            if success:
                logger.info("=" * 60)
                logger.info("✓ Servers started successfully")
                logger.info("=" * 60)
            else:
                logger.error("Failed to auto-start servers")
                logger.info("Try starting manually: bash scripts/start_vllm_servers.sh")
                raise RuntimeError("Server auto-start failed")

        except ImportError as e:
            logger.error(f"Could not import server_manager: {e}")
            raise RuntimeError("Server auto-start unavailable")

    def generate(self, messages: List[Dict[str, str]]) -> LLMResponse:
        """
        Generate completion from messages by sending request to vLLM router.

        Args:
            messages: List of message dicts with 'role' and 'content' keys

        Returns:
            LLMResponse with standardized structure
        """
        try:
            # Prepare OpenAI-compatible payload with model field
            payload = {
                "model": self.model_name,  # REQUIRED for router
                "messages": messages,
                "max_tokens": self.config.get("max_tokens", 2048),
                "temperature": self.config.get("temperature", 0.7),
            }

            # Send request to router's OpenAI-compatible endpoint
            response = requests.post(
                f"{self.router_url}/v1/chat/completions",
                json=payload,
                timeout=300,  # 5 minute timeout
            )
            response.raise_for_status()

            # Parse OpenAI-compatible response
            result = response.json()

            # Extract content from OpenAI format
            content = result["choices"][0]["message"]["content"]
            usage = result.get("usage", {})

            return LLMResponse(
                model=result.get("model", self.model_name),
                input=messages,
                status="processed",
                content=content,
                usage=TokenUsage(
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    total_tokens=usage.get("total_tokens", 0),
                ),
                success=True,
            )

        except requests.exceptions.ConnectionError as e:
            logger.error(f"Could not connect to vLLM router at {self.router_url}: {e}")
            return LLMResponse(
                model=self.model_name,
                input=messages,
                status="error",
                error_msg=f"Router unavailable: {str(e)}",
                success=False,
            )

        except Exception as e:
            logger.error(f"vLLM client request failed: {str(e)}")
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
        """
        Generate completions for a batch of messages.
        Processes requests in parallel to the router.

        Args:
            messages_batch: List of message lists
            max_tokens: Override max_tokens from config
            temperature: Override temperature from config

        Returns:
            List of LLMResponse objects
        """
        # Process in parallel using thread pool
        def process_single(messages):
            # Override config temporarily for this request
            old_max_tokens = self.config.get("max_tokens")
            old_temperature = self.config.get("temperature")

            if max_tokens is not None:
                self.config["max_tokens"] = max_tokens
            if temperature is not None:
                self.config["temperature"] = temperature

            result = self.generate(messages)

            # Restore config
            self.config["max_tokens"] = old_max_tokens
            self.config["temperature"] = old_temperature

            return result

        # Use thread pool to send concurrent requests
        # Router handles load balancing across backends
        max_workers = min(32, len(messages_batch))
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(process_single, messages_batch))

        return results
