"""HuggingFace LLM client that sends requests to HF server."""

import os
from typing import Any, Dict, List
import random
import time

import requests
from loguru import logger

from stindex.llm.response.models import LLMResponse, TokenUsage


class HuggingFaceLLM:
    """HuggingFace LLM client with load balancing across multiple servers."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize HuggingFace LLM client.

        Args:
            config: Configuration dictionary with:
                - base_port: Starting port number (default: 8001)
                - num_gpus: Number of GPUs/servers ("auto" or integer, default: "auto")
                - load_balancing: "round_robin" or "random" (default: "round_robin")
                - model_name: HuggingFace model ID (for reference only)
                - temperature: Sampling temperature
                - max_tokens: Maximum tokens to generate

                Legacy support (deprecated):
                - server_url: Single server URL (e.g., "http://localhost:8001")
                - server_urls: List of server URLs for load balancing
        """
        self.config = config

        # Get server URLs - support both new dynamic detection and legacy hardcoded URLs
        server_urls = config.get("server_urls")
        if server_urls:
            # Legacy mode: hardcoded URLs (deprecated)
            self.server_urls = [url.rstrip("/") for url in server_urls]
            self.multi_server = len(self.server_urls) > 1
            logger.warning("Using legacy hardcoded server_urls (deprecated). Consider using base_port + num_gpus instead.")
            logger.info(f"HuggingFace client initialized with {len(self.server_urls)} servers")
        elif config.get("server_url"):
            # Legacy mode: single server URL (deprecated)
            server_url = config.get("server_url").rstrip("/")
            self.server_urls = [server_url]
            self.multi_server = False
            logger.warning("Using legacy server_url (deprecated). Consider using base_port + num_gpus instead.")
            logger.info(f"HuggingFace client initialized, server: {server_url}")
        else:
            # New dynamic detection mode
            base_port = config.get("base_port", int(os.getenv("STINDEX_HF_BASE_PORT", "8001")))
            num_gpus = config.get("num_gpus", os.getenv("STINDEX_NUM_GPUS", "auto"))

            # Auto-detect servers by probing ports
            if num_gpus == "auto":
                self.server_urls = self._auto_detect_servers(base_port)
            else:
                # Use specified number of GPUs
                try:
                    num_gpus = int(num_gpus)
                    self.server_urls = [f"http://localhost:{base_port + i}" for i in range(num_gpus)]
                except (ValueError, TypeError):
                    logger.error(f"Invalid num_gpus value: {num_gpus}, defaulting to auto-detect")
                    self.server_urls = self._auto_detect_servers(base_port)

            self.multi_server = len(self.server_urls) > 1
            logger.info(f"HuggingFace client initialized with {len(self.server_urls)} server(s) starting at port {base_port}")

        # Load balancing strategy
        self.load_balancing = config.get("load_balancing", "round_robin")
        self.current_server_idx = 0
        self.server_health = {url: True for url in self.server_urls}

        # Verify servers are reachable
        self._verify_servers()

    def _auto_detect_servers(self, base_port: int, max_ports: int = 8) -> List[str]:
        """
        Auto-detect available servers by probing ports.

        Args:
            base_port: Starting port to probe
            max_ports: Maximum number of ports to check

        Returns:
            List of server URLs that responded to health check
        """
        detected_servers = []

        for i in range(max_ports):
            port = base_port + i
            url = f"http://localhost:{port}"

            try:
                response = requests.get(f"{url}/health", timeout=2)
                if response.status_code == 200:
                    detected_servers.append(url)
                    logger.debug(f"Detected server at {url}")
                else:
                    # Stop checking after first non-responsive port
                    break
            except Exception:
                # Stop checking after first unreachable port
                break

        if not detected_servers:
            # Fallback to single server on base port if none detected
            logger.warning(f"No servers detected, falling back to {base_port}")
            detected_servers = [f"http://localhost:{base_port}"]

        return detected_servers

    def _verify_servers(self):
        """Verify all servers are reachable and update health status."""
        for url in self.server_urls:
            try:
                response = requests.get(f"{url}/health", timeout=5)
                if response.status_code == 200:
                    health = response.json()
                    # The config field contains the llm section directly
                    model_name = health.get('config', {}).get('model_name', 'unknown') if health.get('config') else 'unknown'
                    logger.info(f"✓ Connected to HF server: {url} ({model_name})")
                    self.server_health[url] = True
                else:
                    logger.warning(f"HF server {url} returned status {response.status_code}")
                    self.server_health[url] = False
            except Exception as e:
                logger.warning(f"Could not connect to HF server {url}: {e}")
                self.server_health[url] = False

        # Check if any server is healthy
        healthy_servers = [url for url, healthy in self.server_health.items() if healthy]
        if not healthy_servers:
            logger.error("No healthy HF servers available!")
            logger.warning("Make sure servers are running:")
            logger.warning("  Single server:   ./scripts/start_server.sh")
            logger.warning("  Multi-GPU:       ./scripts/start_multi_gpu_servers.sh")

    def _get_next_server(self) -> str:
        """Get next server URL using load balancing strategy."""
        # Filter healthy servers
        healthy_servers = [url for url, healthy in self.server_health.items() if healthy]

        if not healthy_servers:
            # Try to revive servers
            logger.warning("No healthy servers, attempting to reconnect...")
            self._verify_servers()
            healthy_servers = [url for url, healthy in self.server_health.items() if healthy]

            if not healthy_servers:
                # Fallback to first server
                logger.error("No healthy servers available, using first server as fallback")
                return self.server_urls[0]

        # Apply load balancing strategy
        if self.load_balancing == "random":
            return random.choice(healthy_servers)
        else:  # round_robin
            # Find next healthy server
            attempts = 0
            while attempts < len(self.server_urls):
                url = self.server_urls[self.current_server_idx]
                self.current_server_idx = (self.current_server_idx + 1) % len(self.server_urls)

                if url in healthy_servers:
                    return url
                attempts += 1

            # Fallback
            return healthy_servers[0]

    def _mark_server_unhealthy(self, server_url: str):
        """Mark a server as unhealthy after a failed request."""
        self.server_health[server_url] = False
        logger.warning(f"Marked server {server_url} as unhealthy")

    def generate(self, messages: List[Dict[str, str]]) -> LLMResponse:
        """
        Generate completion from messages by sending request to HF server.
        Automatically retries with different servers on failure.

        Args:
            messages: List of message dicts with 'role' and 'content' keys

        Returns:
            LLMResponse with standardized structure
        """
        # Try up to 3 times with different servers
        max_retries = min(3, len(self.server_urls))

        for attempt in range(max_retries):
            server_url = self._get_next_server()

            try:
                # Prepare request payload
                payload = {
                    "messages": messages,
                    "max_tokens": self.config.get("max_tokens"),
                    "temperature": self.config.get("temperature"),
                }

                # Send request to server
                response = requests.post(
                    f"{server_url}/generate",
                    json=payload,
                    timeout=300,  # 5 minute timeout for generation
                )
                response.raise_for_status()

                # Parse response
                result = response.json()

                if result.get("success"):
                    return LLMResponse(
                        model=result.get("model", self.config.get("model_name", "unknown")),
                        input=messages,
                        status="processed",
                        content=result.get("content", ""),
                        usage=TokenUsage(**result.get("usage", {})),
                        success=True,
                    )
                else:
                    error_msg = result.get("error_msg", "Unknown error")
                    logger.error(f"HF server generation failed: {error_msg}")
                    # Don't retry if the generation itself failed
                    return LLMResponse(
                        model=result.get("model", self.config.get("model_name", "unknown")),
                        input=messages,
                        status="error",
                        error_msg=error_msg,
                        success=False,
                    )

            except requests.exceptions.ConnectionError as e:
                logger.error(f"Could not connect to HF server at {server_url}: {e}")
                self._mark_server_unhealthy(server_url)

                if attempt < max_retries - 1:
                    logger.info(f"Retrying with different server...")
                    time.sleep(1)
                    continue
                else:
                    return LLMResponse(
                        model=self.config.get("model_name", "unknown"),
                        input=messages,
                        status="error",
                        error_msg=f"All servers unavailable. Last error: {str(e)}",
                        success=False,
                    )

            except Exception as e:
                logger.error(f"HuggingFace client request failed: {str(e)}")

                if attempt < max_retries - 1:
                    logger.info(f"Retrying with different server...")
                    time.sleep(1)
                    continue
                else:
                    return LLMResponse(
                        model=self.config.get("model_name", "unknown"),
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
        In multi-server mode, distributes batch across servers for parallel processing.

        Args:
            messages_batch: List of message lists
            max_tokens: Override max_tokens from config
            temperature: Override temperature from config

        Returns:
            List of LLMResponse objects
        """
        # If multiple servers available, distribute batch across them
        if self.multi_server and len(self.server_urls) > 1:
            return self._generate_batch_distributed(messages_batch, max_tokens, temperature)
        else:
            return self._generate_batch_single(messages_batch, max_tokens, temperature)

    def _generate_batch_distributed(
        self,
        messages_batch: List[List[Dict[str, str]]],
        max_tokens: int = None,
        temperature: float = None,
    ) -> List[LLMResponse]:
        """
        Distribute batch across multiple servers for parallel processing.
        """
        import concurrent.futures

        # Get healthy servers
        healthy_servers = [url for url, healthy in self.server_health.items() if healthy]

        if not healthy_servers:
            logger.warning("No healthy servers for distributed batch, falling back to single server")
            return self._generate_batch_single(messages_batch, max_tokens, temperature)

        # Split batch across servers
        num_servers = len(healthy_servers)
        batch_size = len(messages_batch)
        chunks_per_server = (batch_size + num_servers - 1) // num_servers

        # Create sub-batches
        sub_batches = []
        for i in range(0, batch_size, chunks_per_server):
            sub_batch = messages_batch[i:i + chunks_per_server]
            server_url = healthy_servers[len(sub_batches) % num_servers]
            sub_batches.append((server_url, sub_batch, i))

        # Process sub-batches in parallel
        results = [None] * batch_size

        def process_sub_batch(args):
            server_url, sub_batch, start_idx = args
            try:
                payload = {
                    "messages_batch": sub_batch,
                    "max_tokens": max_tokens or self.config.get("max_tokens"),
                    "temperature": temperature if temperature is not None else self.config.get("temperature"),
                }

                response = requests.post(
                    f"{server_url}/generate_batch",
                    json=payload,
                    timeout=600,
                )
                response.raise_for_status()

                sub_results = response.json()

                # Convert to LLMResponse objects
                responses = []
                for idx, result in enumerate(sub_results):
                    if result.get("success"):
                        responses.append(
                            LLMResponse(
                                model=result.get("model", self.config.get("model_name", "unknown")),
                                input=sub_batch[idx],
                                status="processed",
                                content=result.get("content", ""),
                                usage=TokenUsage(**result.get("usage", {})),
                                success=True,
                            )
                        )
                    else:
                        error_msg = result.get("error_msg", "Unknown error")
                        responses.append(
                            LLMResponse(
                                model=result.get("model", self.config.get("model_name", "unknown")),
                                input=sub_batch[idx],
                                status="error",
                                error_msg=error_msg,
                                success=False,
                            )
                        )

                return start_idx, responses

            except Exception as e:
                logger.error(f"Sub-batch processing failed on {server_url}: {e}")
                self._mark_server_unhealthy(server_url)
                # Return error responses
                error_responses = [
                    LLMResponse(
                        model=self.config.get("model_name", "unknown"),
                        input=messages,
                        status="error",
                        error_msg=str(e),
                        success=False,
                    )
                    for messages in sub_batch
                ]
                return start_idx, error_responses

        # Execute in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_servers) as executor:
            futures = [executor.submit(process_sub_batch, args) for args in sub_batches]

            for future in concurrent.futures.as_completed(futures):
                start_idx, sub_results = future.result()
                for i, result in enumerate(sub_results):
                    results[start_idx + i] = result

            return results

    def _generate_batch_single(
        self,
        messages_batch: List[List[Dict[str, str]]],
        max_tokens: int = None,
        temperature: float = None,
    ) -> List[LLMResponse]:
        """
        Generate batch using a single server.
        """
        server_url = self._get_next_server()

        try:
            # Prepare request payload
            payload = {
                "messages_batch": messages_batch,
                "max_tokens": max_tokens or self.config.get("max_tokens"),
                "temperature": temperature if temperature is not None else self.config.get("temperature"),
            }

            # Send request to server
            response = requests.post(
                f"{server_url}/generate_batch",
                json=payload,
                timeout=600,  # 10 minute timeout for batch
            )
            response.raise_for_status()

            # Parse response
            results = response.json()

            # Convert to LLMResponse objects
            responses = []
            for idx, result in enumerate(results):
                if result.get("success"):
                    responses.append(
                        LLMResponse(
                            model=result.get("model", self.config.get("model_name", "unknown")),
                            input=messages_batch[idx],
                            status="processed",
                            content=result.get("content", ""),
                            usage=TokenUsage(**result.get("usage", {})),
                            success=True,
                        )
                    )
                else:
                    error_msg = result.get("error_msg", "Unknown error")
                    responses.append(
                        LLMResponse(
                            model=result.get("model", self.config.get("model_name", "unknown")),
                            input=messages_batch[idx],
                            status="error",
                            error_msg=error_msg,
                            success=False,
                        )
                    )

            return responses

        except requests.exceptions.ConnectionError as e:
            error_msg = f"Could not connect to HF server at {server_url}. Is the server running?"
            logger.error(error_msg)
            # Return error responses for all messages
            return [
                LLMResponse(
                    model=self.config.get("model_name", "unknown"),
                    input=messages,
                    status="error",
                    error_msg=error_msg,
                    success=False,
                )
                for messages in messages_batch
            ]
        except Exception as e:
            logger.error(f"HuggingFace batch client request failed: {str(e)}")
            # Return error responses for all messages
            return [
                LLMResponse(
                    model=self.config.get("model_name", "unknown"),
                    input=messages,
                    status="error",
                    error_msg=str(e),
                    success=False,
                )
                for messages in messages_batch
            ]
