"""
Server management utilities for automatic vLLM server and router startup.

Automatically starts backend vLLM servers and router if not already running.
Monitors activity and auto-stops servers after configured timeout.
"""

import os
import subprocess
import time
import threading
from pathlib import Path
from typing import Dict, Optional

import requests
import yaml
from loguru import logger


class ServerManager:
    """Manages vLLM backend servers and router lifecycle."""

    def __init__(self, config_path: str = "vllm"):
        """
        Initialize server manager.

        Args:
            config_path: Path to config file (default: vllm → cfg/vllm.yml)
        """
        # Load config
        config_file = Path(f"cfg/{config_path}.yml")
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")

        with open(config_file, "r") as f:
            self.config = yaml.safe_load(f)

        self.router_config = self.config.get("router", {})
        self.server_config = self.config.get("server", {})
        self.router_port = self.router_config.get("port", 8000)
        self.router_url = f"http://localhost:{self.router_port}"

        # Get enabled models
        self.enabled_models = [
            model for model in self.server_config.get("models", [])
            if model.get("enabled", False)
        ]

        # Directories
        self.project_root = Path.cwd()
        self.log_dir = self.project_root / "logs" / "vllm"
        self.pid_dir = self.log_dir / "pids"
        self.activity_file = self.log_dir / "last_activity"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.pid_dir.mkdir(parents=True, exist_ok=True)

        # Auto-stop configuration
        llm_config = self.config.get("llm", {})
        self.auto_stop_timeout = llm_config.get("auto_stop_timeout", 0)  # 0 = disabled

        # Monitoring thread
        self.monitor_thread = None
        self.stop_monitoring_event = threading.Event()

        # GPU memory allocation tracking
        self._gpu_info = self._detect_gpus()
        self._gpu_allocated_memory = {gpu_idx: 0.0 for gpu_idx in self._gpu_info.keys()}

    def _detect_gpus(self) -> Dict[int, float]:
        """
        Detect available GPUs and their memory capacity.

        Returns:
            Dict mapping GPU index to total memory in GiB
        """
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                check=True
            )
            gpu_info = {}
            for line in result.stdout.strip().split("\n"):
                if line.strip():
                    gpu_idx, memory_mb = line.strip().split(",")
                    gpu_info[int(gpu_idx.strip())] = float(memory_mb.strip()) / 1024  # Convert to GiB
            logger.info(f"Detected {len(gpu_info)} GPU(s): {gpu_info}")
            return gpu_info
        except Exception as e:
            logger.warning(f"Could not detect GPUs ({e}), assuming 1 GPU with 80 GiB")
            return {0: 80.0}

    def _estimate_model_memory(self, model_name: str, gpu_mem_utilization: float) -> float:
        """
        Estimate memory required for a model in GiB.

        Uses heuristics based on model parameter count extracted from name.
        Memory formula (approximate for FP16):
        - Base model weights: param_count_billions * 2 GB (FP16)
        - KV cache: depends on gpu_memory_utilization (vLLM allocates remaining space)
        - Overhead: ~20% for activations, buffers, etc.

        Args:
            model_name: Model name (e.g., "Qwen/Qwen3-8B", "Qwen/Qwen3-4B-Instruct-2507")
            gpu_mem_utilization: GPU memory utilization ratio (e.g., 0.7 = 70%)

        Returns:
            Estimated memory in GiB
        """
        # Extract parameter count from model name
        # Common patterns: "8B", "4B", "70B", "1.5B"
        import re
        match = re.search(r'(\d+\.?\d*)B', model_name, re.IGNORECASE)

        if match:
            param_billions = float(match.group(1))
        else:
            # Default to 7B if can't parse
            logger.warning(f"Could not extract parameter count from {model_name}, assuming 7B")
            param_billions = 7.0

        # Estimate memory (conservative formula for FP16):
        # - Model weights: param_billions * 2 GiB (FP16)
        # - KV cache + activations: estimated based on gpu_mem_utilization
        # The actual memory vLLM will use is roughly:
        #   memory_needed = model_weights / gpu_mem_utilization
        # This is because vLLM uses the remaining space for KV cache

        model_weights_gb = param_billions * 2.0  # FP16
        overhead_factor = 1.2  # 20% overhead for framework, activations, etc.
        base_memory = model_weights_gb * overhead_factor

        # Total memory needed accounting for KV cache allocation
        # vLLM allocates: (gpu_total * gpu_mem_utilization) - model_weights for KV cache
        # So we need: model_weights / (1 - kv_cache_fraction)
        # Approximation: total ≈ base_memory / gpu_mem_utilization
        estimated_memory = base_memory / gpu_mem_utilization

        logger.debug(f"Model {model_name}: {param_billions}B params → estimated {estimated_memory:.2f} GiB")
        return estimated_memory

    def _allocate_gpus(self, model_config: Dict) -> tuple[int, str]:
        """
        Allocate GPUs for a model based on memory requirements.

        Ensures no OOM by:
        1. Estimating actual model memory requirements
        2. Tracking cumulative GPU memory allocation
        3. Keeping total allocation ≤ 90% of GPU capacity
        4. Finding GPUs with sufficient free memory

        Args:
            model_config: Model configuration dict

        Returns:
            Tuple of (tensor_parallel_size, cuda_visible_devices_string)
        """
        model_name = model_config.get("name")
        gpu_mem_utilization = model_config.get("gpu_memory_utilization", 0.9)
        tensor_parallel_size = model_config.get("tensor_parallel_size", 1)

        # Manual GPU assignment via gpu_indices config
        gpu_indices = model_config.get("gpu_indices", None)
        if gpu_indices:
            cuda_visible_devices = ",".join(map(str, gpu_indices))
            # Estimate memory and track allocation
            estimated_memory_total = self._estimate_model_memory(model_name, gpu_mem_utilization)
            memory_per_gpu = estimated_memory_total / len(gpu_indices)

            for gpu_idx in gpu_indices:
                if gpu_idx in self._gpu_info:
                    self._gpu_allocated_memory[gpu_idx] += memory_per_gpu
                    logger.info(f"  GPU {gpu_idx}: +{memory_per_gpu:.2f} GiB → "
                              f"{self._gpu_allocated_memory[gpu_idx]:.2f}/{self._gpu_info[gpu_idx]:.2f} GiB "
                              f"({self._gpu_allocated_memory[gpu_idx]/self._gpu_info[gpu_idx]*100:.1f}%)")

            logger.info(f"Manual GPU assignment for {model_name}: GPUs {cuda_visible_devices}")
            return tensor_parallel_size, cuda_visible_devices

        # Automatic GPU allocation with memory awareness
        estimated_memory_total = self._estimate_model_memory(model_name, gpu_mem_utilization)

        # Enforce max 90% GPU utilization globally
        MAX_GPU_UTILIZATION = 0.9

        if tensor_parallel_size > len(self._gpu_info):
            logger.warning(
                f"Model {model_name} requires {tensor_parallel_size} GPUs, "
                f"but only {len(self._gpu_info)} available. Reducing to {len(self._gpu_info)}"
            )
            tensor_parallel_size = len(self._gpu_info)

        if tensor_parallel_size == 1:
            # Single GPU: find GPU with enough free memory
            memory_needed = estimated_memory_total
            best_gpu = None
            max_free_memory = -1

            for gpu_idx in sorted(self._gpu_info.keys()):
                total_memory = self._gpu_info[gpu_idx]
                allocated_memory = self._gpu_allocated_memory[gpu_idx]
                free_memory = total_memory - allocated_memory
                max_capacity = total_memory * MAX_GPU_UTILIZATION

                # Check if this GPU has enough free capacity
                if allocated_memory + memory_needed <= max_capacity:
                    if free_memory > max_free_memory:
                        max_free_memory = free_memory
                        best_gpu = gpu_idx

            if best_gpu is None:
                # Try to find any GPU with most free space (may OOM, but best effort)
                best_gpu = min(self._gpu_allocated_memory.keys(),
                             key=lambda k: self._gpu_allocated_memory[k])
                logger.warning(
                    f"No GPU has enough free memory for {model_name} ({memory_needed:.2f} GiB), "
                    f"using GPU {best_gpu} (may OOM)"
                )

            # Allocate memory
            self._gpu_allocated_memory[best_gpu] += memory_needed

            cuda_visible_devices = str(best_gpu)
            utilization_pct = self._gpu_allocated_memory[best_gpu] / self._gpu_info[best_gpu] * 100
            logger.info(f"Allocated GPU {best_gpu} to {model_name}")
            logger.info(f"  Estimated memory: {memory_needed:.2f} GiB")
            logger.info(f"  GPU {best_gpu}: {self._gpu_allocated_memory[best_gpu]:.2f}/"
                       f"{self._gpu_info[best_gpu]:.2f} GiB ({utilization_pct:.1f}%)")

        else:
            # Multi-GPU tensor parallel: allocate consecutive GPUs
            memory_per_gpu = estimated_memory_total / tensor_parallel_size
            allocated_gpus = []

            for gpu_idx in sorted(self._gpu_info.keys()):
                total_memory = self._gpu_info[gpu_idx]
                allocated_memory = self._gpu_allocated_memory[gpu_idx]
                max_capacity = total_memory * MAX_GPU_UTILIZATION

                # Check if GPU has capacity
                if allocated_memory + memory_per_gpu <= max_capacity:
                    allocated_gpus.append(gpu_idx)
                    self._gpu_allocated_memory[gpu_idx] += memory_per_gpu

                if len(allocated_gpus) == tensor_parallel_size:
                    break

            if len(allocated_gpus) < tensor_parallel_size:
                logger.warning(
                    f"Could not find {tensor_parallel_size} GPUs with enough capacity, "
                    f"found {len(allocated_gpus)} GPUs (may OOM)"
                )
                # Fill remaining with any available GPUs
                for gpu_idx in sorted(self._gpu_info.keys()):
                    if gpu_idx not in allocated_gpus:
                        allocated_gpus.append(gpu_idx)
                        self._gpu_allocated_memory[gpu_idx] += memory_per_gpu
                    if len(allocated_gpus) == tensor_parallel_size:
                        break

            cuda_visible_devices = ",".join(map(str, allocated_gpus))
            tensor_parallel_size = len(allocated_gpus)
            logger.info(f"Allocated GPUs {cuda_visible_devices} to {model_name} "
                       f"(tensor_parallel_size={tensor_parallel_size})")
            logger.info(f"  Estimated memory per GPU: {memory_per_gpu:.2f} GiB")
            for gpu_idx in allocated_gpus:
                utilization_pct = self._gpu_allocated_memory[gpu_idx] / self._gpu_info[gpu_idx] * 100
                logger.info(f"  GPU {gpu_idx}: {self._gpu_allocated_memory[gpu_idx]:.2f}/"
                          f"{self._gpu_info[gpu_idx]:.2f} GiB ({utilization_pct:.1f}%)")

        return tensor_parallel_size, cuda_visible_devices

    def is_router_running(self) -> bool:
        """Check if router is running."""
        try:
            response = requests.get(self.router_url, timeout=2)
            return response.status_code == 200
        except Exception:
            return False

    def is_backend_running(self, port: int) -> bool:
        """Check if a backend vLLM server is running on given port."""
        try:
            response = requests.get(f"http://localhost:{port}/health", timeout=2)
            return response.status_code == 200
        except Exception:
            return False

    def get_server_status(self) -> Dict[str, any]:
        """
        Get status of all servers.

        Returns:
            Dict with 'router' and 'backends' status
        """
        router_running = self.is_router_running()
        backends_status = {}

        for model in self.enabled_models:
            port = model.get("port")
            model_name = model.get("name")
            backends_status[model_name] = self.is_backend_running(port)

        return {
            "router": router_running,
            "backends": backends_status,
        }

    def start_backend_server(self, model_config: Dict) -> Optional[int]:
        """
        Start a single backend vLLM server.

        Args:
            model_config: Model configuration dict

        Returns:
            PID of started process, or None if failed
        """
        model_name = model_config.get("name")
        port = model_config.get("port")
        gpu_mem = model_config.get("gpu_memory_utilization", 0.9)
        trust_remote_code = model_config.get("trust_remote_code", True)

        # Get disable_custom_all_reduce from llm config
        llm_config = self.config.get("llm", {})
        disable_custom_all_reduce = llm_config.get("disable_custom_all_reduce", False)

        # Determine tensor parallel size and GPU allocation
        # If only one model is enabled, use all available GPUs
        # If multiple models, use intelligent memory-aware allocation
        if len(self.enabled_models) == 1:
            # Single model: auto-detect and use all GPUs
            try:
                gpu_list_output = subprocess.run(
                    ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                    capture_output=True,
                    text=True,
                    check=True
                )
                gpu_indices = [g.strip() for g in gpu_list_output.stdout.strip().split("\n")]
                tensor_parallel_size = len(gpu_indices)
                cuda_visible_devices = ",".join(gpu_indices)
                logger.info(f"Single model mode: auto-detected {tensor_parallel_size} GPU(s)")
                logger.info(f"  Using GPUs: {cuda_visible_devices}")
            except Exception as e:
                logger.warning(f"Could not detect GPUs ({e}), defaulting to GPU 0")
                tensor_parallel_size = 1
                cuda_visible_devices = "0"
        else:
            # Multiple models: use intelligent memory-aware GPU allocation
            tensor_parallel_size, cuda_visible_devices = self._allocate_gpus(model_config)

        # Sanitize model name for file naming
        safe_name = model_name.replace("/", "_").replace(":", "_")
        log_file = self.log_dir / f"{safe_name}.log"
        pid_file = self.pid_dir / f"{safe_name}.pid"

        logger.info(f"Starting backend: {model_name} on port {port}")
        logger.info(f"  Tensor parallel size: {tensor_parallel_size}")
        logger.info(f"  CUDA_VISIBLE_DEVICES: {cuda_visible_devices}")
        logger.info(f"  Disable custom all-reduce: {disable_custom_all_reduce}")

        # Build command
        cmd = [
            "python", "-m", "stindex.server.vllm",
            "--model", model_name,
            "--port", str(port),
            "--tensor-parallel-size", str(tensor_parallel_size),
            "--gpu-memory-utilization", str(gpu_mem),
        ]

        if trust_remote_code:
            cmd.append("--trust-remote-code")

        if disable_custom_all_reduce:
            cmd.append("--disable-custom-all-reduce")

        # Prepare environment with CUDA_VISIBLE_DEVICES
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

        # Start server in background
        try:
            with open(log_file, "w") as f:
                process = subprocess.Popen(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,  # Detach from parent
                    env=env,  # Pass environment with CUDA_VISIBLE_DEVICES
                )

            # Save PID
            with open(pid_file, "w") as f:
                f.write(str(process.pid))

            logger.info(f"  Started with PID: {process.pid}")
            logger.debug(f"  Log: {log_file}")

            return process.pid

        except Exception as e:
            logger.error(f"Failed to start backend {model_name}: {e}")
            return None

    def start_router(self) -> Optional[int]:
        """
        Start the router server.

        Returns:
            PID of started process, or None if failed
        """
        host = self.router_config.get("host", "0.0.0.0")
        port = self.router_config.get("port", 8000)

        log_file = self.log_dir / "router.log"
        pid_file = self.pid_dir / "router.pid"

        logger.info(f"Starting router on {host}:{port}")

        cmd = [
            "python", "-m", "stindex.server.router",
            "--host", host,
            "--port", str(port),
        ]

        # Start router in background
        try:
            with open(log_file, "w") as f:
                process = subprocess.Popen(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,  # Detach from parent
                )

            # Save PID
            with open(pid_file, "w") as f:
                f.write(str(process.pid))

            logger.info(f"  Started with PID: {process.pid}")
            logger.debug(f"  Log: {log_file}")

            return process.pid

        except Exception as e:
            logger.error(f"Failed to start router: {e}")
            return None

    def wait_for_backend(self, port: int, timeout: int = 120) -> bool:
        """
        Wait for backend server to be ready.

        Args:
            port: Port to check
            timeout: Maximum time to wait in seconds

        Returns:
            True if server is ready, False if timeout
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.is_backend_running(port):
                return True
            time.sleep(2)
        return False

    def wait_for_router(self, timeout: int = 60) -> bool:
        """
        Wait for router to be ready.

        Args:
            timeout: Maximum time to wait in seconds (default: 60s for vLLM import)

        Returns:
            True if router is ready, False if timeout
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.is_router_running():
                return True
            time.sleep(1)
        return False

    def ensure_servers_running(self, wait_for_ready: bool = True) -> bool:
        """
        Ensure all required servers are running, start them if not.

        Args:
            wait_for_ready: Whether to wait for servers to be fully initialized

        Returns:
            True if all servers are running, False otherwise
        """
        status = self.get_server_status()

        # Check if no servers are configured
        if not self.enabled_models:
            logger.error("No enabled models found in config. Enable at least one model in cfg/vllm.yml")
            return False

        # Start backends first
        backends_to_start = []
        for model in self.enabled_models:
            model_name = model.get("name")
            if not status["backends"].get(model_name, False):
                backends_to_start.append(model)

        if backends_to_start:
            logger.info(f"Starting {len(backends_to_start)} backend server(s)...")

            for model in backends_to_start:
                pid = self.start_backend_server(model)
                if not pid:
                    logger.error(f"Failed to start backend: {model.get('name')}")
                    return False

            if wait_for_ready:
                logger.info("Waiting for backends to initialize (this may take 1-2 minutes)...")
                for model in backends_to_start:
                    port = model.get("port")
                    model_name = model.get("name")
                    logger.info(f"  Waiting for {model_name} on port {port}...")

                    if not self.wait_for_backend(port, timeout=120):
                        logger.error(f"Backend {model_name} did not start in time")
                        logger.info(f"Check logs: {self.log_dir}/{model_name.replace('/', '_')}.log")
                        return False

                logger.info("✓ All backends ready")
        else:
            logger.info("✓ All backend servers already running")

        # Start router if not running
        if not status["router"]:
            logger.info("Starting router...")
            pid = self.start_router()
            if not pid:
                logger.error("Failed to start router")
                return False

            if wait_for_ready:
                logger.info("Waiting for router to initialize (may take ~1 minute for vLLM import)...")
                if not self.wait_for_router(timeout=60):
                    logger.error("Router did not start in time")
                    logger.info(f"Check logs: {self.log_dir}/router.log")
                    return False

                logger.info("✓ Router ready")
        else:
            logger.info("✓ Router already running")

        # Start monitoring thread if auto-stop is enabled
        if self.auto_stop_timeout > 0:
            self.start_monitoring()

        return True

    def stop_servers(self):
        """
        Stop all running servers (router + backends).

        Shutdown order (important for clean shutdown):
        1. Stop monitoring thread (prevents interference)
        2. Delete activity file (clean slate for next start)
        3. Stop router (prevents new requests)
        4. Stop backends (release GPU memory)
        """
        logger.info("Stopping all servers...")

        # Stop monitoring thread first
        self.stop_monitoring_thread()

        # Delete activity file to prevent stale timestamps
        if self.activity_file.exists():
            try:
                self.activity_file.unlink()
                logger.debug("Removed activity tracking file")
            except Exception as e:
                logger.warning(f"Could not remove activity file: {e}")

        # Stop router first
        router_pid_file = self.pid_dir / "router.pid"
        if router_pid_file.exists():
            try:
                with open(router_pid_file, "r") as f:
                    pid = int(f.read().strip())
                subprocess.run(["kill", str(pid)], check=False)
                router_pid_file.unlink()
                logger.info("✓ Router stopped")
            except Exception as e:
                logger.warning(f"Could not stop router: {e}")

        # Stop all backend servers
        for model in self.enabled_models:
            safe_name = model.get("name").replace("/", "_").replace(":", "_")
            pid_file = self.pid_dir / f"{safe_name}.pid"

            if pid_file.exists():
                try:
                    with open(pid_file, "r") as f:
                        pid = int(f.read().strip())
                    subprocess.run(["kill", str(pid)], check=False)
                    pid_file.unlink()
                    logger.info(f"✓ Stopped backend: {model.get('name')}")
                except Exception as e:
                    logger.warning(f"Could not stop backend {model.get('name')}: {e}")

        # Wait for GPU memory to be released
        time.sleep(2)
        logger.info("✓ All servers stopped")

    def _monitor_activity(self):
        """
        Monitor activity and auto-stop servers after timeout.

        Runs in background thread.
        """
        logger.info(f"Activity monitoring started (timeout: {self.auto_stop_timeout}s)")

        check_interval = min(30, self.auto_stop_timeout // 2)  # Check every 30s or half the timeout

        while not self.stop_monitoring_event.is_set():
            try:
                # Check if activity file exists and read last activity time
                if self.activity_file.exists():
                    with open(self.activity_file, "r") as f:
                        last_activity = float(f.read().strip())

                    # Calculate time since last activity
                    inactive_time = time.time() - last_activity

                    if inactive_time >= self.auto_stop_timeout:
                        logger.info(f"No activity for {inactive_time:.0f}s (timeout: {self.auto_stop_timeout}s)")
                        logger.info("Auto-stopping servers...")
                        self.stop_servers()
                        break  # Exit monitoring thread after stopping

            except Exception as e:
                logger.warning(f"Error in activity monitoring: {e}")

            # Wait before next check
            self.stop_monitoring_event.wait(check_interval)

        logger.info("Activity monitoring stopped")

    def start_monitoring(self):
        """Start activity monitoring thread."""
        if self.monitor_thread and self.monitor_thread.is_alive():
            logger.debug("Monitoring thread already running")
            return

        self.stop_monitoring_event.clear()
        self.monitor_thread = threading.Thread(
            target=self._monitor_activity,
            daemon=True,
            name="vllm-activity-monitor"
        )
        self.monitor_thread.start()
        logger.info(f"✓ Auto-stop monitoring enabled ({self.auto_stop_timeout}s timeout)")

    def stop_monitoring_thread(self):
        """Stop activity monitoring thread."""
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.stop_monitoring_event.set()
            self.monitor_thread.join(timeout=5)
            logger.info("Monitoring thread stopped")


def ensure_vllm_servers_running(config_path: str = "vllm") -> bool:
    """
    Convenience function to ensure vLLM servers are running.

    Args:
        config_path: Config file to use

    Returns:
        True if servers are running, False otherwise
    """
    manager = ServerManager(config_path)
    return manager.ensure_servers_running()
