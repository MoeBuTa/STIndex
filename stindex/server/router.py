"""
Multi-Model Router Server for STIndex.

This FastAPI server acts as a unified endpoint that routes requests to
different vLLM backend servers based on model name.

Architecture:
    Client → Router (port 8000) → vLLM Backend 1 (port 8001) [Qwen3-8B]
                                → vLLM Backend 2 (port 8002) [Qwen3-4B-Thinking]

Usage:
    python -m stindex.server.router
    # Or with explicit config:
    python -m stindex.server.router --config vllm
"""

import argparse
from typing import List, Dict, Optional, Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import httpx
import uvicorn
from loguru import logger


# ============================================================================
# Request/Response Models (OpenAI-compatible)
# ============================================================================

class Message(BaseModel):
    role: str = Field(..., description="Message role (system, user, assistant)")
    content: str = Field(..., description="Message content")


class ChatCompletionRequest(BaseModel):
    model: str = Field(..., description="Model name to use")
    messages: List[Message] = Field(..., description="List of messages")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, ge=1, le=32768)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    top_k: int = Field(default=-1, ge=-1)
    stream: bool = Field(default=False)


class ChatCompletionResponse(BaseModel):
    id: str
    model: str
    choices: List[Dict[str, Any]]
    usage: Dict[str, int]


# ============================================================================
# Model Registry
# ============================================================================

class ModelRegistry:
    """Registry mapping model names to backend vLLM servers."""

    def __init__(self, model_configs: Dict[str, str]):
        """
        Initialize model registry.

        Args:
            model_configs: Dict mapping model names to backend URLs
                Example: {
                    "Qwen/Qwen3-Coder-30B-A3B-Instruct": "http://localhost:8001",
                    "Qwen/Qwen3-4B-Thinking-2507": "http://localhost:8002"
                }
        """
        self.model_to_backend = model_configs
        self.http_client = httpx.AsyncClient(timeout=300.0)

        logger.info("Model Registry initialized with:")
        for model, backend in model_configs.items():
            logger.info(f"  - {model} → {backend}")

    def get_backend_url(self, model_name: str) -> str:
        """Get backend URL for a model name."""
        if model_name not in self.model_to_backend:
            raise ValueError(
                f"Model '{model_name}' not found. "
                f"Available models: {list(self.model_to_backend.keys())}"
            )
        return self.model_to_backend[model_name]

    def list_models(self) -> List[str]:
        """List all available models."""
        return list(self.model_to_backend.keys())

    async def forward_request(
        self,
        model_name: str,
        request: ChatCompletionRequest
    ) -> ChatCompletionResponse:
        """
        Forward request to appropriate backend vLLM server.

        Args:
            model_name: Name of model to use
            request: Chat completion request

        Returns:
            ChatCompletionResponse from backend
        """
        backend_url = self.get_backend_url(model_name)

        # Prepare request data
        request_data = {
            "messages": [{"role": msg.role, "content": msg.content} for msg in request.messages],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "top_p": request.top_p,
            "top_k": request.top_k,
            "stream": request.stream,
        }

        logger.debug(f"Forwarding request to {backend_url} for model {model_name}")

        try:
            # Forward to backend vLLM server
            response = await self.http_client.post(
                f"{backend_url}/v1/chat/completions",
                json=request_data,
                timeout=300.0,
            )

            if response.status_code != 200:
                error_msg = f"Backend returned status {response.status_code}: {response.text}"
                logger.error(error_msg)
                raise HTTPException(status_code=response.status_code, detail=error_msg)

            # Return backend response
            return response.json()

        except httpx.RequestError as e:
            error_msg = f"Error connecting to backend {backend_url}: {str(e)}"
            logger.error(error_msg)
            raise HTTPException(status_code=503, detail=error_msg)


# ============================================================================
# FastAPI App
# ============================================================================

# Global registry
registry: Optional[ModelRegistry] = None

app = FastAPI(
    title="STIndex Multi-Model Router",
    description="Unified endpoint for multiple vLLM backend servers",
    version="0.1.0",
)


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "running",
        "service": "multi-model-router",
        "available_models": registry.list_models() if registry else []
    }


@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI-compatible)."""
    if registry is None:
        raise HTTPException(status_code=503, detail="Registry not initialized")

    models = [
        {
            "id": model_name,
            "object": "model",
            "created": 0,
            "owned_by": "stindex"
        }
        for model_name in registry.list_models()
    ]

    return {
        "object": "list",
        "data": models
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest) -> ChatCompletionResponse:
    """
    Chat completions endpoint (OpenAI-compatible).

    Routes request to appropriate backend based on model name.
    """
    if registry is None:
        raise HTTPException(status_code=503, detail="Registry not initialized")

    try:
        # Forward request to backend
        response = await registry.forward_request(request.model, request)
        return response

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Multi-Model Router Server for STIndex")

    # Configuration
    parser.add_argument("--config", type=str, default="vllm",
                        help="Config name to load (default: vllm, loads cfg/vllm.yml)")
    parser.add_argument("--host", type=str, default=None,
                        help="Server host (overrides config)")
    parser.add_argument("--port", type=int, default=None,
                        help="Server port (overrides config)")

    # Manual model configuration (alternative to config file)
    parser.add_argument("--models", type=str, nargs="+",
                        help="Model names (e.g., Qwen/Qwen3-8B)")
    parser.add_argument("--backends", type=str, nargs="+",
                        help="Backend URLs (e.g., http://localhost:8001)")

    args = parser.parse_args()

    # Load model configurations
    model_configs = {}
    host = args.host or "0.0.0.0"
    port = args.port or 8000

    if args.models and args.backends:
        # Manual configuration via CLI
        if len(args.models) != len(args.backends):
            raise ValueError("Number of models must match number of backends")
        model_configs = dict(zip(args.models, args.backends))
    else:
        # Load from config file (default: vllm.yml)
        from stindex.utils.config import load_config_from_file
        config = load_config_from_file(args.config)

        # Get router configuration
        router_config = config.get("router", {})
        model_configs = router_config.get("models", {})

        # Override host/port if not provided via CLI
        if args.host is None:
            host = router_config.get("host", "0.0.0.0")
        if args.port is None:
            port = router_config.get("port", 8000)

    if not model_configs:
        raise ValueError(
            "No models configured. Either:\n"
            "  1. Provide --models and --backends arguments, OR\n"
            "  2. Configure models in cfg/vllm.yml under 'router.models'"
        )

    # Initialize registry
    global registry
    registry = ModelRegistry(model_configs)

    logger.info("=" * 80)
    logger.info("STIndex Multi-Model Router Server")
    logger.info("=" * 80)
    logger.info(f"Router listening on {host}:{port}")
    logger.info(f"Configured models: {len(model_configs)}")
    for model, backend in model_configs.items():
        logger.info(f"  • {model}")
        logger.info(f"    → {backend}")
    logger.info("=" * 80)

    # Run server
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
