"""
vLLM FastAPI Server for STIndex.

This server hosts LLM models using vLLM for efficient inference.
Provides OpenAI-compatible API endpoints for chat completions.

Usage:
    # Start server with specific model
    python -m stindex.server.vllm_server --model Qwen/Qwen3-8B --port 8001

    # Or use config from vllm.yml (reads vllm.models section)
    python -m stindex.server.vllm_server --config-name qwen3-8b
"""

import argparse
import os
from typing import List, Dict, Optional, Any
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn
from loguru import logger

# vLLM imports
try:
    from vllm import AsyncLLMEngine
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.sampling_params import SamplingParams
    from vllm.utils import random_uuid
except ImportError:
    logger.error("vLLM not installed. Install with: pip install vllm")
    raise


# ============================================================================
# Request/Response Models
# ============================================================================

class Message(BaseModel):
    role: str = Field(..., description="Message role (system, user, assistant)")
    content: str = Field(..., description="Message content")


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = Field(default=None, description="Model name (optional)")
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
# Global Engine and Tokenizer
# ============================================================================

engine: Optional[AsyncLLMEngine] = None
engine_args: Optional[AsyncEngineArgs] = None
tokenizer = None  # Store tokenizer for chat template
model_name: str = ""


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI."""
    # Startup
    logger.info("Starting vLLM engine...")
    global engine, tokenizer

    engine = AsyncLLMEngine.from_engine_args(engine_args)

    # Load tokenizer for chat template
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=engine_args.trust_remote_code,
    )

    logger.info(f"✓ vLLM engine initialized with model: {model_name}")
    logger.info(f"✓ Tokenizer loaded for chat template")

    yield

    # Shutdown
    logger.info("Shutting down vLLM engine...")


# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(
    title="STIndex vLLM Server",
    description="FastAPI server for vLLM inference with OpenAI-compatible API",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "status": "running",
        "model": model_name,
        "service": "stindex-vllm"
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model": model_name,
        "engine_ready": engine is not None,
    }


@app.get("/v1/models")
async def list_models():
    """List available models (OpenAI-compatible)."""
    return {
        "object": "list",
        "data": [
            {
                "id": model_name,
                "object": "model",
                "created": 0,
                "owned_by": "stindex"
            }
        ]
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest) -> ChatCompletionResponse:
    """
    Chat completions endpoint (OpenAI-compatible).

    This endpoint accepts chat messages and returns completions using vLLM.
    """
    if engine is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    request_id = random_uuid()

    try:
        # Convert messages to prompt using tokenizer's chat template
        messages_dict = [{"role": msg.role, "content": msg.content} for msg in request.messages]

        # Apply chat template to convert messages to prompt
        prompt = tokenizer.apply_chat_template(
            messages_dict,
            tokenize=False,
            add_generation_prompt=True
        )

        # Create sampling params
        sampling_params = SamplingParams(
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            top_p=request.top_p,
            top_k=request.top_k if request.top_k > 0 else -1,
        )

        # Generate using vLLM with the formatted prompt
        results_generator = engine.generate(
            prompt=prompt,
            sampling_params=sampling_params,
            request_id=request_id,
        )

        # For non-streaming, wait for completion
        final_output = None
        async for request_output in results_generator:
            final_output = request_output

        if final_output is None:
            raise HTTPException(status_code=500, detail="No output generated")

        # Extract completion
        output = final_output.outputs[0]
        completion_text = output.text

        # Calculate token usage
        prompt_tokens = len(final_output.prompt_token_ids)
        completion_tokens = len(output.token_ids)

        # Build response in OpenAI format
        response = ChatCompletionResponse(
            id=request_id,
            model=model_name,
            choices=[
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": completion_text,
                    },
                    "finish_reason": output.finish_reason,
                }
            ],
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }
        )

        return response

    except Exception as e:
        logger.error(f"Error during generation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="vLLM FastAPI Server for STIndex")

    # Model configuration
    parser.add_argument("--model", type=str, default=None,
                        help="Model name or path (HuggingFace ID)")
    parser.add_argument("--config-name", type=str, default=None,
                        help="Model config name from vllm.yml (e.g., 'qwen3-8b')")

    # vLLM configuration
    parser.add_argument("--tensor-parallel-size", type=int, default=None,
                        help="Number of GPUs for tensor parallelism (auto-detected if not set)")
    parser.add_argument("--dtype", type=str, default="auto",
                        choices=["auto", "float16", "bfloat16", "float32"],
                        help="Model dtype")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9,
                        help="GPU memory utilization (0.0-1.0)")
    parser.add_argument("--max-model-len", type=int, default=None,
                        help="Maximum model context length")
    parser.add_argument("--max-num-seqs", type=int, default=256,
                        help="Maximum number of sequences to process in parallel")
    parser.add_argument("--trust-remote-code", action="store_true",
                        help="Trust remote code")
    parser.add_argument("--disable-custom-all-reduce", action="store_true",
                        help="Disable vLLM custom all-reduce (use standard NCCL)")

    # Server configuration
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Server host")
    parser.add_argument("--port", type=int, default=8001,
                        help="Server port")

    args = parser.parse_args()

    # Determine model configuration source
    if args.config_name:
        # Load from vllm.yml
        from stindex.utils.config import load_config_from_file
        config = load_config_from_file("vllm")

        # Find matching model in vllm.models
        vllm_models = config.get("server", {}).get("models", [])
        model_config = None
        for model in vllm_models:
            # Match by name or simplified name
            model_id = model.get("name", "").lower()
            if args.config_name.lower() in model_id or model_id.endswith(args.config_name.lower()):
                model_config = model
                break

        if not model_config:
            logger.error(f"Model config '{args.config_name}' not found in vllm.yml")
            exit(1)

        # Apply config
        args.model = model_config.get("name")
        args.port = model_config.get("port", args.port)
        args.gpu_memory_utilization = model_config.get("gpu_memory_utilization", args.gpu_memory_utilization)
        args.trust_remote_code = model_config.get("trust_remote_code", args.trust_remote_code)

        # Get disable_custom_all_reduce from llm config
        llm_config = config.get("llm", {})
        if "disable_custom_all_reduce" in llm_config:
            args.disable_custom_all_reduce = llm_config.get("disable_custom_all_reduce", False)

        logger.info(f"Loaded config for model: {args.model}")

    elif args.model is None:
        logger.error("Must specify either --model or --config-name")
        exit(1)

    # Auto-detect tensor parallel size from CUDA_VISIBLE_DEVICES if not set
    if args.tensor_parallel_size is None:
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if cuda_visible:
            # Count GPUs from CUDA_VISIBLE_DEVICES
            gpus = [g.strip() for g in cuda_visible.split(",") if g.strip()]
            args.tensor_parallel_size = len(gpus)
        else:
            # Default to 1 GPU
            args.tensor_parallel_size = 1

        logger.info(f"Auto-detected tensor_parallel_size: {args.tensor_parallel_size}")

    # Set global model name
    global model_name, engine_args
    model_name = args.model

    logger.info("=" * 80)
    logger.info("STIndex vLLM Server")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model}")
    logger.info(f"Tensor Parallel Size: {args.tensor_parallel_size}")
    logger.info(f"Dtype: {args.dtype}")
    logger.info(f"GPU Memory Utilization: {args.gpu_memory_utilization}")
    logger.info(f"Max Sequences: {args.max_num_seqs}")
    logger.info(f"Disable Custom All-Reduce: {args.disable_custom_all_reduce}")
    logger.info(f"Host: {args.host}:{args.port}")
    logger.info("=" * 80)

    # Create engine args
    engine_args = AsyncEngineArgs(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        disable_custom_all_reduce=args.disable_custom_all_reduce,
        # Enable for better performance
        enable_chunked_prefill=False,
    )

    # Run server
    logger.info("Starting FastAPI server...")
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
