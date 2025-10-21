"""HuggingFace LLM server with persistent model loading in GPU memory."""

import os
from typing import Any, Dict, List, Optional

import torch
from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from stindex.utils.config import load_config_from_file
from stindex.utils.tokenizer import apply_qwen3_8b_template

# Global model and tokenizer (loaded once on startup)
model: Optional[Any] = None
tokenizer: Optional[Any] = None
config: Optional[Dict] = None

app = FastAPI(
    title="STIndex HuggingFace LLM Server",
    description="HuggingFace model server with persistent GPU loading",
    version="1.0.0",
)


class GenerateRequest(BaseModel):
    """Request model for generation endpoint."""

    messages: List[Dict[str, str]]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None


class GenerateBatchRequest(BaseModel):
    """Request model for batch generation endpoint."""

    messages_batch: List[List[Dict[str, str]]]
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None


class TokenUsage(BaseModel):
    """Token usage information."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class GenerateResponse(BaseModel):
    """Response model for generation endpoint."""

    model: str
    content: str
    usage: TokenUsage
    success: bool
    error_msg: Optional[str] = None


def _format_messages(messages: List[Dict[str, str]], enable_thinking: bool = True) -> str:
    """
    Format messages into a prompt string using the tokenizer's chat template.

    Args:
        messages: List of message dicts with 'role' and 'content' keys
        enable_thinking: Whether to enable thinking mode for Qwen3 models (default: True)

    Returns:
        Formatted prompt string
    """
    global tokenizer, config

    # Try to use the tokenizer's chat template if available
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        try:
            # Check if this is a Qwen3 model
            model_name = config.get("llm", {}).get("model_name", "").lower()
            is_qwen3 = "qwen3" in model_name

            if is_qwen3:
                # Use chat template with enable_thinking for Qwen3
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking
                )
            else:
                # Use chat template without enable_thinking for other models
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            return prompt
        except Exception as e:
            logger.warning(f"Chat template failed, falling back to simple formatting: {e}")

    # Fallback: Simple formatting for compatibility
    formatted_parts = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")

        if role == "system":
            formatted_parts.append(f"System: {content}")
        elif role == "user":
            formatted_parts.append(f"User: {content}")
        elif role == "assistant":
            formatted_parts.append(f"Assistant: {content}")

    # Add assistant prompt at the end
    formatted_parts.append("Assistant:")
    return "\n\n".join(formatted_parts)


@app.on_event("startup")
async def startup_event():
    """Load model on server startup (keeps in GPU memory)."""
    global model, tokenizer, config

    # Load config from hf.yml
    config_path = os.getenv("STINDEX_CONFIG", "hf")
    logger.info(f"Loading HuggingFace model with config: {config_path}")

    try:
        config = load_config_from_file(config_path)
        llm_config = config.get("llm", {})

        model_name = llm_config.get("model_name", "Qwen/Qwen3-8B")
        device = llm_config.get("device", "auto")
        torch_dtype_str = llm_config.get("torch_dtype", "float16")

        logger.info(f"Loading model: {model_name}")
        logger.info(f"Device: {device}, dtype: {torch_dtype_str}")

        # Map dtype string to torch dtype
        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }
        torch_dtype = dtype_map.get(torch_dtype_str, torch.float16)

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=llm_config.get("trust_remote_code", False),
        )

        # Apply Qwen3-8B chat template for Qwen3 models to ensure consistent behavior
        tokenizer = apply_qwen3_8b_template(tokenizer, model_name)

        # Load model to GPU
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device,
            trust_remote_code=llm_config.get("trust_remote_code", False),
        )
        model.eval()

        # Set pad token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        logger.info(f"✓ Model loaded successfully to GPU: {model_name}")
        logger.info(f"✓ Model device: {model.device}")

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global model, tokenizer, config
    logger.info("Shutting down HuggingFace server and releasing GPU memory...")
    model = None
    tokenizer = None
    config = None


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "running",
        "service": "STIndex HuggingFace LLM Server",
        "model_loaded": model is not None,
        "model_name": config.get("llm", {}).get("model_name", "unknown") if config else "unknown",
    }


@app.get("/health")
async def health():
    """Detailed health check."""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "tokenizer_loaded": tokenizer is not None,
        "config": config.get("llm", {}) if config else None,
        "device": str(model.device) if model else None,
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """
    Generate completion from messages.

    The model is kept loaded in GPU memory across requests.
    """
    global model, tokenizer, config

    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Get enable_thinking from config (default: True for Qwen3 models)
        llm_config = config.get("llm", {})
        enable_thinking = llm_config.get("enable_thinking", True)

        # Format messages into prompt
        prompt = _format_messages(request.messages, enable_thinking=enable_thinking)
        logger.debug(f"Formatted prompt length: {len(prompt)} chars")

        # Tokenize input
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config.get("llm", {}).get("max_input_length", 4096),
        )

        # Move to device
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        input_length = inputs["input_ids"].shape[1]
        logger.debug(f"Input tokens: {input_length}")

        # Prepare generation config
        llm_config = config.get("llm", {})
        temperature = request.temperature if request.temperature is not None else llm_config.get("temperature", 0.0)
        use_sampling = temperature > 0

        gen_kwargs = {
            "max_new_tokens": request.max_tokens or llm_config.get("max_tokens", 2048),
            "do_sample": use_sampling,
            "pad_token_id": tokenizer.pad_token_id,
        }

        # Handle EOS token ID
        if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
            gen_kwargs["eos_token_id"] = tokenizer.eos_token_id

        if use_sampling:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = llm_config.get("top_p", 1.0)

        logger.info(f"Generating with {gen_kwargs['max_new_tokens']} max tokens...")

        # Generate
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)

        # Decode response (remove prompt)
        response_tokens = outputs[0][input_length:]
        generated_text = tokenizer.decode(response_tokens, skip_special_tokens=True)

        # Calculate usage
        usage = TokenUsage(
            prompt_tokens=input_length,
            completion_tokens=len(response_tokens),
            total_tokens=input_length + len(response_tokens),
        )

        return GenerateResponse(
            model=llm_config.get("model_name", "unknown"),
            content=generated_text,
            usage=usage,
            success=True,
        )

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return GenerateResponse(
            model=config.get("llm", {}).get("model_name", "unknown"),
            content="",
            usage=TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
            success=False,
            error_msg=str(e),
        )


@app.post("/generate_batch")
async def generate_batch(request: GenerateBatchRequest):
    """
    Generate completions for a batch of messages.

    The model is kept loaded in GPU memory across requests.
    """
    global model, tokenizer, config

    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Get enable_thinking from config (default: True for Qwen3 models)
        llm_config = config.get("llm", {})
        enable_thinking = llm_config.get("enable_thinking", True)

        # Format all messages
        prompts = [_format_messages(msgs, enable_thinking=enable_thinking) for msgs in request.messages_batch]

        # Tokenize with padding
        tokenizer.padding_side = "left"
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config.get("llm", {}).get("max_input_length", 4096),
        ).to(model.device)

        # Store prompt lengths
        prompt_lengths = inputs["attention_mask"].sum(dim=1).tolist()

        # Prepare generation config
        llm_config = config.get("llm", {})
        temperature = request.temperature if request.temperature is not None else llm_config.get("temperature", 0.0)
        use_sampling = temperature > 0

        gen_kwargs = {
            "max_new_tokens": request.max_tokens or llm_config.get("max_tokens", 2048),
            "do_sample": use_sampling,
            "pad_token_id": tokenizer.pad_token_id,
        }

        # Handle EOS token ID
        if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
            gen_kwargs["eos_token_id"] = tokenizer.eos_token_id

        if use_sampling:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = llm_config.get("top_p", 1.0)

        logger.info(f"Generating batch of {len(prompts)} with {gen_kwargs['max_new_tokens']} max tokens...")

        # Generate batch
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)

        # Decode completions
        responses = []
        for idx, output in enumerate(outputs):
            completion_ids = output[prompt_lengths[idx] :]
            completion = tokenizer.decode(completion_ids, skip_special_tokens=True)

            usage = TokenUsage(
                prompt_tokens=prompt_lengths[idx],
                completion_tokens=len(completion_ids),
                total_tokens=prompt_lengths[idx] + len(completion_ids),
            )

            responses.append(
                GenerateResponse(
                    model=llm_config.get("model_name", "unknown"),
                    content=completion,
                    usage=usage,
                    success=True,
                )
            )

        logger.info(f"Successfully generated {len(responses)} completions")
        return responses

    except Exception as e:
        logger.error(f"Batch generation failed: {e}")
        return [
            GenerateResponse(
                model=config.get("llm", {}).get("model_name", "unknown"),
                content="",
                usage=TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
                success=False,
                error_msg=str(e),
            )
            for _ in request.messages_batch
        ]


def start_server(host: str = "0.0.0.0", port: int = 8001):
    """Start the HuggingFace LLM server."""
    import uvicorn

    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    port = int(os.getenv("STINDEX_HF_PORT", "8001"))
    start_server(port=port)
