#!/usr/bin/env python3
"""
server.py — minimal OpenAI-compatible server for LiteRT LM.

This is intentionally simple:
  - loads one local LiteRT LM model
  - exposes /v1/health, /v1/models, /v1/chat/completions
  - accepts OpenAI-style chat messages
  - returns OpenAI-style responses
  - supports basic SSE streaming
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, List, Optional

import litert_lm
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

HF_REPO = "litert-community/gemma-4-E2B-it-litert-lm"
HF_FILENAME = "gemma-4-E2B-it.litertlm"
DEFAULT_MODEL_ID = os.environ.get("MODEL_ID", "gemma-4-e2b-it")
DEFAULT_SYSTEM_PROMPT = os.environ.get(
    "SYSTEM_PROMPT",
    "You are a helpful assistant.",
)

engine: Optional[litert_lm.Engine] = None


def resolve_model_path() -> str:
    path = os.environ.get("MODEL_PATH", "").strip()
    if path:
        return path

    from huggingface_hub import hf_hub_download

    print(f"Downloading {HF_REPO}/{HF_FILENAME} (first run only)...")
    return hf_hub_download(repo_id=HF_REPO, filename=HF_FILENAME)


MODEL_PATH = resolve_model_path()


def load_engine() -> None:
    global engine
    if engine is not None:
        return
    print(f"Loading model from {MODEL_PATH}...")
    # engine = litert_lm.Engine(MODEL_PATH, backend=litert_lm.Backend.GPU)
    engine = litert_lm.Engine(MODEL_PATH, backend=litert_lm.Backend.CPU)
    engine.__enter__()
    print("Model loaded.")


def _cors_origins_from_env() -> List[str]:
    raw = os.environ.get("SERVER_CORS_ALLOW_ORIGINS", "*").strip()
    if not raw or raw == "*":
        return ["*"]
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def _flatten_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") in {"text", "input_text", "output_text"} and isinstance(
                    item.get("text"), str
                ):
                    parts.append(item["text"])
                elif isinstance(item.get("content"), str):
                    parts.append(item["content"])
        return "\n".join(part for part in parts if part.strip())
    return ""


def _normalize_messages(messages: Any) -> List[Dict[str, Any]]:
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")

    normalized: List[Dict[str, Any]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"message {index} must be an object")
        role = message.get("role")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"message {index} has invalid role: {role}")
        text = _flatten_content(message.get("content"))
        if not text.strip():
            continue
        normalized.append({"role": role, "content": [{"type": "text", "text": text}]})

    if not normalized:
        raise ValueError("messages must contain text content")
    return normalized


def _serialize_response(
    text: str,
    model: str,
    completion_id: Optional[str] = None,
) -> Dict[str, Any]:
    completion_id = completion_id or f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": text,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


def _chunk_words(text: str) -> List[str]:
    words = text.split()
    if not words:
        return [""]
    return [word + (" " if i < len(words) - 1 else "") for i, word in enumerate(words)]


def _serialize_stream_chunk(
    piece: str,
    model: str,
    completion_id: str,
    role: Optional[str] = None,
    finish_reason: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": {
                    **({"role": role} if role is not None else {}),
                    **({"content": piece} if piece else {}),
                },
                "finish_reason": finish_reason,
            }
        ],
    }


def generate_completion(messages: List[Dict[str, Any]]) -> str:
    if engine is None:
        raise RuntimeError("model engine is not loaded")

    has_system = any(message["role"] == "system" for message in messages)
    payload = (
        messages
        if has_system
        else [
            {"role": "system", "content": [{"type": "text", "text": DEFAULT_SYSTEM_PROMPT}]},
            *messages,
        ]
    )

    conversation = engine.create_conversation(messages=payload)
    conversation.__enter__()
    try:
        response = conversation.send_message(payload[-1])
    finally:
        close = getattr(conversation, "__exit__", None)
        if close is not None:
            close(None, None, None)

    content = response.get("content", [])
    if isinstance(content, list):
        for item in content:
            if (
                isinstance(item, dict)
                and item.get("type") == "text"
                and isinstance(item.get("text"), str)
            ):
                return item["text"]
    return ""


@asynccontextmanager
async def lifespan(_: FastAPI):
    await asyncio.get_event_loop().run_in_executor(None, load_engine)
    yield


app = FastAPI(title="LiteRT LM Server", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins_from_env(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/v1/health")
async def health() -> Dict[str, Any]:
    return {
        "status": "healthy",
        "model": DEFAULT_MODEL_ID,
        "model_path": MODEL_PATH,
        "loaded": engine is not None,
    }


@app.get("/v1/models")
async def list_models() -> Dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": DEFAULT_MODEL_ID,
                "object": "model",
                "created": 0,
                "owned_by": "litert-lm",
            }
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(body: Dict[str, Any]):
    try:
        messages = _normalize_messages(body.get("messages"))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    model = (
        body.get("model")
        if isinstance(body.get("model"), str) and body.get("model")
        else DEFAULT_MODEL_ID
    )
    stream = bool(body.get("stream", False))

    try:
        text = await asyncio.to_thread(generate_completion, messages)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"generation failed: {exc}") from exc

    if not stream:
        return JSONResponse(content=_serialize_response(text=text, model=model))

    completion_id = f"chatcmpl-{uuid.uuid4().hex}"

    async def event_stream() -> AsyncGenerator[str, None]:
        yield f"data: {json.dumps(_serialize_stream_chunk('', model, completion_id, role='assistant'))}\n\n"
        for piece in _chunk_words(text):
            yield f"data: {json.dumps(_serialize_stream_chunk(piece, model, completion_id))}\n\n"
        yield f"data: {json.dumps(_serialize_stream_chunk('', model, completion_id, finish_reason='stop'))}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal OpenAI-compatible LiteRT LM server")
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
