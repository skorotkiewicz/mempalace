#!/usr/bin/env python3
"""
agent.py — MemPalace-backed OpenAI-compatible proxy.

Purpose:
  - Accept OpenAI-style `/v1/chat/completions` requests from any client
  - Inject MemPalace wake-up context plus relevant memory search hits
  - Forward the request to a user-supplied OpenAI-compatible LLM endpoint
  - Persist the conversation back into MemPalace for future recall
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

from fastapi import Request

from mempalace.config import MempalaceConfig, sanitize_content, sanitize_name
from mempalace.layers import MemoryStack
from mempalace.palace import get_collection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("mempalace_agent")

HTTP_TIMEOUT_S = 300
MEMORY_HEADER = "# MemPalace Memory Context"
MEMORY_FOOTER = "# End MemPalace Memory Context"
SEARCH_HEADER = "## Relevant Verbatim Memories"
DEFAULT_STORE_WING = "conversations"
DEFAULT_STORE_ROOM = "general"


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _json_env_dict(name: str) -> Dict[str, str]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{name} must be valid JSON") from exc
    if not isinstance(parsed, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in parsed.items()
    ):
        raise ValueError(f"{name} must be a JSON object of string:string pairs")
    return parsed


def _normalize_base_url(url: str) -> str:
    return url.rstrip("/")


def _endpoint_url(base_url: str, path: str) -> str:
    return f"{_normalize_base_url(base_url)}{path}"


def _flatten_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type in {"text", "input_text", "output_text"}:
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(item.get("content"), str):
                parts.append(item["content"])
        return "\n".join(part for part in parts if part.strip())
    return ""


def _last_message_text(messages: List[Dict[str, Any]], role: str) -> str:
    for message in reversed(messages):
        if message.get("role") == role:
            return _flatten_content(message.get("content"))
    return ""


def _extract_assistant_text(payload: Dict[str, Any]) -> str:
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    message = choices[0].get("message", {})
    return _flatten_content(message.get("content"))


def _extract_stream_delta_text(chunk: Dict[str, Any]) -> str:
    choices = chunk.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    delta = choices[0].get("delta", {})
    return _flatten_content(delta.get("content"))


def _messages_are_new_chat(messages: List[Dict[str, Any]]) -> bool:
    has_user = False
    for message in messages:
        role = message.get("role")
        if role == "user":
            has_user = True
        elif role == "assistant":
            return False
    return has_user


def _memory_prompt(wake_up: str, existing_system: str, search_results: str) -> str:
    parts = [
        MEMORY_HEADER,
        "Use this memory context as background. Prefer verbatim memory evidence when relevant.",
        "",
        wake_up.strip(),
    ]
    if search_results.strip():
        parts.extend(["", SEARCH_HEADER, search_results.strip()])
    parts.extend(["", MEMORY_FOOTER])
    if existing_system.strip():
        parts.extend(["", "# Original System Instructions", existing_system.strip()])
    return "\n".join(parts)


def _inject_memory_context(
    messages: List[Dict[str, Any]],
    wake_up: str,
    search_results: str,
) -> List[Dict[str, Any]]:
    existing_system_parts: List[str] = []
    non_system_messages: List[Dict[str, Any]] = []

    for message in messages:
        if message.get("role") == "system":
            text = _flatten_content(message.get("content"))
            if text.strip():
                existing_system_parts.append(text.strip())
            continue
        non_system_messages.append(message)

    merged_system = "\n\n".join(existing_system_parts)
    system_message = {
        "role": "system",
        "content": _memory_prompt(wake_up, merged_system, search_results),
    }
    return [system_message, *non_system_messages]


def _validate_messages(messages: Any) -> List[Dict[str, Any]]:
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    valid_roles = {"system", "user", "assistant", "tool", "function"}
    validated: List[Dict[str, Any]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ValueError(f"message {index} must be an object")
        role = message.get("role")
        if role not in valid_roles:
            raise ValueError(f"message {index} has invalid role: {role}")
        validated.append(message)
    return validated


@dataclass
class BackendConfig:
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "local"
    model: str = ""
    extra_headers: Dict[str, str] = field(default_factory=dict)
    force_model: bool = True

    @classmethod
    def from_env(cls) -> "BackendConfig":
        return cls(
            base_url=_normalize_base_url(
                os.environ.get("AGENT_LLM_BASE_URL") or os.environ.get("LLM_BASE_URL") or cls.base_url
            ),
            api_key=os.environ.get("AGENT_LLM_API_KEY")
            or os.environ.get("LLM_API_KEY")
            or cls.api_key,
            model=os.environ.get("AGENT_LLM_MODEL") or os.environ.get("LLM_MODEL") or "",
            extra_headers=_json_env_dict("AGENT_LLM_EXTRA_HEADERS")
            if os.environ.get("AGENT_LLM_EXTRA_HEADERS")
            else {},
            force_model=_env_bool("AGENT_LLM_FORCE_MODEL", True),
        )

    def resolve_model(self, request_model: Any) -> str:
        if self.force_model and self.model:
            return self.model
        if isinstance(request_model, str) and request_model.strip():
            return request_model.strip()
        if self.model:
            return self.model
        return "local"

    def headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers.update(self.extra_headers)
        return headers


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8001

    @classmethod
    def from_env(cls) -> "ServerConfig":
        base_url = os.environ.get("AGENT_BASE_URL", "").strip()
        if base_url:
            parsed = urllib.parse.urlparse(base_url)
            return cls(host=parsed.hostname or cls.host, port=parsed.port or cls.port)
        return cls(
            host=os.environ.get("AGENT_HOST", cls.host),
            port=int(os.environ.get("AGENT_PORT", cls.port)),
        )


class ConversationStore:
    """Persist exchanges directly as conversation drawers."""

    def __init__(self, palace_path: str, wing: str = DEFAULT_STORE_WING, room: str = DEFAULT_STORE_ROOM):
        self._palace_path = palace_path
        self._wing = sanitize_name(wing, "wing")
        self._room = sanitize_name(room, "room")

    def _drawer_id(self, user_text: str, assistant_text: str, model: str) -> str:
        digest = hashlib.sha256(
            f"{self._wing}|{self._room}|{model}|{user_text}|{assistant_text}".encode("utf-8")
        ).hexdigest()
        return f"drawer_{self._wing}_{self._room}_{digest[:24]}"

    def save(self, user_text: str, assistant_text: str, model: str) -> None:
        user_text = sanitize_content(user_text, max_length=500_000)
        assistant_text = sanitize_content(assistant_text, max_length=500_000)
        drawer_id = self._drawer_id(user_text, assistant_text, model)
        content = f"User: {user_text}\n\nAssistant ({model}): {assistant_text}"
        collection = get_collection(self._palace_path, create=True)

        try:
            existing = collection.get(ids=[drawer_id])
            if existing and existing.get("ids"):
                return
        except Exception:
            pass

        collection.add(
            ids=[drawer_id],
            documents=[content],
            metadatas=[
                {
                    "wing": self._wing,
                    "room": self._room,
                    "source_file": "agent_conversation",
                    "chunk_index": 0,
                    "added_by": "mempalace_agent",
                    "filed_at": datetime.now().isoformat(),
                    "type": "conversation",
                    "model": model,
                }
            ],
        )

    def save_async_fire_and_forget(self, user_text: str, assistant_text: str, model: str) -> None:
        if not user_text.strip() or not assistant_text.strip():
            return
        loop = asyncio.get_running_loop()
        loop.run_in_executor(None, self._safe_save, user_text, assistant_text, model)

    def _safe_save(self, user_text: str, assistant_text: str, model: str) -> None:
        try:
            self.save(user_text, assistant_text, model)
        except Exception as exc:
            logger.warning("Conversation store failed: %s", exc)


class BackendProxy:
    """Thin OpenAI-compatible HTTP client over stdlib urllib."""

    def __init__(self, config: BackendConfig) -> None:
        self._config = config

    def _request(self, method: str, path: str, payload: Optional[Dict[str, Any]] = None) -> Any:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            _endpoint_url(self._config.base_url, path),
            data=data,
            headers=self._config.headers(),
            method=method,
        )
        return urllib.request.urlopen(request, timeout=HTTP_TIMEOUT_S)

    def list_models(self) -> Dict[str, Any]:
        try:
            with self._request("GET", "/models") as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            logger.warning("Backend /models failed: %s", exc)
            fallback_model = self._config.model or "local"
            return {
                "object": "list",
                "data": [
                    {
                        "id": fallback_model,
                        "object": "model",
                        "created": 0,
                        "owned_by": "mempalace-agent",
                    }
                ],
            }

    def chat_completion(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        try:
            with self._request("POST", "/chat/completions", payload=payload) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"backend HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"backend unavailable: {exc.reason}") from exc

    def stream_chat_completion(self, payload: Dict[str, Any]) -> Any:
        try:
            return self._request("POST", "/chat/completions", payload=payload)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"backend HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"backend unavailable: {exc.reason}") from exc


class MemPalaceProxyAgent:
    def __init__(
        self,
        data_path: str,
        backend: BackendConfig,
        auto_store: bool = True,
        wake_up_wing: Optional[str] = None,
        search_results: int = 3,
        search_enabled: bool = True,
        store_wing: str = DEFAULT_STORE_WING,
        store_room: str = DEFAULT_STORE_ROOM,
    ) -> None:
        self._data_path = Path(data_path).expanduser().resolve()
        self._backend = backend
        self._auto_store = auto_store
        self._wake_up_wing = wake_up_wing
        self._search_results = max(1, search_results)
        self._search_enabled = search_enabled
        self._backend_proxy = BackendProxy(backend)

        self._palace_path = str(self._data_path / ".mempalace" / "palace")
        os.makedirs(self._palace_path, exist_ok=True)
        os.environ["MEMPALACE_PALACE_PATH"] = self._palace_path

        self._stack = MemoryStack(palace_path=self._palace_path)
        self._store = ConversationStore(self._palace_path, wing=store_wing, room=store_room)
        self._wake_up_cache = self._stack.wake_up(wing=self._wake_up_wing)

        logger.info(
            "MemPalace proxy ready | data=%s | backend=%s | model=%s",
            self._data_path,
            self._backend.base_url,
            self._backend.model or "<request model>",
        )

    def _refresh_wake_up_if_needed(self, messages: List[Dict[str, Any]]) -> None:
        if _messages_are_new_chat(messages):
            self._wake_up_cache = self._stack.wake_up(wing=self._wake_up_wing)

    def _search_context(self, messages: List[Dict[str, Any]]) -> str:
        if not self._search_enabled:
            return ""
        query = _last_message_text(messages, "user").strip()
        if not query:
            return ""
        try:
            results = self._stack.search(query, n_results=self._search_results)
        except Exception as exc:
            logger.warning("Memory search failed: %s", exc)
            return ""
        if not isinstance(results, str):
            return ""
        stripped = results.strip()
        if not stripped or stripped.startswith("No results") or stripped.startswith("Search error"):
            return ""
        return stripped

    def build_backend_payload(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        original_messages = _validate_messages(request_data.get("messages"))
        self._refresh_wake_up_if_needed(original_messages)
        search_results = self._search_context(original_messages)
        enhanced_messages = _inject_memory_context(
            original_messages,
            wake_up=self._wake_up_cache,
            search_results=search_results,
        )

        payload = {
            key: value
            for key, value in request_data.items()
            if key not in {"messages", "model", "stream"}
        }
        payload["messages"] = enhanced_messages
        payload["model"] = self._backend.resolve_model(request_data.get("model"))
        payload["stream"] = bool(request_data.get("stream", False))
        return payload

    async def chat_completions(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        payload = self.build_backend_payload(request_data)
        return await asyncio.to_thread(self._backend_proxy.chat_completion, payload)

    def stream_chat_completions(
        self,
        request_data: Dict[str, Any],
    ) -> Iterable[str]:
        payload = self.build_backend_payload(request_data)
        user_text = _last_message_text(request_data["messages"], "user")
        model_name = payload["model"]

        def iterator() -> Iterator[str]:
            collected: List[str] = []
            response = self._backend_proxy.stream_chat_completion(payload)
            try:
                for raw_line in response:
                    line = raw_line.decode("utf-8", errors="replace")
                    if line.startswith("data: "):
                        data = line[6:].strip()
                        if data and data != "[DONE]":
                            try:
                                chunk = json.loads(data)
                            except json.JSONDecodeError:
                                chunk = {}
                            text = _extract_stream_delta_text(chunk)
                            if text:
                                collected.append(text)
                    yield line
            finally:
                response.close()
                if self._auto_store and collected:
                    self._store.save_async_fire_and_forget(
                        user_text=user_text,
                        assistant_text="".join(collected),
                        model=model_name,
                    )

        return iterator()

    async def list_models(self) -> Dict[str, Any]:
        return await asyncio.to_thread(self._backend_proxy.list_models)

    def search_memory(self, query: str, wing: Optional[str], room: Optional[str], n_results: int) -> str:
        return self._stack.search(query, wing=wing, room=room, n_results=n_results)

    def memory_status(self) -> Dict[str, Any]:
        return self._stack.status()

    async def maybe_store_non_stream(
        self,
        request_data: Dict[str, Any],
        response_data: Dict[str, Any],
    ) -> None:
        if not self._auto_store:
            return
        user_text = _last_message_text(request_data.get("messages", []), "user")
        assistant_text = _extract_assistant_text(response_data)
        model_name = self._backend.resolve_model(request_data.get("model"))
        self._store.save_async_fire_and_forget(
            user_text=user_text,
            assistant_text=assistant_text,
            model=model_name,
        )


def create_app(agent: MemPalaceProxyAgent):
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import JSONResponse, StreamingResponse
    except ImportError as exc:
        raise RuntimeError(
            "fastapi is required to run agent.py. Install fastapi and uvicorn."
        ) from exc

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield

    app = FastAPI(
        title="MemPalace Agent",
        description="OpenAI-compatible proxy with MemPalace memory injection",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.get("/v1/health")
    async def health() -> Dict[str, Any]:
        return {
            "status": "healthy",
            "backend": agent._backend.base_url,
            "model": agent._backend.model or "<request model>",
            "auto_store": agent._auto_store,
        }

    @app.get("/v1/models")
    async def list_models() -> Dict[str, Any]:
        return await agent.list_models()

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        try:
            request_data = await request.json()
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="invalid JSON body") from exc

        try:
            if request_data.get("stream", False):
                stream = agent.stream_chat_completions(request_data)
                return StreamingResponse(stream, media_type="text/event-stream")

            response_data = await agent.chat_completions(request_data)
            await agent.maybe_store_non_stream(request_data, response_data)
            return JSONResponse(content=response_data)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @app.post("/v1/memory/search")
    async def memory_search(request: Request) -> Dict[str, Any]:
        try:
            data = await request.json()
        except json.JSONDecodeError as exc:
            raise HTTPException(status_code=400, detail="invalid JSON body") from exc
        query = data.get("query", "")
        if not isinstance(query, str) or not query.strip():
            raise HTTPException(status_code=400, detail="query must be a non-empty string")
        results = agent.search_memory(
            query=query,
            wing=data.get("wing"),
            room=data.get("room"),
            n_results=int(data.get("n_results", 5)),
        )
        return {"results": results}

    @app.get("/v1/memory/status")
    async def memory_status() -> Dict[str, Any]:
        return agent.memory_status()

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MemPalace agent proxy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment:
  AGENT_LLM_BASE_URL      Backend base URL, e.g. http://localhost:8000/v1
  AGENT_LLM_API_KEY       Backend bearer token
  AGENT_LLM_MODEL         Model forced by the proxy
  AGENT_LLM_FORCE_MODEL   Force env model over request model (default: true)
  AGENT_LLM_EXTRA_HEADERS JSON object of extra backend headers
  AGENT_BASE_URL          Full bind URL, e.g. http://0.0.0.0:8001/v1
  AGENT_HOST              Bind host
  AGENT_PORT              Bind port
        """,
    )
    parser.add_argument("--data", required=True, help="User data directory")
    parser.add_argument("--host", default=None, help="Bind host")
    parser.add_argument("--port", type=int, default=None, help="Bind port")
    parser.add_argument("--wing", default=None, help="Optional wake-up wing filter")
    parser.add_argument("--search-results", type=int, default=3, help="Injected memory hit count")
    parser.add_argument("--no-search", action="store_true", help="Disable automatic memory search")
    parser.add_argument(
        "--no-auto-store",
        action="store_true",
        help="Do not write conversations back into MemPalace",
    )
    parser.add_argument("--store-wing", default=DEFAULT_STORE_WING, help="Wing for stored chats")
    parser.add_argument("--store-room", default=DEFAULT_STORE_ROOM, help="Room for stored chats")
    return parser.parse_args()


def main() -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("uvicorn is required to run agent.py. Install uvicorn.") from exc

    args = parse_args()
    backend = BackendConfig.from_env()
    server = ServerConfig.from_env()
    if args.host:
        server.host = args.host
    if args.port:
        server.port = args.port

    data_path = Path(args.data).expanduser().resolve()
    data_path.mkdir(parents=True, exist_ok=True)
    config = MempalaceConfig(config_dir=str(data_path / ".mempalace"))
    config.init()
    os.environ["MEMPALACE_PALACE_PATH"] = str(data_path / ".mempalace" / "palace")

    logger.info("=" * 60)
    logger.info("MemPalace Agent Starting")
    logger.info("  data    : %s", data_path)
    logger.info("  backend : %s", backend.base_url)
    logger.info("  model   : %s", backend.model or "<request model>")
    logger.info("  endpoint: http://%s:%d/v1", server.host, server.port)
    logger.info("=" * 60)

    agent = MemPalaceProxyAgent(
        data_path=str(data_path),
        backend=backend,
        auto_store=not args.no_auto_store,
        wake_up_wing=args.wing,
        search_results=args.search_results,
        search_enabled=not args.no_search,
        store_wing=args.store_wing,
        store_room=args.store_room,
    )
    app = create_app(agent)
    uvicorn.run(app, host=server.host, port=server.port, log_level="info")


if __name__ == "__main__":
    main()
