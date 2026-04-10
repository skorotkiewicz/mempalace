#!/usr/bin/env python3
"""
agent.py — MemPalace OpenAI-Compatible Agent Proxy

A proxy server that exposes ALL MemPalace features through the OpenAI API:
- Chat completions with automatic memory enhancement
- Function calling for: search, knowledge graph, diary, drawer management, etc.

Usage:
  AGENT_BASE_URL="http://0.0.0.0:8889/v1" LLM_BASE_URL="http://192.168.0.100:8000/v1" \\
    python agent.py --data=/path/to/user/data

Environment:
  LLM_BASE_URL    Backend LLM endpoint (default: http://localhost:8000/v1)
  LLM_API_KEY     Backend LLM API key (default: local)
  LLM_MODEL       Backend LLM model name (default: local)
  AGENT_BASE_URL  Agent bind URL (default: http://0.0.0.0:8001/v1)
  AGENT_HOST      Agent bind host (default: 0.0.0.0)
  AGENT_PORT      Agent bind port (default: 8001)

Connect your client to the AGENT_BASE_URL.
All MemPalace features are available through the /v1/chat/completions endpoint via function calling.
"""

import argparse
import hashlib
import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

import uvicorn

# FastAPI imports
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

# OpenAI client for backend (works with any OpenAI-compatible API)
from openai import AsyncOpenAI

from mempalace.config import MempalaceConfig, sanitize_name, sanitize_content

# MemPalace imports
from mempalace.layers import MemoryStack
from mempalace.palace import get_collection
from mempalace.knowledge_graph import KnowledgeGraph
from mempalace.palace_graph import traverse, find_tunnels, graph_stats
from mempalace.searcher import search_memories

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("mempalace_agent")


# =============================================================================
# MemPalace Tool Definitions (OpenAI Function Calling Format)
# =============================================================================

MEMPALACE_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "mempalace_search",
            "description": "Search the memory palace for relevant information. Returns verbatim drawer content with similarity scores. Use this when you need to find past conversations, decisions, or information.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for"},
                    "wing": {"type": "string", "description": "Filter by wing/project (optional)"},
                    "room": {"type": "string", "description": "Filter by room/topic (optional)"},
                    "limit": {
                        "type": "integer",
                        "description": "Max results (default: 5)",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mempalace_list_wings",
            "description": "List all wings in the palace with drawer counts. Wings represent people or projects.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mempalace_list_rooms",
            "description": "List rooms within a wing. Rooms represent topics within a project/person.",
            "parameters": {
                "type": "object",
                "properties": {
                    "wing": {
                        "type": "string",
                        "description": "Wing to list rooms for (optional - lists all if omitted)",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mempalace_get_taxonomy",
            "description": "Get full taxonomy: wing -> room -> drawer count tree. Shows the complete palace structure.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mempalace_add_drawer",
            "description": "Add content to the palace. Stores verbatim content in a wing/room.",
            "parameters": {
                "type": "object",
                "properties": {
                    "wing": {"type": "string", "description": "Wing name (project/person)"},
                    "room": {"type": "string", "description": "Room name (topic/aspect)"},
                    "content": {"type": "string", "description": "Verbatim content to store"},
                    "source_file": {
                        "type": "string",
                        "description": "Source identifier (optional)",
                    },
                },
                "required": ["wing", "room", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mempalace_delete_drawer",
            "description": "Delete a drawer by ID. Irreversible.",
            "parameters": {
                "type": "object",
                "properties": {
                    "drawer_id": {"type": "string", "description": "ID of drawer to delete"},
                },
                "required": ["drawer_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mempalace_check_duplicate",
            "description": "Check if content already exists in the palace before filing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "Content to check"},
                    "threshold": {
                        "type": "number",
                        "description": "Similarity threshold 0-1 (default: 0.9)",
                        "default": 0.9,
                    },
                },
                "required": ["content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mempalace_kg_query",
            "description": "Query the knowledge graph for an entity's relationships. Returns facts with temporal validity.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity": {
                        "type": "string",
                        "description": "Entity to query (e.g., 'Max', 'MyProject')",
                    },
                    "as_of": {"type": "string", "description": "Date filter YYYY-MM-DD (optional)"},
                    "direction": {
                        "type": "string",
                        "description": "outgoing, incoming, or both (default: both)",
                        "enum": ["outgoing", "incoming", "both"],
                        "default": "both",
                    },
                },
                "required": ["entity"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mempalace_kg_add",
            "description": "Add a fact to the knowledge graph. Subject -> predicate -> object with optional time window.",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string", "description": "The entity"},
                    "predicate": {
                        "type": "string",
                        "description": "Relationship type (e.g., 'works_on', 'loves')",
                    },
                    "object": {"type": "string", "description": "Connected entity"},
                    "valid_from": {
                        "type": "string",
                        "description": "When this became true YYYY-MM-DD (optional)",
                    },
                },
                "required": ["subject", "predicate", "object"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mempalace_kg_invalidate",
            "description": "Mark a fact as no longer true (set end date).",
            "parameters": {
                "type": "object",
                "properties": {
                    "subject": {"type": "string", "description": "Entity"},
                    "predicate": {"type": "string", "description": "Relationship"},
                    "object": {"type": "string", "description": "Connected entity"},
                    "ended": {
                        "type": "string",
                        "description": "When it stopped YYYY-MM-DD (default: today)",
                    },
                },
                "required": ["subject", "predicate", "object"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mempalace_kg_timeline",
            "description": "Get chronological timeline of facts for an entity or all entities.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity": {
                        "type": "string",
                        "description": "Entity to get timeline for (optional - all if omitted)",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mempalace_kg_stats",
            "description": "Get knowledge graph statistics: entities, triples, relationship types.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mempalace_diary_write",
            "description": "Write to the agent's personal diary. Each agent gets their own diary wing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "agent_name": {"type": "string", "description": "Agent/diary name"},
                    "entry": {"type": "string", "description": "Diary entry content"},
                    "topic": {
                        "type": "string",
                        "description": "Topic tag (default: general)",
                        "default": "general",
                    },
                },
                "required": ["agent_name", "entry"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mempalace_diary_read",
            "description": "Read recent diary entries for an agent.",
            "parameters": {
                "type": "object",
                "properties": {
                    "agent_name": {"type": "string", "description": "Agent/diary name"},
                    "last_n": {
                        "type": "integer",
                        "description": "Number of entries (default: 10)",
                        "default": 10,
                    },
                },
                "required": ["agent_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mempalace_traverse",
            "description": "Walk the palace graph from a room. Find connected ideas across wings.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_room": {"type": "string", "description": "Room to start from"},
                    "max_hops": {
                        "type": "integer",
                        "description": "Connections to follow (default: 2)",
                        "default": 2,
                    },
                },
                "required": ["start_room"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mempalace_find_tunnels",
            "description": "Find rooms that bridge two wings (hallways connecting domains).",
            "parameters": {
                "type": "object",
                "properties": {
                    "wing_a": {"type": "string", "description": "First wing (optional)"},
                    "wing_b": {"type": "string", "description": "Second wing (optional)"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mempalace_graph_stats",
            "description": "Get palace graph statistics: rooms, tunnels, edges.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "mempalace_status",
            "description": "Get full palace status: drawers, wings, rooms, protocol.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


class MemPalaceAgent:
    """
    MemPalace Agent Proxy — adds memory to any OpenAI-compatible API.

    The agent:
      - Loads L0 (identity) + L1 (essential story) on startup
      - Injects memory context into system prompts
      - Exposes ALL MemPalace features via function calling
      - Forwards requests to backend LLM
      - Auto-stores conversations
    """

    def __init__(
        self,
        data_path: str,
        backend_url: str,
        backend_key: str,
        backend_model: str,
        auto_store: bool = True,
        wake_up_wing: Optional[str] = None,
    ):
        self.data_path = Path(data_path).expanduser().resolve()
        self.backend_url = backend_url.rstrip("/")
        self.backend_key = backend_key
        self.backend_model = backend_model
        self.auto_store = auto_store
        self.wake_up_wing = wake_up_wing

        # Initialize palace at the data path
        self.palace_path = str(self.data_path / ".mempalace" / "palace")
        os.makedirs(self.palace_path, exist_ok=True)

        # Override env var for this process
        os.environ["MEMPALACE_PALACE_PATH"] = self.palace_path

        # Initialize memory stack
        self.stack = MemoryStack(palace_path=self.palace_path)

        # Initialize knowledge graph
        kg_path = os.path.join(self.palace_path, "knowledge_graph.sqlite3")
        self.kg = KnowledgeGraph(db_path=kg_path)

        # Load wake-up context once
        self.wake_up_context = self.stack.wake_up(wing=wake_up_wing)
        logger.info(f"Loaded wake-up context (~{len(self.wake_up_context) // 4} tokens)")

        # OpenAI client for backend (works with any OpenAI-compatible API)
        self.client = AsyncOpenAI(
            base_url=self.backend_url,
            api_key=self.backend_key,
            timeout=300.0,
        )

        logger.info(f"MemPalace Agent initialized")
        logger.info(f"  Data path: {self.data_path}")
        logger.info(f"  Palace path: {self.palace_path}")
        logger.info(f"  Backend: {self.backend_url}")
        logger.info(f"  Model: {self.backend_model}")

    def _build_system_prompt(self, existing_system: Optional[str] = None) -> str:
        """Build enhanced system prompt with MemPalace context."""
        parts = []

        # Add MemPalace wake-up context first
        parts.append("# MemPalace Memory Context")
        parts.append(
            "The following is your memory palace context. Use it to answer questions accurately."
        )
        parts.append("")
        parts.append(self.wake_up_context)
        parts.append("")
        parts.append("# End MemPalace Context")
        parts.append("")
        parts.append("# MemPalace Tools Available")
        parts.append("You have access to MemPalace memory tools via function calling:")
        parts.append("- mempalace_search: Search for information")
        parts.append("- mempalace_kg_query: Query knowledge graph")
        parts.append("- mempalace_kg_add: Add facts to knowledge graph")
        parts.append("- mempalace_diary_write/read: Personal agent diary")
        parts.append("- mempalace_list_wings/rooms/taxonomy: Browse palace structure")
        parts.append("- mempalace_add/delete_drawer: Manage drawers")
        parts.append("- mempalace_traverse/find_tunnels: Navigate palace graph")
        parts.append("- mempalace_status: Get palace overview")
        parts.append("")

        # Add existing system prompt if provided
        if existing_system:
            parts.append("# Your Original Instructions")
            parts.append(existing_system)

        return "\n".join(parts)

    def _enhance_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Enhance message list with MemPalace context in system prompt."""
        enhanced = []
        existing_system = None

        for msg in messages:
            if msg.get("role") == "system":
                existing_system = msg.get("content", "")
            else:
                enhanced.append(msg)

        # Build enhanced system prompt with memory
        system_prompt = self._build_system_prompt(existing_system)

        # Insert system prompt at the beginning
        enhanced.insert(0, {"role": "system", "content": system_prompt})

        return enhanced

    async def chat_completions(
        self, request_data: Dict[str, Any]
    ) -> Union[Dict[str, Any], AsyncGenerator[str, None]]:
        """Handle chat completions with memory enhancement and function calling."""
        messages = request_data.get("messages", [])
        if not messages:
            raise HTTPException(status_code=400, detail="No messages provided")

        is_streaming = request_data.get("stream", False)

        # Check if this is a function call response from user (tool output)
        last_message = messages[-1] if messages else {}
        if last_message.get("role") == "tool":
            # User is sending tool output, just forward to backend
            enhanced_messages = self._enhance_messages(messages)
            return await self._forward_to_backend(enhanced_messages, request_data, is_streaming)

        # Check if this is a search query we should augment
        if last_message.get("role") == "user":
            user_query = last_message.get("content", "")
            search_results = self.stack.search(user_query, n_results=3)
            if search_results and "No results" not in search_results:
                messages = self._inject_search_context(messages, search_results)

        # Enhance with MemPalace wake-up context
        enhanced_messages = self._enhance_messages(messages)

        # Check if tools/functions were requested
        requested_tools = request_data.get("tools") or request_data.get("functions")

        # If tools requested or this is a tool call chain, include MemPalace tools
        if requested_tools or self._has_tool_calls(messages):
            # Add MemPalace tools to the request
            all_tools = list(MEMPALACE_TOOLS)
            if requested_tools:
                all_tools.extend(requested_tools)

            # Forward with tools
            return await self._forward_with_tools(
                enhanced_messages, request_data, is_streaming, all_tools
            )

        # Simple forward without tools
        return await self._forward_to_backend(enhanced_messages, request_data, is_streaming)

    def _has_tool_calls(self, messages: List[Dict[str, str]]) -> bool:
        """Check if any message has tool calls."""
        for msg in messages:
            if msg.get("tool_calls") or msg.get("function_call"):
                return True
        return False

    async def _forward_with_tools(
        self,
        messages: List[Dict[str, str]],
        request_data: Dict[str, Any],
        is_streaming: bool,
        tools: List[Dict[str, Any]],
    ) -> Union[Dict[str, Any], AsyncGenerator[str, None]]:
        """Forward request with MemPalace tools available."""
        kwargs = {
            "model": self.backend_model,
            "messages": messages,
            "tools": tools,
            "stream": is_streaming,
        }

        # Forward optional parameters
        for param in [
            "temperature",
            "max_tokens",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "stop",
        ]:
            if param in request_data:
                kwargs[param] = request_data[param]

        try:
            response = await self.client.chat.completions.create(**kwargs)

            if is_streaming:
                return self._stream_with_tools(response, messages, kwargs)
            else:
                # Check if response has tool calls
                if response.choices and response.choices[0].message.tool_calls:
                    return await self._handle_tool_calls(response, messages, kwargs)

                # No tool calls, regular response
                result = self._openai_response_to_dict(response)
                response_content = self._extract_response_content(response)
                self._store_conversation(messages, response_content, self.backend_model)
                return result

        except Exception as e:
            logger.error(f"Backend error: {e}")
            raise HTTPException(status_code=502, detail=f"Backend error: {e}")

    async def _handle_tool_calls(
        self,
        response,
        messages: List[Dict[str, str]],
        kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Handle tool calls from the model."""
        message = response.choices[0].message
        tool_calls = message.tool_calls

        # Build response with tool calls
        result = self._openai_response_to_dict(response)

        # Execute MemPalace tool calls
        tool_outputs = []
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)

            # Execute the tool
            tool_result = await self._execute_tool(function_name, function_args)

            tool_outputs.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": json.dumps(tool_result),
                }
            )

        # Add tool outputs to result (for client to continue conversation)
        result["tool_outputs"] = tool_outputs

        return result

    async def _execute_tool(
        self, function_name: str, function_args: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a MemPalace tool."""
        try:
            if function_name == "mempalace_search":
                return search_memories(
                    query=function_args.get("query", ""),
                    palace_path=self.palace_path,
                    wing=function_args.get("wing"),
                    room=function_args.get("room"),
                    n_results=function_args.get("limit", 5),
                )

            elif function_name == "mempalace_list_wings":
                return self._tool_list_wings()

            elif function_name == "mempalace_list_rooms":
                return self._tool_list_rooms(function_args.get("wing"))

            elif function_name == "mempalace_get_taxonomy":
                return self._tool_get_taxonomy()

            elif function_name == "mempalace_add_drawer":
                return self._tool_add_drawer(
                    wing=function_args.get("wing", ""),
                    room=function_args.get("room", ""),
                    content=function_args.get("content", ""),
                    source_file=function_args.get("source_file"),
                )

            elif function_name == "mempalace_delete_drawer":
                return self._tool_delete_drawer(function_args.get("drawer_id", ""))

            elif function_name == "mempalace_check_duplicate":
                return self._tool_check_duplicate(
                    function_args.get("content", ""),
                    function_args.get("threshold", 0.9),
                )

            elif function_name == "mempalace_kg_query":
                results = self.kg.query_entity(
                    entity=function_args.get("entity", ""),
                    as_of=function_args.get("as_of"),
                    direction=function_args.get("direction", "both"),
                )
                return {"entity": function_args.get("entity"), "facts": results}

            elif function_name == "mempalace_kg_add":
                triple_id = self.kg.add_triple(
                    subject=sanitize_name(function_args.get("subject", ""), "subject"),
                    predicate=sanitize_name(function_args.get("predicate", ""), "predicate"),
                    object=sanitize_name(function_args.get("object", ""), "object"),
                    valid_from=function_args.get("valid_from"),
                )
                return {"success": True, "triple_id": triple_id}

            elif function_name == "mempalace_kg_invalidate":
                self.kg.invalidate(
                    subject=sanitize_name(function_args.get("subject", ""), "subject"),
                    predicate=sanitize_name(function_args.get("predicate", ""), "predicate"),
                    object=sanitize_name(function_args.get("object", ""), "object"),
                    ended=function_args.get("ended"),
                )
                return {"success": True}

            elif function_name == "mempalace_kg_timeline":
                results = self.kg.timeline(function_args.get("entity"))
                return {"timeline": results}

            elif function_name == "mempalace_kg_stats":
                return self.kg.stats()

            elif function_name == "mempalace_diary_write":
                return self._tool_diary_write(
                    agent_name=function_args.get("agent_name", ""),
                    entry=function_args.get("entry", ""),
                    topic=function_args.get("topic", "general"),
                )

            elif function_name == "mempalace_diary_read":
                return self._tool_diary_read(
                    agent_name=function_args.get("agent_name", ""),
                    last_n=function_args.get("last_n", 10),
                )

            elif function_name == "mempalace_traverse":
                collection = get_collection(self.palace_path, "mempalace_drawers")
                return traverse(
                    start_room=function_args.get("start_room", ""),
                    col=collection,
                    max_hops=function_args.get("max_hops", 2),
                )

            elif function_name == "mempalace_find_tunnels":
                collection = get_collection(self.palace_path, "mempalace_drawers")
                return find_tunnels(
                    wing_a=function_args.get("wing_a"),
                    wing_b=function_args.get("wing_b"),
                    col=collection,
                )

            elif function_name == "mempalace_graph_stats":
                collection = get_collection(self.palace_path, "mempalace_drawers")
                return graph_stats(col=collection)

            elif function_name == "mempalace_status":
                return self._tool_status()

            else:
                return {"error": f"Unknown tool: {function_name}"}

        except Exception as e:
            logger.error(f"Tool execution error ({function_name}): {e}")
            return {"error": str(e)}

    # =============================================================================
    # Tool Implementations
    # =============================================================================

    def _tool_list_wings(self) -> Dict[str, Any]:
        """List all wings."""
        try:
            collection = get_collection(self.palace_path, "mempalace_drawers")
            all_meta = collection.get(include=["metadatas"], limit=10000)["metadatas"]
            wings = {}
            for m in all_meta:
                w = m.get("wing", "unknown")
                wings[w] = wings.get(w, 0) + 1
            return {"wings": wings}
        except Exception as e:
            return {"error": str(e)}

    def _tool_list_rooms(self, wing: Optional[str] = None) -> Dict[str, Any]:
        """List rooms."""
        try:
            collection = get_collection(self.palace_path, "mempalace_drawers")
            kwargs = {"include": ["metadatas"], "limit": 10000}
            if wing:
                kwargs["where"] = {"wing": wing}
            all_meta = collection.get(**kwargs)["metadatas"]
            rooms = {}
            for m in all_meta:
                r = m.get("room", "unknown")
                rooms[r] = rooms.get(r, 0) + 1
            return {"wing": wing or "all", "rooms": rooms}
        except Exception as e:
            return {"error": str(e)}

    def _tool_get_taxonomy(self) -> Dict[str, Any]:
        """Get full taxonomy."""
        try:
            collection = get_collection(self.palace_path, "mempalace_drawers")
            all_meta = collection.get(include=["metadatas"], limit=10000)["metadatas"]
            taxonomy = {}
            for m in all_meta:
                w = m.get("wing", "unknown")
                r = m.get("room", "unknown")
                if w not in taxonomy:
                    taxonomy[w] = {}
                taxonomy[w][r] = taxonomy[w].get(r, 0) + 1
            return {"taxonomy": taxonomy}
        except Exception as e:
            return {"error": str(e)}

    def _tool_add_drawer(
        self, wing: str, room: str, content: str, source_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """Add a drawer."""
        try:
            wing_clean = sanitize_name(wing, "wing")
            room_clean = sanitize_name(room, "room")
            content_clean = sanitize_content(content)

            collection = get_collection(self.palace_path, "mempalace_drawers")

            content_hash = hashlib.sha256(
                (wing_clean + room_clean + content_clean[:100]).encode()
            ).hexdigest()[:24]
            drawer_id = f"drawer_{wing_clean}_{room_clean}_{content_hash}"

            # Check if exists
            try:
                existing = collection.get(ids=[drawer_id])
                if existing and existing["ids"]:
                    return {"success": True, "reason": "already_exists", "drawer_id": drawer_id}
            except Exception:
                pass

            collection.upsert(
                ids=[drawer_id],
                documents=[content_clean],
                metadatas=[
                    {
                        "wing": wing_clean,
                        "room": room_clean,
                        "source_file": source_file or "",
                        "chunk_index": 0,
                        "added_by": "mempalace_agent",
                        "filed_at": datetime.now().isoformat(),
                    }
                ],
            )
            return {"success": True, "drawer_id": drawer_id, "wing": wing_clean, "room": room_clean}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _tool_delete_drawer(self, drawer_id: str) -> Dict[str, Any]:
        """Delete a drawer."""
        try:
            collection = get_collection(self.palace_path, "mempalace_drawers")
            existing = collection.get(ids=[drawer_id])
            if not existing["ids"]:
                return {"success": False, "error": f"Drawer not found: {drawer_id}"}
            collection.delete(ids=[drawer_id])
            return {"success": True, "drawer_id": drawer_id}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _tool_check_duplicate(self, content: str, threshold: float = 0.9) -> Dict[str, Any]:
        """Check for duplicate content."""
        try:
            collection = get_collection(self.palace_path, "mempalace_drawers")
            results = collection.query(
                query_texts=[content],
                n_results=5,
                include=["metadatas", "documents", "distances"],
            )
            duplicates = []
            if results["ids"] and results["ids"][0]:
                for i, drawer_id in enumerate(results["ids"][0]):
                    dist = results["distances"][0][i]
                    similarity = round(1 - dist, 3)
                    if similarity >= threshold:
                        meta = results["metadatas"][0][i]
                        doc = results["documents"][0][i]
                        duplicates.append(
                            {
                                "id": drawer_id,
                                "wing": meta.get("wing", "?"),
                                "room": meta.get("room", "?"),
                                "similarity": similarity,
                                "content": doc[:200] + "..." if len(doc) > 200 else doc,
                            }
                        )
            return {"is_duplicate": len(duplicates) > 0, "matches": duplicates}
        except Exception as e:
            return {"error": str(e)}

    def _tool_diary_write(
        self, agent_name: str, entry: str, topic: str = "general"
    ) -> Dict[str, Any]:
        """Write to agent diary."""
        try:
            agent_clean = sanitize_name(agent_name, "agent_name")
            entry_clean = sanitize_content(entry)

            wing = f"wing_{agent_clean.lower().replace(' ', '_')}"
            room = "diary"

            collection = get_collection(self.palace_path, "mempalace_drawers")

            now = datetime.now()
            entry_id = f"diary_{wing}_{now.strftime('%Y%m%d_%H%M%S')}_{hashlib.sha256(entry_clean[:50].encode()).hexdigest()[:12]}"

            collection.add(
                ids=[entry_id],
                documents=[entry_clean],
                metadatas=[
                    {
                        "wing": wing,
                        "room": room,
                        "hall": "hall_diary",
                        "topic": topic,
                        "type": "diary_entry",
                        "agent": agent_clean,
                        "filed_at": now.isoformat(),
                        "date": now.strftime("%Y-%m-%d"),
                    }
                ],
            )
            return {
                "success": True,
                "entry_id": entry_id,
                "agent": agent_clean,
                "topic": topic,
                "timestamp": now.isoformat(),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def _tool_diary_read(self, agent_name: str, last_n: int = 10) -> Dict[str, Any]:
        """Read agent diary."""
        try:
            agent_clean = agent_name.lower().replace(" ", "_")
            wing = f"wing_{agent_clean}"

            collection = get_collection(self.palace_path, "mempalace_drawers")

            results = collection.get(
                where={"$and": [{"wing": wing}, {"room": "diary"}]},
                include=["documents", "metadatas"],
                limit=10000,
            )

            if not results["ids"]:
                return {"agent": agent_name, "entries": [], "message": "No diary entries yet."}

            entries = []
            for doc, meta in zip(results["documents"], results["metadatas"]):
                entries.append(
                    {
                        "date": meta.get("date", ""),
                        "timestamp": meta.get("filed_at", ""),
                        "topic": meta.get("topic", ""),
                        "content": doc,
                    }
                )

            entries.sort(key=lambda x: x["timestamp"], reverse=True)
            entries = entries[:last_n]

            return {
                "agent": agent_name,
                "entries": entries,
                "total": len(results["ids"]),
                "showing": len(entries),
            }
        except Exception as e:
            return {"error": str(e)}

    def _tool_status(self) -> Dict[str, Any]:
        """Get palace status."""
        try:
            collection = get_collection(self.palace_path, "mempalace_drawers")
            count = collection.count()

            wings = {}
            rooms = {}
            try:
                all_meta = collection.get(include=["metadatas"], limit=10000)["metadatas"]
                for m in all_meta:
                    w = m.get("wing", "unknown")
                    r = m.get("room", "unknown")
                    wings[w] = wings.get(w, 0) + 1
                    rooms[r] = rooms.get(r, 0) + 1
            except Exception:
                pass

            return {
                "total_drawers": count,
                "wings": wings,
                "rooms": rooms,
                "palace_path": self.palace_path,
            }
        except Exception as e:
            return {"error": str(e)}

    async def _forward_to_backend(
        self,
        messages: List[Dict[str, str]],
        request_data: Dict[str, Any],
        is_streaming: bool,
    ) -> Union[Dict[str, Any], AsyncGenerator[str, None]]:
        """Forward request to backend LLM."""
        kwargs = {
            "model": self.backend_model,
            "messages": messages,
            "stream": is_streaming,
        }

        # Forward optional parameters
        for param in [
            "temperature",
            "max_tokens",
            "top_p",
            "frequency_penalty",
            "presence_penalty",
            "stop",
        ]:
            if param in request_data:
                kwargs[param] = request_data[param]

        try:
            response = await self.client.chat.completions.create(**kwargs)

            if is_streaming:
                return self._stream_response(response, messages)
            else:
                result = self._openai_response_to_dict(response)
                response_content = self._extract_response_content(response)
                self._store_conversation(messages, response_content, self.backend_model)
                return result

        except Exception as e:
            logger.error(f"Backend error: {e}")
            raise HTTPException(status_code=502, detail=f"Backend error: {e}")

    async def _stream_response(
        self, stream, original_messages: List[Dict[str, str]]
    ) -> AsyncGenerator[str, None]:
        """Stream OpenAI response as SSE events and store conversation."""
        content_buffer = []

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
                content_buffer.append(chunk.choices[0].delta.content)

            data = json.dumps(self._openai_stream_chunk_to_dict(chunk))
            yield f"data: {data}\n\n"

        yield "data: [DONE]\n\n"

        if content_buffer:
            full_response = "".join(content_buffer)
            self._store_conversation(original_messages, full_response, self.backend_model)

    async def _stream_with_tools(
        self, stream, original_messages: List[Dict[str, str]], kwargs: Dict[str, Any]
    ) -> AsyncGenerator[str, None]:
        """Stream response with tool support."""
        # For streaming with tools, we just pass through and let client handle it
        async for chunk in stream:
            data = json.dumps(self._openai_stream_chunk_to_dict(chunk))
            yield f"data: {data}\n\n"
        yield "data: [DONE]\n\n"

    def _openai_response_to_dict(self, response) -> Dict[str, Any]:
        """Convert OpenAI response object to dict."""
        result = {
            "id": response.id,
            "object": response.object,
            "created": response.created,
            "model": response.model,
            "choices": [],
        }

        for choice in response.choices:
            choice_dict = {
                "index": choice.index,
                "finish_reason": choice.finish_reason,
            }

            # Handle message or delta
            if hasattr(choice, "message") and choice.message:
                msg = choice.message
                choice_dict["message"] = {
                    "role": msg.role,
                    "content": msg.content,
                }
                # Handle tool calls
                if msg.tool_calls:
                    choice_dict["message"]["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in msg.tool_calls
                    ]

            result["choices"].append(choice_dict)

        if response.usage:
            result["usage"] = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return result

    def _openai_stream_chunk_to_dict(self, chunk) -> Dict[str, Any]:
        """Convert OpenAI stream chunk to dict."""
        result = {
            "id": chunk.id,
            "object": chunk.object,
            "created": chunk.created,
            "model": chunk.model,
            "choices": [],
        }

        for choice in chunk.choices:
            choice_dict = {
                "index": choice.index,
                "finish_reason": choice.finish_reason,
            }

            if choice.delta:
                choice_dict["delta"] = {}
                if choice.delta.role:
                    choice_dict["delta"]["role"] = choice.delta.role
                if choice.delta.content:
                    choice_dict["delta"]["content"] = choice.delta.content
                # Handle tool call deltas
                if choice.delta.tool_calls:
                    choice_dict["delta"]["tool_calls"] = [
                        {
                            "index": tc.index,
                            "id": tc.id if tc.id else None,
                            "type": tc.type if tc.type else None,
                            "function": {
                                "name": tc.function.name if tc.function.name else None,
                                "arguments": tc.function.arguments
                                if tc.function.arguments
                                else None,
                            },
                        }
                        for tc in choice.delta.tool_calls
                    ]

            result["choices"].append(choice_dict)

        return result

    def _inject_search_context(
        self, messages: List[Dict[str, str]], search_results: str
    ) -> List[Dict[str, str]]:
        """Inject L3 search results into the conversation context."""
        has_system = False
        for msg in messages:
            if msg.get("role") == "system":
                has_system = True
                original = msg.get("content", "")
                msg["content"] = f"{original}\n\n## Relevant Memories\n{search_results}"
                break

        if not has_system:
            messages.insert(
                0, {"role": "system", "content": f"## Relevant Memories\n{search_results}"}
            )

        return messages

    async def models(self) -> Dict[str, Any]:
        """Return available models (proxied from backend)."""
        try:
            response = await self.client.models.list()
            return {
                "object": "list",
                "data": [
                    {
                        "id": model.id,
                        "object": "model",
                        "created": getattr(model, "created", 0),
                        "owned_by": getattr(model, "owned_by", "unknown"),
                    }
                    for model in response.data
                ],
            }
        except Exception:
            return {
                "object": "list",
                "data": [
                    {
                        "id": self.backend_model,
                        "object": "model",
                        "created": 0,
                        "owned_by": "mempalace-agent",
                    }
                ],
            }

    async def close(self):
        """Cleanup resources."""
        await self.client.close()

    def _store_conversation(
        self,
        messages: List[Dict[str, str]],
        response_content: str,
        model: str,
        wing: str = "conversations",
        room: str = "general",
    ):
        """Store conversation exchange to MemPalace."""
        if not self.auto_store:
            return

        try:
            user_msg = ""
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    user_msg = msg.get("content", "")
                    break

            content = f"User: {user_msg}\n\nAssistant ({model}): {response_content}"

            collection = get_collection(self.palace_path, "mempalace_drawers")

            content_hash = hashlib.sha256((wing + room + content[:100]).encode()).hexdigest()[:24]
            drawer_id = f"drawer_{wing}_{room}_{content_hash}"

            try:
                existing = collection.get(ids=[drawer_id])
                if existing and existing["ids"]:
                    return
            except Exception:
                pass

            wing_clean = sanitize_name(wing, "wing")
            room_clean = sanitize_name(room, "room")

            collection.add(
                ids=[drawer_id],
                documents=[content],
                metadatas=[
                    {
                        "wing": wing_clean,
                        "room": room_clean,
                        "source_file": "agent_conversation",
                        "chunk_index": 0,
                        "added_by": "mempalace_agent",
                        "filed_at": datetime.now().isoformat(),
                        "type": "conversation",
                    }
                ],
            )
            logger.debug(f"Stored conversation to {wing_clean}/{room_clean}: {drawer_id}")
        except Exception as e:
            logger.warning(f"Failed to store conversation: {e}")

    def _extract_response_content(self, response) -> str:
        """Extract content from OpenAI response."""
        if hasattr(response, "choices") and response.choices:
            return response.choices[0].message.content or ""
        return ""


# FastAPI app
def create_app(agent: MemPalaceAgent) -> FastAPI:
    """Create FastAPI application with the agent."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Lifespan context manager for startup/shutdown."""
        yield
        await agent.close()

    app = FastAPI(
        title="MemPalace Agent Proxy",
        description="OpenAI-compatible proxy with MemPalace memory enhancement",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.get("/v1/models")
    async def list_models():
        """List available models."""
        # Include MemPalace tools in the model description
        models = await agent.models()
        # Add a note about available tools
        if "data" in models and models["data"]:
            models["data"][0]["tools_available"] = [t["function"]["name"] for t in MEMPALACE_TOOLS]
        return models

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        """Chat completions endpoint - THE ONLY ENDPOINT YOU NEED."""
        try:
            request_data = await request.json()
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON")

        is_streaming = request_data.get("stream", False)
        result = await agent.chat_completions(request_data)

        if is_streaming:
            return StreamingResponse(result, media_type="text/event-stream")
        else:
            return JSONResponse(content=result)

    return app


def main():
    parser = argparse.ArgumentParser(
        description="MemPalace Agent Proxy — OpenAI-compatible API with memory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment Variables (Backend LLM):
  LLM_BASE_URL    Backend LLM endpoint (default: http://localhost:8000/v1)
  LLM_API_KEY     Backend LLM API key (default: local)
  LLM_MODEL       Backend LLM model name (default: local)

Environment Variables (Agent Binding):
  AGENT_BASE_URL  Full URL like http://0.0.0.0:8889/v1 (parsed for host:port)
  AGENT_HOST      Agent bind host (default: 0.0.0.0)
  AGENT_PORT      Agent bind port (default: 8001)

Priority: AGENT_BASE_URL > AGENT_HOST/AGENT_PORT > --host/--port args

All MemPalace features are available via function calling through /v1/chat/completions.
Available tools: mempalace_search, mempalace_kg_query, mempalace_kg_add, 
mempalace_diary_write, mempalace_diary_read, mempalace_list_wings, 
mempalace_list_rooms, mempalace_get_taxonomy, mempalace_add_drawer, 
mempalace_delete_drawer, mempalace_check_duplicate, mempalace_kg_invalidate,
mempalace_kg_timeline, mempalace_kg_stats, mempalace_traverse, 
mempalace_find_tunnels, mempalace_graph_stats, mempalace_status

Example:
  AGENT_BASE_URL="http://0.0.0.0:8889/v1" LLM_BASE_URL="http://192.168.0.100:8000/v1" \\
    uv run python agent.py --data=/home/user/data
        """,
    )
    parser.add_argument(
        "--data", required=True, help="Path to user data directory (where palace will be stored)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to run agent on (default: 8001, or from AGENT_BASE_URL)",
    )
    parser.add_argument(
        "--host", default=None, help="Host to bind to (default: 0.0.0.0, or from AGENT_BASE_URL)"
    )
    parser.add_argument(
        "--no-auto-store", action="store_true", help="Disable automatic conversation storage"
    )
    parser.add_argument("--wing", default=None, help="Specific wing to load for wake-up context")

    args = parser.parse_args()

    # Get agent binding config from environment
    agent_base_url = os.environ.get("AGENT_BASE_URL", "")
    if agent_base_url:
        try:
            from urllib.parse import urlparse

            parsed = urlparse(agent_base_url)
            env_host = parsed.hostname or "0.0.0.0"
            env_port = parsed.port or 8001
        except Exception:
            env_host = "0.0.0.0"
            env_port = 8001
    else:
        env_host = os.environ.get("AGENT_HOST", "0.0.0.0")
        env_port = int(os.environ.get("AGENT_PORT", "8001"))

    host = args.host if args.host else env_host
    port = args.port if args.port else env_port

    # Get backend config from environment
    backend_url = os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1")
    backend_key = os.environ.get("LLM_API_KEY", "local")
    backend_model = os.environ.get("LLM_MODEL", "local")

    # Ensure data directory exists
    data_path = Path(args.data).expanduser().resolve()
    data_path.mkdir(parents=True, exist_ok=True)

    # Initialize palace if needed
    palace_path = data_path / ".mempalace" / "palace"
    palace_path.mkdir(parents=True, exist_ok=True)

    # Initialize MemPalace config
    config = MempalaceConfig(config_dir=str(data_path / ".mempalace"))
    config.init()

    logger.info("=" * 60)
    logger.info("MemPalace Agent Proxy Starting")
    logger.info("=" * 60)
    logger.info(f"Data directory: {data_path}")
    logger.info(f"Palace path: {palace_path}")
    logger.info(f"Backend LLM: {backend_url}")
    logger.info(f"Backend model: {backend_model}")
    logger.info(f"Agent endpoint: http://{host}:{port}/v1")
    logger.info(f"Available tools: {len(MEMPALACE_TOOLS)} MemPalace functions")
    logger.info("=" * 60)

    # Create agent
    agent = MemPalaceAgent(
        data_path=str(data_path),
        backend_url=backend_url,
        backend_key=backend_key,
        backend_model=backend_model,
        auto_store=not args.no_auto_store,
        wake_up_wing=args.wing,
    )

    # Create FastAPI app
    app = create_app(agent)

    # Run server
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
