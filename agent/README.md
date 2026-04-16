# MemPalace Agent

MemPalace Agent is a small OpenAI-compatible proxy that sits in front of any LLM endpoint and gives it memory.

It does three things:

- loads MemPalace wake-up context
- searches verbatim memory on each request
- writes the conversation back into the palace

Your model endpoint stays yours. MemPalace stays local.

Main module: `agent.py`

## Install

```bash
cd agent
uv sync
```

## Run

```bash
AGENT_LLM_BASE_URL="http://localhost:8000/v1" \
AGENT_LLM_MODEL="your-model" \
uv run agent.py --data ~/my-agent
```

By default the proxy serves on `http://127.0.0.1:8001/v1`.

You can also bind it explicitly:

```bash
AGENT_BASE_URL="http://0.0.0.0:8889/v1" \
AGENT_LLM_BASE_URL="http://192.168.0.124:8888/v1" \
AGENT_LLM_MODEL="your-model" \
uv run agent.py --data ~/my-agent
```

Then point any OpenAI-compatible client at:

```text
http://127.0.0.1:8889/v1
```

## Example

```bash
curl http://127.0.0.1:8001/v1/chat/completions \
  -X POST \
  -H 'Content-Type: application/json' \
  --data-raw '{"messages":[{"role":"user","content":"hi"}],"stream":false}'
```

## Environment

- `AGENT_LLM_BASE_URL`: backend LLM base URL
- `AGENT_LLM_API_KEY`: optional bearer token for the backend
- `AGENT_LLM_MODEL`: model forced by the proxy
- `AGENT_LLM_FORCE_MODEL`: if `true`, ignore request model and use `AGENT_LLM_MODEL`
- `AGENT_LLM_EXTRA_HEADERS`: JSON object of extra backend headers
- `AGENT_BASE_URL`: proxy bind URL
- `AGENT_HOST`: proxy bind host
- `AGENT_PORT`: proxy bind port

Legacy `LLM_BASE_URL`, `LLM_API_KEY`, and `LLM_MODEL` are also accepted.

## Endpoints

- `POST /v1/chat/completions`
- `GET /v1/models`
- `GET /v1/health`
- `POST /v1/memory/search`
- `GET /v1/memory/status`

## Notes

- The memory store lives under `--data/.mempalace/`
- Use `127.0.0.1` or a real host when connecting clients; do not send client traffic to `0.0.0.0`
- Streaming is supported
