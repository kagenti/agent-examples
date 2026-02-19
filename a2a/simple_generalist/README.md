# Simple Generalist Agent

This service exposes an A2A-compatible agent server that runs an AG2 agent and can optionally use MCP tools.

It was created to test the AppWorld MCP server. The agent prompt in `src/simple_generalist/agent/prompts.py` is geared toward using AppWorld as the MCP server.

## What It Actually Does

- Runs a Starlette A2A app via `a2a-sdk`.
- Creates a `GeneralistAgent` per request.
- Uses AG2 `ConversableAgent` + `UserProxyAgent` chat flow (`a_initiate_chat`) as the execution loop.
- If `MCP_SERVER_URL` is set, connects to that URL and builds an AG2 MCP toolkit for that request.
- Streams progress events during execution and returns a final text response.

## Current Behavior and Limits

- MCP connection is made per request (not at startup).
- `MAX_ITERATIONS` is used (`max_turns` in AG2 chat).
- Single MCP server is supported per request.

## Project Layout

```
src/simple_generalist/
├── main.py                      # Entrypoint and uvicorn startup
├── config/settings.py           # Environment-based settings
├── a2a_server/server.py         # A2A app, executor, MCP hookup
└── agent/
    ├── generalist_agent.py      # AG2 agent setup + task run
    └── prompts.py               # System prompt template
```


## Configuration

Key variables from `.env.template`:

- `LOG_LEVEL`
- `A2A_HOST`
- `A2A_PORT`
- `A2A_PUBLIC_URL` (optional public URL to advertise in AgentCard)
- `MCP_SERVER_URL`
- `LLM_MODEL`
- `LLM_API_KEY`
- `LLM_BASE_URL`
- `EXTRA_HEADERS`
- `LLM_TEMPERATURE`
- `MAX_ITERATIONS` (used)

## Run

Local:

1. Copy `.env.template` to `.env`.
2. Edit the environment variables in `.env`.
3. Start the server:
   ```bash
   uv run server
   ```

Kagenti:

1. Import `.env.template` in the UI as environment variables.
2. Customize the values as needed for your deployment.

## Notes

- OpenTelemetry/OpenLIT initialization is enabled when `OTEL_EXPORTER_OTLP_ENDPOINT` is present.
- If you use AppWorld MCP, expect a very large tool list (around 500 tools) to be injected into the prompt. Use a frontier model or a very large open-source model. This setup was tested successfully with `Qwen3 235B`.
