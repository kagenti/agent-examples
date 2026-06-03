# Claude Code A2A Agent

An A2A agent that drives the `claude` CLI headlessly. Each A2A `context_id` maps to
a persistent Claude Code session with its own isolated working directory, so the
agent can serve concurrent sessions from one user and sessions from multiple users.
The model is reached through a LiteLLM (Anthropic-compatible) endpoint.

## Configuration

The only required value is `ANTHROPIC_AUTH_TOKEN`. Everything else has a default.

| Env | Default |
|---|---|
| `ANTHROPIC_AUTH_TOKEN` | (required) |
| `ANTHROPIC_BASE_URL` | `https://ete-litellm.ai-models.vpc-int.res.ibm.com` |
| `ANTHROPIC_MODEL` | `sonnet` |
| `ANTHROPIC_DEFAULT_HAIKU_MODEL` | `haiku` |
| `WORKSPACE_ROOT` | `/workspace` |
| `MAX_SESSIONS` | `100` |
| `MAX_CONCURRENT` | `8` |
| `TURN_TIMEOUT_S` | `600` |
| `HOST` / `PORT` | `0.0.0.0` / `8000` |

## Run locally

```bash
export ANTHROPIC_AUTH_TOKEN=...   # your LiteLLM key
uv run server
```

## Test

```bash
uv run pytest -v
```
