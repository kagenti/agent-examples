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

## Deploy on Kagenti

The image is published to GHCR by the repo's `Build-Publish` workflow (on `v*`
tags), so you just point Kagenti at it — no local build needed:

```
ghcr.io/kagenti/agent-examples/claude_code_agent:latest
```

UI → **Agents → Import New Agent** → **Deploy from Existing Image**:

| Field | Value |
|---|---|
| Container Image | `ghcr.io/kagenti/agent-examples/claude_code_agent` |
| Image Tag | `latest` (or a released `vX.Y.Z`) |
| Namespace | `team1` |
| Agent Name | `claude-code-agent` |
| Protocol | `a2a` |
| Service port → target | `8080` → `8000` |

Under **Environment variables**, add `ANTHROPIC_AUTH_TOKEN` = your LiteLLM token
(required; ui-v2 does not read the `sample-environments` ConfigMap). Override
`ANTHROPIC_BASE_URL` / `ANTHROPIC_MODEL` only if the defaults don't fit — note the
default base URL is IBM-internal and must be reachable from the cluster.

Create, wait for the pod to be Ready, then open the agent in the chat and send a
prompt.

> Workspaces are ephemeral (pod lifetime). Each A2A session gets its own Claude
> Code session + working directory; concurrent and multi-user sessions stay isolated.

Pre-built manifests are also available at
`kagenti/examples/agents/claude_code_agent_*.yaml` in the `kagenti/kagenti` repo.
