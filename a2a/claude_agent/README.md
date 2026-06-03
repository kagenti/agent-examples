# Claude A2A Agent

An A2A agent that drives the `claude` CLI headlessly. Each A2A `context_id` maps to
a persistent Claude session with its own isolated working directory, so the
agent can serve concurrent sessions from one user and sessions from multiple users.
The model is reached through a LiteLLM (Anthropic-compatible) endpoint.

## Configuration

Set `ANTHROPIC_AUTH_TOKEN` and `ANTHROPIC_BASE_URL`; everything else has a default.

| Env | Default |
|---|---|
| `ANTHROPIC_AUTH_TOKEN` | (required) |
| `ANTHROPIC_BASE_URL` | (required) — your LiteLLM endpoint, e.g. `https://litellm.example.com`; empty falls back to `api.anthropic.com` |
| `ANTHROPIC_MODEL` | `sonnet` |
| `ANTHROPIC_DEFAULT_HAIKU_MODEL` | `haiku` |
| `WORKSPACE_ROOT` | `/workspace` |
| `MAX_SESSIONS` | `100` |
| `MAX_CONCURRENT` | `8` |
| `TURN_TIMEOUT_S` | `600` |
| `HOST` / `PORT` | `0.0.0.0` / `8000` |

## Security / trust model

The agent runs Claude with `--dangerously-skip-permissions`, so **a prompt can
execute arbitrary code inside the container** (edit files, run shell/tools) and read
the pod's environment. The container/pod is the only isolation boundary. Therefore:

- Treat prompts as fully trusted, and **isolate per tenant** (don't share one pod
  across untrusted users).
- **Do not co-locate unrelated secrets** in this pod — only the LiteLLM token it
  needs. Workspaces are ephemeral (pod lifetime) and isolated per A2A session.

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
ghcr.io/kagenti/agent-examples/claude_agent:latest
```

UI → **Agents → Import New Agent** → **Deploy from Existing Image**:

| Field | Value |
|---|---|
| Container Image | `ghcr.io/kagenti/agent-examples/claude_agent` |
| Image Tag | `latest` (or a released `vX.Y.Z`) |
| Namespace | `team1` |
| Agent Name | `claude-agent` |
| Protocol | `a2a` |
| Service port → target | `8080` → `8000` |

Under **Environment variables**, add (ui-v2 does not read the `sample-environments`
ConfigMap, so set these in the form):

- `ANTHROPIC_AUTH_TOKEN` = your LiteLLM token — **required**
- `ANTHROPIC_BASE_URL` = your LiteLLM endpoint (must be reachable from the cluster)
  — **required**
- `ANTHROPIC_MODEL` — optional, defaults to `sonnet`.

Create, wait for the pod to be Ready, then open the agent in the chat and send a
prompt.

> Workspaces are ephemeral (pod lifetime). Each A2A session gets its own Claude
> session + working directory; concurrent and multi-user sessions stay isolated.

### Session continuity (design gap)

Each turn spawns a fresh `claude` process and resumes by `--resume <uuid>`, which
reloads the conversation transcript stored on disk at `~/.claude/projects/<cwd>/<uuid>.jsonl`.
Because the workspace is ephemeral (an `emptyDir`), a **pod restart loses that
transcript**: a returning A2A `context_id` derives the same session UUID and so
silently starts a *fresh* conversation that appears continuous to the caller.
Durable continuity across restarts would require a persistent volume or an
external session store — out of scope for this example.

Pre-built manifests are also available at
`kagenti/examples/agents/claude_agent_*.yaml` in the `kagenti/kagenti` repo.
