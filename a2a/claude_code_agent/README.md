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

## Deploy on Kagenti (Kind)

Build the image, load it into the cluster's in-cluster registry, then create the
agent from the UI.

### 1. Build and push to the in-cluster registry

`kubectl port-forward` to the registry is unreliable for large images on
Kind/podman, so push from inside the Kind node over the cluster network:

```bash
# Build (run from this directory)
podman build -t localhost/claude-code-agent:dev .

# Load into the Kind node, then tag + push to the in-cluster registry
kind load docker-image localhost/claude-code-agent:dev --name kagenti
REG=registry.cr-system.svc.cluster.local:5000/claude-code-agent:v0.0.1
podman exec kagenti-control-plane ctr -n k8s.io images tag \
  localhost/claude-code-agent:dev $REG
podman exec kagenti-control-plane ctr -n k8s.io images push --plain-http $REG

# Verify
podman exec kagenti-control-plane \
  curl -s http://registry.cr-system.svc.cluster.local:5000/v2/claude-code-agent/tags/list
```

(Replace `kagenti` / `kagenti-control-plane` if your cluster/node names differ.)

### 2. Create the agent in the UI

UI → **Agents → Import New Agent** → **Deploy from Existing Image**:

| Field | Value |
|---|---|
| Container Image | `registry.cr-system.svc.cluster.local:5000/claude-code-agent` |
| Image Tag | `v0.0.1` |
| Image Pull Secret | _(blank)_ |
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
