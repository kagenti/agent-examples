# Wiki Memory Tool

Multi-agent wiki memory service for the Kagenti Research Wiki use case.
Provides persistent, git-backed knowledge storage with per-topic access control, GitHub OAuth login, and SPIFFE workload identity.

## Status

This is a **standalone MCP service** — a self-contained wiki with REST + MCP APIs,
git-backed storage, and identity-aware access control.

**What this is:**
- A working memory/knowledge store for multi-agent systems
- SPIFFE-based agent identity (simulated via headers today)
- GitHub OAuth for human users
- Deployable to any Kubernetes cluster (Kind, OpenShift, etc.)

**What this is not (yet):**
- Not wired into the Kagenti operator (no AgentService CR)
- No real SPIRE SVID verification (header-based, not mTLS)
- No Keycloak federation
- These are tracked as follow-up in [kagenti#1461](https://github.com/kagenti/kagenti/issues/1461)

## Quick Start (Local)

### Prerequisites

- Python 3.14+
- [uv](https://docs.astral.sh/uv/) package manager
- git

### Install and Run

```bash
cd mcp/wiki_memory_tool
uv sync

# Terminal 1 — start the service (local-only, no GitHub push)
uv run python run_local.py --clean

# Terminal 2 — run the test agents
uv run python test_agents.py
```

### Run with GitHub Remote Storage

```bash
export WIKI_REMOTE_URL="https://x-access-token:<PAT>@github.com/<org>/<repo>.git"

# Start service — clones from GitHub, pushes after each write
uv run python run_local.py --remote --clean

# Run test agents
uv run python test_agents.py
```

## GitHub OAuth Setup

### Step 1: Create a GitHub OAuth App

1. Go to **GitHub Settings > Developer settings > OAuth Apps > New OAuth App**
   (or for an org: **Organization Settings > Developer settings > OAuth Apps**)
2. Fill in:
   - **Application name**: `Wiki Memory Service`
   - **Homepage URL**: `https://<your-wiki-service-url>` (e.g. `https://wiki-memory-service-team1.apps.ykt1.hcp.res.ibm.com`)
   - **Authorization callback URL**: `https://<your-wiki-service-url>/auth/github/callback`
3. Click **Register application**
4. Note the **Client ID** (e.g. `Ov23liMX4KIYA2hFFhuL`)
5. Generate a **Client secret** and save it securely

### Step 2: Enable Device Flow

1. In the OAuth App settings, check **Enable Device Flow**
2. This allows CLI login without a browser redirect

### Step 3: Create GitHub Organization and Teams

1. Create an organization (e.g. `kaslomorg`) at https://github.com/organizations/new
2. Create teams under the org at `https://github.com/orgs/<org>/teams`:
   - `ml-team` — ML researchers (read access to ml topic)
   - `ml-writers` — ML content creators (write access to ai/ml topics)
   - `platform-admins` — Platform administrators (admin access to all topics)
   - `security-team` — Security researchers (read/write access to security topic)
3. Add members to each team

### Step 4: Configure the Service

Set these environment variables (or K8s secrets):

```bash
export GITHUB_CLIENT_ID="Ov23liMX4KIYA2hFFhuL"
export GITHUB_CLIENT_SECRET="<your-client-secret>"
export JWT_SECRET_KEY="$(openssl rand -hex 32)"
```

For Kubernetes deployment, create the secret:

```bash
kubectl create secret generic wiki-github-oauth \
  --from-literal=GITHUB_CLIENT_ID=Ov23liMX4KIYA2hFFhuL \
  --from-literal=GITHUB_CLIENT_SECRET=<secret> \
  --from-literal=JWT_SECRET_KEY=$(openssl rand -hex 32) \
  -n <namespace>
```

### Step 5: Configure ACL with GitHub Identities

Edit `test_acl.yaml` (local) or the `wiki-memory-acl` ConfigMap (K8s) to map teams to topics:

```yaml
topics:
  ai:
    writers:
      - "github:team:kaslomorg/ml-writers"
      - "github:team:kaslomorg/platform-admins"
    readers:
      - "github:org:kaslomorg"    # all org members can read
      - "*"                        # or all authenticated users
    admins:
      - "github:user:aslom"
      - "github:team:kaslomorg/platform-admins"
```

Identity prefix reference:

| Prefix | Matches |
|--------|---------|
| `spiffe://...` | Workload identity (agents) |
| `github:user:<login>` | Individual GitHub user |
| `github:team:<org>/<team>` | GitHub team membership |
| `github:org:<org>` | Any member of GitHub org |
| `*` | Any authenticated identity |

### Step 6: Remove GitHub Org OAuth App Access Restrictions

By default, GitHub organizations restrict third-party OAuth app access. This blocks the wiki service from reading team memberships via `/user/teams`. You must remove (or approve) the restriction:

1. Go to **Organization Settings > Third-party access > OAuth application policy**
   (`https://github.com/organizations/<org>/settings/oauth_application_policy`)
2. Either:
   - **Remove restrictions** entirely (allows all approved apps), or
   - **Approve** the Wiki Memory Service app specifically
3. Verify by logging in again — `kwiki whoami` should show your groups

Without this, `GET /user/teams` returns an empty list even though the user belongs to teams, because GitHub blocks the OAuth app from seeing org data.

## CLI Usage

### Shell Alias (recommended)

Add this to your `~/.zshrc` (or `~/.bashrc`) so you can use `kwiki` from anywhere:

```bash
alias kwiki='uv run --directory ~/sandbox/kagenti-mvp/agent-examples/mcp/wiki_memory_tool python wiki_cli.py --base-url https://wiki-memory-service-team1.apps.ykt1.hcp.res.ibm.com'
```

Then reload: `source ~/.zshrc`

### Login via GitHub Device Flow

```bash
kwiki login
```

Output:
```
==================================================
  GitHub Device Authorization
==================================================

  1. Open this URL in your browser:

     https://github.com/login/device

  2. Enter this code:

     ABCD-1234

  Code expires in 15 minutes.
==================================================

Waiting for authorization.....

Logged in as aslom
Groups: kaslomorg/ml-writers, kaslomorg/platform-admins
```

### Check Identity

```bash
kwiki whoami
```

Output:
```
User:   aslom
Status: valid (expires in 6d 23h)
Groups: kaslomorg/ml-writers, kaslomorg/platform-admins
Server: https://wiki-memory-service-team1.apps.ykt1.hcp.res.ibm.com

Access:
  ai: read, write, admin
    read  <- github:team:kaslomorg/platform-admins
    write <- github:team:kaslomorg/platform-admins
    admin <- github:user:aslom
  security: read, write
    read  <- github:team:kaslomorg/platform-admins
    write <- github:team:kaslomorg/platform-admins
```

### Renew Token

Renew your JWT token without going through the full device flow again (works within the 7-day grace window):

```bash
kwiki renew
```

Output:
```
Token renewed for aslom (expires in 168h)
```

If the token is too old or the server doesn't support renewal, `kwiki renew` falls back to the device flow login automatically.

### Write a Wiki Page (as authenticated user)

```bash
kwiki discover write ai my-notes.md --content "# My Notes\n\nContent here..."
```

### Query Wiki Pages

```bash
kwiki query list-topics
kwiki query search ai "transformer architecture"
kwiki query read ai transformers.md
```

### Logout

```bash
kwiki logout
```

### Full CLI Reference

```bash
kwiki login                          # GitHub device flow login
kwiki logout                         # Remove cached token
kwiki whoami                         # Show current identity, groups, and permissions
kwiki renew                          # Renew token (7-day grace window)

kwiki discover write <topic> <path> --content "..."   # Write a page
kwiki discover write <topic> <path> --file ./doc.md   # Write from file
kwiki discover write <topic> <path> --draft --content "..."  # Submit as draft
kwiki discover novelty <topic> "Title" "Abstract"     # Check novelty
kwiki discover template                               # List available templates
kwiki discover template paper-summary                 # Get a specific template

kwiki query list-topics              # List all topics
kwiki query list-pages <topic>       # List pages in a topic
kwiki query search <topic> "query"   # TF-IDF search within a topic
kwiki query search-all "query"       # Search across ALL topics
kwiki query read <topic> <path>      # Read a page (with frontmatter)
kwiki query activity [topic]         # Recent changes (topic or global)
kwiki query backlinks <topic> <path> # Find pages linking to a page
kwiki query tags <topic>             # List all tags with page counts
kwiki query tag <topic> <tag>        # List pages with a specific tag
kwiki query graph <topic>            # Page link graph (nodes + edges)
kwiki query drafts <topic>           # List pending drafts

kwiki admin approve <topic> <path>   # Approve a draft
kwiki admin reject <topic> <path> --reason "..."  # Reject a draft
kwiki admin init-pages               # Initialize GitHub Pages layout
```

### Examples: Querying the AI Topic

The `ai` topic contains pages on transformer architecture, RAG patterns, fine-tuning, and evaluation:

```bash
# What topics are available?
kwiki query list-topics

# What pages exist in the ai topic?
kwiki query list-pages ai

# Search for specific concepts
kwiki query search ai "attention mechanism"
kwiki query search ai "LoRA fine-tuning"
kwiki query search ai "retrieval augmented generation"
kwiki query search ai "evaluation metrics BLEU"

# Read a specific page
kwiki query read ai transformers.md
kwiki query read ai rag-patterns.md
kwiki query read ai fine-tuning.md
```

### Examples: Writing a New Page from an arXiv Paper

Use an AI agent (Claude, etc.) to summarize a paper and write it to the wiki:

```bash
# Step 1: Check if similar content already exists
kwiki discover novelty ai "Mixture of Experts" "Sparse MoE architectures for scaling LLMs efficiently"

# Step 2: If novel, have your agent generate markdown and write it
kwiki discover write ai mixture-of-experts.md --content "$(cat <<'EOF'
# Mixture of Experts (MoE)

Source: [Switch Transformers](https://arxiv.org/abs/2101.03961) (Fedus et al., 2021)

## Summary

Mixture of Experts routes each token to a subset of specialist
sub-networks (experts), enabling model scaling without proportional
compute increase.

## Key Ideas

- **Sparse gating**: each token activates only top-k experts (typically k=1 or k=2)
- **Load balancing loss**: auxiliary loss prevents all tokens routing to one expert
- **Capacity factor**: limits tokens per expert to prevent memory overflow

## Architecture

```
Input → Router (softmax) → Top-K experts → Weighted sum → Output
```

## Results

- Switch Transformer: 7x speedup over T5-Base at same compute budget
- Mixtral 8x7B: competitive with GPT-3.5 using only 2 active experts per token

## References

- Fedus et al. "Switch Transformers" (2021) — https://arxiv.org/abs/2101.03961
- Jiang et al. "Mixtral of Experts" (2024) — https://arxiv.org/abs/2401.04088
EOF
)"
```

### Workflow: Agent-Assisted Paper Summarization

For a more automated workflow, use Claude Code or another agent to fetch, summarize, and write:

```bash
# 1. Agent fetches and summarizes the paper (example using Claude Code)
#    Ask: "Summarize https://arxiv.org/abs/2305.14314 as a wiki page in markdown"

# 2. Save agent output to a file
#    (agent writes summary to /tmp/paper-summary.md)

# 3. Check novelty before writing
kwiki discover novelty ai "Tree of Thoughts" "Deliberate problem solving with LLMs using tree search"

# 4. Write the page from the file
kwiki discover write ai tree-of-thoughts.md --file /tmp/paper-summary.md \
  --message "Add Tree of Thoughts paper summary (Yao et al. 2023)"

# 5. Verify it's searchable
kwiki query search ai "tree of thoughts deliberation"
kwiki query read ai tree-of-thoughts.md
```

### Page Content Guidelines

When writing wiki pages (manually or via agent), follow this structure:

```markdown
# Title

Source: [Paper Name](https://arxiv.org/abs/XXXX.XXXXX) (Authors, Year)

## Summary
2-3 sentence overview of the key contribution.

## Key Ideas
- Bullet points of main concepts
- Include formulas or pseudocode if relevant

## Architecture / Method
Describe the approach with diagrams or code blocks.

## Results
Key experimental findings and comparisons.

## References
- Links to paper, code, related work
```

## MCP Integration

### Local Mode (stdio)

Add to your Claude Code MCP config:

```json
{
  "mcpServers": {
    "wiki-memory": {
      "command": "uv",
      "args": ["run", "python", "mcp_server.py"],
      "cwd": "/path/to/mcp/wiki_memory_tool"
    }
  }
}
```

Tools available:
- `wiki_list_topics` — List all topics and page counts
- `wiki_query` — Search a topic (TF-IDF)
- `wiki_search_all` — Search across all topics
- `wiki_read` — Read a page (includes frontmatter)
- `wiki_write` — Write/update a page (supports `draft=True`)
- `wiki_check_novelty` — Check if content is novel
- `wiki_activity` — Recent changes (git log)
- `wiki_backlinks` — Find pages linking to a page
- `wiki_list_tags` — List all tags in a topic
- `wiki_pages_by_tag` — Find pages by tag
- `wiki_graph` — Page link graph (nodes + edges)
- `wiki_get_template` — Get page templates
- `wiki_list_drafts` — List pending drafts
- `wiki_approve_draft` — Approve a draft (admin)

### Remote Mode (with auth)

Set `WIKI_SERVICE_URL` to connect to the remote wiki service with your cached token:

```json
{
  "mcpServers": {
    "wiki-memory": {
      "command": "uv",
      "args": ["run", "python", "mcp_server.py"],
      "cwd": "/path/to/mcp/wiki_memory_tool",
      "env": {
        "WIKI_SERVICE_URL": "https://wiki-memory-service-team1.apps.ykt1.hcp.res.ibm.com"
      }
    }
  }
}
```

Login first with `wiki_cli.py login` — the MCP server reads the cached token from `~/.wiki-memory/token.json`.

## Claude Code Skills

The wiki memory tool provides 6 skills that can be invoked directly in Claude Code with `/kwiki:*` commands.

### Install Skills

```bash
# Install into current project (symlinks)
uv run python install_kwiki_skills.py .claude/skills

# Install into user-global skills
uv run python install_kwiki_skills.py --global

# Copy instead of symlink (for distribution)
uv run python install_kwiki_skills.py --copy /path/to/.claude/skills
```

### Available Skills

| Skill | Description |
|-------|-------------|
| `/kwiki:query-cli` | Search and read wiki pages using the `kwiki` CLI |
| `/kwiki:query-api` | Search and read wiki pages via REST API (curl) |
| `/kwiki:query-mcp` | Search and read wiki pages using MCP tools |
| `/kwiki:discover-cli` | Write pages using the `kwiki` CLI after novelty check |
| `/kwiki:discover-api` | Write pages via REST API (curl) after novelty check |
| `/kwiki:discover-mcp` | Write pages using MCP tools after novelty check |

### Example: Using /kwiki:query-cli

Type `/kwiki:query-cli` in Claude Code, then ask it to search, read, or explore:

```
> /kwiki:query-cli
> Search the ai topic for "attention" and show backlinks for transformers.md

$ kwiki query list-topics
  ai (3 pages)
  security (0 pages)
  ml (0 pages)

$ kwiki query search ai "attention mechanism"
  ai/transformers.md (score=0.1652)
    tags: [paper, attention, transformer]

$ kwiki query backlinks ai transformers.md
  rag-patterns.md
  fine-tuning.md

$ kwiki query tags ai
  attention (1 pages)
  fine-tuning (1 pages)
  paper (2 pages)
  rag (1 pages)
  transformer (1 pages)

$ kwiki query graph ai
Nodes (3):
  rag-patterns.md: RAG Patterns [paper, rag, retrieval]
  transformers.md: Attention Is All You Need [paper, attention, transformer]
  fine-tuning.md: Fine-Tuning Guide [guide, fine-tuning, lora]

Edges (5):
  rag-patterns.md -> transformers.md
  transformers.md -> rag-patterns.md
  transformers.md -> fine-tuning.md
  fine-tuning.md -> transformers.md
  fine-tuning.md -> rag-patterns.md
```

### Example: Using /kwiki:discover-cli

Type `/kwiki:discover-cli` in Claude Code to write new knowledge:

```
> /kwiki:discover-cli
> Write a summary of LoRA to the ai topic

$ kwiki discover template paper-summary
--- Paper Summary ---
  ---
  tags: [paper, summary]
  ---
  # {Title}
  Source: [{Paper Name}]({URL}) (Authors, Year)
  ...

$ kwiki discover novelty ai "LoRA Fine-Tuning" "Low-rank adaptation for efficient model tuning"
NOVEL: No sufficiently similar content found

$ kwiki discover write ai lora.md --content "---\ntags: [paper, fine-tuning]\n---\n# LoRA..."
Written: ai/lora.md by discovery-agent
Suggested links:
  ai/transformers.md (score=0.1538)

$ kwiki discover write ai experimental.md --draft --content "# Experimental\n..."
Draft: ai/experimental.md by discovery-agent

$ kwiki query drafts ai
  experimental.md
```

### Example: Using /kwiki:query-api

Type `/kwiki:query-api` for curl-based access:

```
> /kwiki:query-api
> List topics, search for "retrieval", and show the graph

$ curl -s http://localhost:8321/topics -H "X-Spiffe-Id: ..." -H "X-Original-Subject: ..."
→ {"topics": [{"topic_id": "ai", "page_count": 2}, {"topic_id": "security", "page_count": 0}]}

$ curl -s -X POST http://localhost:8321/search -d '{"query": "retrieval"}'
→ {"results": [{"path": "ai/rag-patterns.md", "score": 0.1171, "snippet": "Retrieval augmented generation.", "topic_id": "ai"}]}

$ curl -s http://localhost:8321/topics/ai/tags
→ {"topic": "ai", "tags": {"paper": 2, "rag": 1, "attention": 1}}

$ curl -s http://localhost:8321/topics/ai/graph
→ {"topic": "ai", "nodes": [{"id": "rag-patterns.md", "title": "RAG Patterns", "tags": ["paper","rag"]}, ...], "edges": [{"source": "rag-patterns.md", "target": "transformers.md"}, ...]}

$ curl -s http://localhost:8321/templates
→ {"templates": [{"id": "paper-summary", "name": "Paper Summary", ...}, {"id": "concept-overview", ...}, {"id": "how-to-guide", ...}, {"id": "comparison", ...}]}
```

### Example: Using /kwiki:discover-api

Type `/kwiki:discover-api` to write via REST API with full control:

```
> /kwiki:discover-api
> Check novelty for "LoRA" and write it as a draft

$ curl -s http://localhost:8321/templates/paper-summary
→ {"id": "paper-summary", "name": "Paper Summary", "content": "---\ntags: [paper, summary]\n---\n# {Title}\n..."}

$ curl -s -X POST http://localhost:8321/topics/ai/check-novelty -d '{"title": "LoRA", "abstract": "Low-rank adaptation"}'
→ {"novel": true, "reason": "No sufficiently similar content found"}

$ curl -s -X POST "http://localhost:8321/topics/ai/pages/lora.md?draft=true" -d '{"content": "---\ntags: [paper]\n---\n# LoRA\n..."}'
→ {"status": "draft", "path": "ai/lora.md", "author": "discovery-agent"}

$ curl -s http://localhost:8321/topics/ai/drafts
→ {"topic": "ai", "drafts": ["lora.md"]}
```

### Example: Using /kwiki:query-mcp and /kwiki:discover-mcp

These skills use MCP tools directly (requires MCP server configured):

```
> /kwiki:query-mcp
> What topics exist, search for "attention", and show backlinks

>>> wiki_list_topics()
Topics:
- ai (2 pages)
- security (0 pages)
- ml (0 pages)

>>> wiki_query(topic_id="ai", query="attention")
Search results for 'attention' in 'ai':
- ai/transformers.md (score=0.2555)
  tags: [paper, attention]

>>> wiki_search_all(query="retrieval")
Global search results for 'retrieval':
- [ai] ai/rag-patterns.md (score=0.1171)
  Retrieval augmented generation.

>>> wiki_backlinks(topic_id="ai", path="transformers.md")
Pages linking to ai/transformers.md:
- rag-patterns.md

>>> wiki_list_tags(topic_id="ai")
Tags in 'ai':
- attention (1 pages)
- paper (2 pages)
- rag (1 pages)

>>> wiki_graph(topic_id="ai")
Nodes: 2, Edges: 2
  rag-patterns.md: RAG Patterns ['paper', 'rag']
  transformers.md: Transformers ['paper', 'attention']
  rag-patterns.md -> transformers.md
  transformers.md -> rag-patterns.md
```

```
> /kwiki:discover-mcp
> Check novelty for LoRA and write it as a draft

>>> wiki_get_template()
Available templates:
- paper-summary: Paper Summary — Summarize an academic paper or technical report
- concept-overview: Concept Overview — Explain a technical concept or method
- how-to-guide: How-To Guide — Step-by-step practical guide
- comparison: Comparison — Compare approaches, tools, or methods

>>> wiki_check_novelty(topic_id="ai", title="LoRA", abstract="Low-rank adaptation")
{"novel": true, "reason": "No sufficiently similar content found"}

>>> wiki_write(topic_id="ai", path="lora.md", content="---\ntags: [paper]\n---\n# LoRA\n...", draft=True)
Draft: ai/lora.md

>>> wiki_list_drafts(topic_id="ai")
Drafts in 'ai':
- lora.md
```

### Skill Workflow: Login → Query → Discover

The recommended workflow when using skills:

1. **Login first** (one-time): `kwiki login`
2. **Query** existing knowledge: `/kwiki:query-cli` or `/kwiki:query-mcp`
3. **Discover** and write new knowledge: `/kwiki:discover-cli` or `/kwiki:discover-mcp`

Skills automatically follow best practices:
- Always check novelty before writing (avoids duplicates)
- Use structured markdown format with YAML frontmatter and tags
- Return suggested links to related content after writing
- Support draft mode for content requiring human review
- Respect ACL (returns 403 if you lack write access)

## REST API

### Authentication Options

**Option A: GitHub OAuth Token** (for users)
```bash
curl -H "Authorization: Bearer <wiki-jwt>" https://wiki-service/topics
```

**Option B: SPIFFE Headers** (for agents)
```bash
curl -H "X-Spiffe-Id: spiffe://kagenti.example.com/ns/topic-ai/sa/discovery-agent" \
     https://wiki-service/topics/ai/pages/doc.md
```

### Endpoints

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/healthz` | GET | none | Health check |
| `/topics` | GET | read | List all topics |
| `/topics/{id}/pages` | GET | read | List pages in topic |
| `/topics/{id}/pages/{path}` | GET | read | Read a page (includes frontmatter) |
| `/topics/{id}/pages/{path}` | POST | write | Write a page (returns suggested links) |
| `/topics/{id}/pages/{path}?draft=true` | POST | write | Submit as draft for review |
| `/topics/{id}/query` | POST | read | Search a topic |
| `/topics/{id}/check-novelty` | POST | read | Check content novelty |
| `/topics/{id}/activity` | GET | read | Recent changes for a topic |
| `/activity` | GET | any | Recent changes across all topics |
| `/search` | POST | read | Global search across all topics |
| `/topics/{id}/backlinks/{path}` | GET | read | Find pages linking to a page |
| `/topics/{id}/tags` | GET | read | List all tags with page counts |
| `/topics/{id}/tags/{tag}` | GET | read | List pages with a specific tag |
| `/topics/{id}/graph` | GET | read | Page link graph (nodes + edges) |
| `/topics/{id}/drafts` | GET | write | List pending drafts |
| `/topics/{id}/drafts/{path}/approve` | POST | admin | Approve a draft |
| `/topics/{id}/drafts/{path}/reject` | POST | admin | Reject a draft |
| `/templates` | GET | none | List page templates |
| `/templates/{id}` | GET | none | Get a page template |
| `/topics/{id}/pages/{path}` | DELETE | admin | Delete a page |
| `/auth/github/login` | GET | none | Start OAuth browser flow |
| `/auth/github/callback` | GET | none | OAuth callback |
| `/auth/github/device` | POST | none | Start device flow |
| `/auth/github/device/token` | POST | none | Poll device flow |
| `/auth/whoami` | GET | any | Show current identity |
| `/auth/permissions` | GET | any | Show per-topic permissions with explanations |
| `/auth/renew` | POST | any | Renew JWT token (within 7-day grace window) |
| `/admin/reload-acl` | POST | admin | Reload ACL config |
| `/admin/init-pages` | POST | admin | Initialize GitHub Pages layout and fix page front-matter/links |

## Versioning

The service version is defined in **`pyproject.toml`** (single source of truth):

```toml
[project]
version = "0.2.0"
```

### How version flows

1. **`pyproject.toml`** → `version = "X.Y.Z"` — edit this to bump
2. **`wiki_service.py`** → reads version from `pyproject.toml` at startup, exposes in `/healthz` and OpenAPI
3. **`deploy.py`** → reads version from `pyproject.toml`, tags docker image as `quay.io/aslomnet/wiki-memory-service:X.Y.Z`
4. **`k8s/deployment.yaml`** → references the image tag (update manually or use `deploy.py --build`)

### Bumping the version

```bash
# 1. Edit pyproject.toml
#    version = "0.3.0"

# 2. Update k8s/deployment.yaml image tag to match
#    image: quay.io/aslomnet/wiki-memory-service:0.3.0

# 3. Build, push, and deploy
uv run deploy.py --build
```

The `/healthz` endpoint returns the running version for verification.

## Kubernetes Deployment

### Build and Push

```bash
cd mcp/wiki_memory_tool

# Read version from pyproject.toml and tag accordingly
VERSION=$(grep '^version' pyproject.toml | cut -d'"' -f2)
docker build --platform linux/amd64 -t quay.io/aslomnet/wiki-memory-service:$VERSION .
docker push quay.io/aslomnet/wiki-memory-service:$VERSION

# Also push :latest for convenience
docker tag quay.io/aslomnet/wiki-memory-service:$VERSION quay.io/aslomnet/wiki-memory-service:latest
docker push quay.io/aslomnet/wiki-memory-service:latest
```

### Deploy

```bash
# Create namespace
kubectl create ns wiki-memory-service

# Apply manifests
kubectl apply -f k8s/serviceaccount.yaml
kubectl apply -f k8s/acl-configmap.yaml
kubectl apply -f k8s/deployment.yaml

# Create secrets
kubectl create secret generic wiki-github-remote \
  --from-literal=WIKI_REMOTE_URL="https://x-access-token:<PAT>@github.com/<org>/<repo>.git" \
  -n wiki-memory-service

kubectl create secret generic wiki-github-oauth \
  --from-literal=GITHUB_CLIENT_ID=<client-id> \
  --from-literal=GITHUB_CLIENT_SECRET=<client-secret> \
  --from-literal=JWT_SECRET_KEY=$(openssl rand -hex 32) \
  -n wiki-memory-service
```

### Verify

```bash
curl https://<wiki-service-url>/healthz
curl -X POST https://<wiki-service-url>/auth/github/device
```

## UX/DX Features

### Activity Feed

Track recent changes across the wiki:

```bash
# Global activity
kwiki query activity

# Topic-specific activity
kwiki query activity ai
```

Output:
```
  2024-01-15 14:30:00 +0000 discovery-agent: ingest: ai/transformers.md
  2024-01-15 14:29:00 +0000 discovery-agent: ingest: ai/rag-patterns.md
```

### Tags & Frontmatter

Pages can include YAML frontmatter with tags:

```markdown
---
tags: [paper, attention, transformer]
---
# Transformers

Content here...
```

Query by tags:

```bash
kwiki query tags ai              # List all tags: paper (4), attention (2), ...
kwiki query tag ai paper         # Pages with "paper" tag
```

### Backlinks

Find pages that reference a given page:

```bash
kwiki query backlinks ai transformers.md
```

Output:
```
  rag-patterns.md
  fine-tuning.md
```

### Global Search

Search across all accessible topics simultaneously:

```bash
kwiki query search-all "attention mechanism"
```

Output:
```
  [ai] ai/transformers.md (score=0.1572)
    The transformer model uses self-attention mechanisms...
  [ml] ml/bert.md (score=0.0834)
    BERT uses bidirectional attention...
```

### Page Templates

Get structured templates for new pages:

```bash
kwiki discover template                    # List all templates
kwiki discover template paper-summary      # Get paper summary template
kwiki discover template concept-overview   # Get concept overview template
kwiki discover template how-to-guide       # Get how-to guide template
kwiki discover template comparison         # Get comparison template
```

### Draft/Review Queue

Submit pages for human review before publishing:

```bash
# Write as draft
kwiki discover write ai new-concept.md --draft --content "# Draft\n..."

# List pending drafts
kwiki query drafts ai

# Approve (requires admin access)
kwiki admin approve ai new-concept.md

# Reject with reason
kwiki admin reject ai new-concept.md --reason "needs more references"
```

### Page Graph

Visualize relationships between pages:

```bash
kwiki query graph ai
```

Output:
```
Nodes (4):
  transformers.md: Transformers [paper, attention]
  rag-patterns.md: RAG Patterns [paper, rag]
  fine-tuning.md: Fine-Tuning [guide]
  evaluation.md: Evaluation Metrics [metrics]

Edges (5):
  transformers.md -> rag-patterns.md
  transformers.md -> fine-tuning.md
  rag-patterns.md -> transformers.md
  fine-tuning.md -> evaluation.md
  evaluation.md -> transformers.md
```

### Suggested Links on Write

When writing a page, the service automatically suggests related pages:

```bash
kwiki discover write ai moe.md --content "# Mixture of Experts\n..."
```

Output:
```
Written: ai/moe.md by discovery-agent
Suggested links:
  ai/transformers.md (score=0.0923)
  ai/fine-tuning.md (score=0.0412)
```

### GitHub Pages Initialization

The wiki content is stored in a git repo that doubles as a GitHub Pages site. The `init-pages` command sets up the Jekyll scaffolding and fixes existing pages for proper rendering.

```bash
kwiki admin init-pages
```

Output:
```
GitHub Pages initialized (6 files):
  _config.yml
  index.md
  _layouts/default.html
  _layouts/page.html
  _includes/nav.html
  assets/css/style.css
```

**Requires:** `platform-admins` team membership (admin access to `_system` topic).

#### What it does

1. **Writes Jekyll scaffold files** to the wiki repo root:
   - `_config.yml` — Jekyll configuration with `kramdown` markdown, baseurl, and defaults that auto-apply `layout: page` to all files in topic directories
   - `index.md` — Front page that lists all wiki pages with links and tags
   - `_layouts/default.html` — Base HTML layout with sidebar navigation
   - `_layouts/page.html` — Page layout showing title, tags, and content
   - `_includes/nav.html` — Navigation listing all pages
   - `assets/css/style.css` — Light/dark mode CSS using `prefers-color-scheme`

2. **Adds YAML front-matter** to every existing `.md` file that needs it:
   - `layout: page` — tells Jekyll which layout to use
   - `title: "..."` — extracted from the first `# Heading` in the file
   - Preserves existing `tags` and other metadata

3. **Converts internal links to Jekyll `{% link %}` syntax:**

   | Before | After |
   |--------|-------|
   | `[RAG](rag-patterns.md)` | `[RAG]({% link ai/rag-patterns.md %})` |
   | `[[transformers]]` | `[Transformers]({% link ai/transformers.md %})` |

   This ensures Jekyll validates links at build time — if a linked page is missing, the build fails instead of producing a broken link.

4. **Commits and pushes** all changes to the remote repository, triggering a GitHub Pages rebuild.

#### Light/Dark Mode

The CSS uses `@media (prefers-color-scheme: dark)` — no JavaScript required. The site automatically matches the user's browser/OS preference.

#### Re-running

Safe to run multiple times. It overwrites scaffold files with the latest version and re-processes all `.md` files to fix any new pages that were added since the last run.

## Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Per-topic directories** | Topics are the isolation boundary, agents are actors with scoped access |
| **Git storage** | Full audit trail, commits attributed per agent, GitHub as persistent backend |
| **TF-IDF search** | Zero external deps for MVP; add vector search later |
| **SPIFFE + GitHub OAuth** | Workload identity for agents, GitHub identity for humans |
| **GitHub teams -> ACL** | Org structure maps directly to wiki topic access |
| **HMAC-SHA256 JWT** | Self-issued tokens, no external auth server needed |
| **uv package manager** | Fast, reproducible builds in both local dev and container |
| **Immediate push strategy** | Each write is pushed to GitHub immediately for durability |

## Project Files

```
wiki_memory_tool/
├── wiki_service.py          # FastAPI service (OAuth, ACL, git, search)
├── wiki_cli.py              # CLI tool (login, discover, query)
├── mcp_server.py            # MCP server (local + remote modes)
├── run_local.py             # Local runner (--clean, --remote flags)
├── test_agents.py           # Integration test agents
├── install_kwiki_skills.py  # Skill installer (symlink/copy into .claude/skills)
├── test_acl.yaml            # Local ACL config
├── pyproject.toml           # uv-managed deps
├── Dockerfile               # Container build (uv + Python 3.14)
├── deploy.py                # Deployment automation script
├── skills/                  # Skill documentation (6 kwiki skills)
│   ├── wiki-discovery-api/
│   ├── wiki-discovery-cli/
│   ├── wiki-discovery-mcp/
│   ├── wiki-query-api/
│   ├── wiki-query-cli/
│   └── wiki-query-mcp/
└── k8s/
    ├── deployment.yaml
    ├── acl-configmap.yaml
    └── serviceaccount.yaml
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `WIKI_ROOT` | `/data/wiki` | Local git repo path for wiki pages |
| `ACL_FILE` | `/config/acl.yaml` | Path to per-topic ACL YAML |
| `SPIFFE_TRUST_DOMAIN` | `kagenti.example.com` | SPIFFE trust domain |
| `WIKI_REMOTE_URL` | *(empty)* | Git remote URL (enables clone + push) |
| `WIKI_PUSH_STRATEGY` | `immediate` | Push strategy: `immediate` |
| `GITHUB_CLIENT_ID` | *(empty)* | GitHub OAuth App client ID |
| `GITHUB_CLIENT_SECRET` | *(empty)* | GitHub OAuth App client secret |
| `JWT_SECRET_KEY` | *(empty)* | HMAC-SHA256 key for signing wiki JWTs |
| `JWT_EXPIRY_HOURS` | `168` | JWT token expiry in hours (default 7 days) |
| `WIKI_GITHUB_ORG` | `kaslomorg` | GitHub org for team resolution |
| `WIKI_SERVICE_URL` | *(empty)* | Remote wiki URL (for MCP remote mode) |
| `WIKI_INSECURE_TLS` | `0` | Set to `1` to disable TLS certificate verification (dev only) |

## Troubleshooting Authorization

### Groups showing as empty after login

If `kwiki whoami` shows `Groups: (none)` but the user belongs to GitHub teams:

1. **Check org OAuth app restrictions** — the most common cause. Go to:
   `https://github.com/organizations/<org>/settings/oauth_application_policy`
   
   If the org has third-party access restrictions enabled, the wiki service's OAuth app cannot read team memberships. Either remove restrictions or approve the app.

2. **Verify the OAuth app has `read:org` scope** — the device flow requests this scope. If the user denied it during authorization, teams won't resolve.

3. **Check pod logs** for team resolution output:
   ```bash
   oc logs deployment/wiki-memory-service -n <namespace> | grep -i team
   ```

### Debugging a deployed service

Use this workflow to diagnose authorization issues on the cluster:

```bash
# 1. Set KUBECONFIG if needed
export KUBECONFIG=~/.kube/config-kagenti-eventing

# 2. Check the pod is running the expected image
oc get pods -n <namespace> -o jsonpath='{.items[0].status.containerStatuses[0].imageID}'

# 3. Force a fresh rollout after rebuilding the image
oc rollout restart deployment/wiki-memory-service -n <namespace>

# 4. Watch rollout progress
oc rollout status deployment/wiki-memory-service -n <namespace>

# 5. Tail logs for auth-related output
oc logs -f deployment/wiki-memory-service -n <namespace> | grep -E '\[teams\]|\[auth\]|401|403'

# 6. Re-login from CLI and check logs for team resolution
kwiki login
kwiki whoami
```

### ACL identity format

The service normalizes identity formats. These all match in ACL rules:

| JWT subject | ACL entry | Match? |
|-------------|-----------|--------|
| `github:aslom` | `github:user:aslom` | Yes (normalized) |
| `github:aslom` | `github:team:kaslomorg/ml-writers` | Yes (if user is in team) |
| `github:aslom` | `github:org:kaslomorg` | Yes (if user is org member) |

### Token renewal issues

- `kwiki renew` returns 404 — the server doesn't have the `/auth/renew` endpoint yet. Redeploy with the latest image.
- `kwiki renew` fails with "token too old" — the token is past the 7-day grace window. Use `kwiki login` instead.
- After renewal, `kwiki whoami` still shows old expiry — the CLI reads the locally cached token. Renewal updates the cache automatically.

### Rebuilding and deploying

```bash
cd mcp/wiki_memory_tool

# Build with version from pyproject.toml
VERSION=$(grep '^version' pyproject.toml | cut -d'"' -f2)
docker build --platform linux/amd64 -t quay.io/aslomnet/wiki-memory-service:$VERSION .
docker push quay.io/aslomnet/wiki-memory-service:$VERSION

# Deploy (rollout picks up the new image due to imagePullPolicy: Always)
oc rollout restart deployment/wiki-memory-service -n <namespace>
oc rollout status deployment/wiki-memory-service -n <namespace>

# Verify — response includes version
curl -s https://<wiki-service-url>/healthz
```
