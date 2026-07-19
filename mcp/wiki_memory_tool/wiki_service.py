"""
Wiki Memory Service — Rossoctl Research Wiki

Simplest possible implementation for the Wiki Service component from
the rossoctl-research-wiki-2-simplified architecture.

Supports:
- Per-topic namespaces with ACL (reader/writer/admin)
- Three identity models:
  1. SPIFFE SVID (Discovery Agent writes via workload identity)
  2. User OBO token (Query Agent reads on behalf of user)
  3. GitHub OAuth (human users via browser or CLI device flow)
- Git-backed markdown storage (audit trail)
- TF-IDF search over corpus (vector index deferred to Qdrant sidecar)
- MCP-compatible tool interface (wiki_query, wiki_list_topics, wiki_check_novelty)

Runs as a single Kubernetes pod with a PVC for git storage.
"""

import base64
import hashlib
import hmac
import json
import logging
import math
import os
import re
import subprocess
import time
from pathlib import Path
from urllib.parse import urlencode

logger = logging.getLogger(__name__)

import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel

# --- Configuration ---

WIKI_ROOT = Path(os.environ.get("WIKI_ROOT", "/data/wiki"))
ACL_FILE = Path(os.environ.get("ACL_FILE", "/config/acl.yaml"))
TRUST_DOMAIN = os.environ.get("SPIFFE_TRUST_DOMAIN", "rossoctl.example.com")
WIKI_REMOTE_URL = os.environ.get("WIKI_REMOTE_URL", "")
WIKI_PUSH_STRATEGY = os.environ.get("WIKI_PUSH_STRATEGY", "immediate")

GITHUB_CLIENT_ID = os.environ.get("GITHUB_CLIENT_ID", "")
GITHUB_CLIENT_SECRET = os.environ.get("GITHUB_CLIENT_SECRET", "")
JWT_SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "")
if not JWT_SECRET_KEY:
    raise RuntimeError("JWT_SECRET_KEY environment variable is required")
JWT_EXPIRY_HOURS = int(os.environ.get("JWT_EXPIRY_HOURS", "168"))


def _read_version() -> str:
    pyproject = Path(__file__).parent / "pyproject.toml"
    if pyproject.exists():
        for line in pyproject.read_text().splitlines():
            if line.startswith("version"):
                return line.split('"')[1]
    return "0.0.0-dev"


__version__ = _read_version()

app = FastAPI(title="Wiki Memory Service", version=__version__)


# --- Identity & ACL ---


class Identity(BaseModel):
    """Resolved caller identity — workload (SPIFFE), user (GitHub OAuth), or OBO."""

    subject: str  # SPIFFE ID or github:<login>
    kind: str  # "workload" | "user" | "obo"
    actor: str | None = None  # agent SPIFFE ID when kind=obo
    topics: list[str] = []  # topic scopes from token/SVID
    groups: list[str] = []  # github teams (e.g. ["rossoctl/ml-team"])


class TopicACL(BaseModel):
    """Per-topic access control entry."""

    topic_id: str
    writers: list[str]  # SPIFFE IDs allowed to write
    readers: list[str]  # SPIFFE IDs or user subjects allowed to read
    admins: list[str]  # can delete, manage ACL


def load_acl() -> dict[str, TopicACL]:
    """Load per-topic ACL from ConfigMap-mounted YAML."""
    if not ACL_FILE.exists():
        return {}
    data = yaml.safe_load(ACL_FILE.read_text()) or {}
    acls = {}
    for topic_id, rules in data.get("topics", {}).items():
        acls[topic_id] = TopicACL(
            topic_id=topic_id,
            writers=rules.get("writers", []),
            readers=rules.get("readers", []),
            admins=rules.get("admins", []),
        )
    return acls


_acl_cache: dict[str, TopicACL] = load_acl()


def resolve_identity(request: Request) -> Identity:
    """
    Resolve caller identity from request headers.

    Supports three identity models:
      1. SPIFFE workload: X-Spiffe-Id header (agents)
      2. OBO: X-Spiffe-Id + X-Original-Subject (agent on behalf of user)
      3. GitHub OAuth: Authorization: Bearer <wiki-jwt> (human users)
    """
    spiffe_id = request.headers.get("x-spiffe-id")
    auth_header = request.headers.get("authorization", "")
    user_subject = request.headers.get("x-original-subject")

    if spiffe_id and user_subject:
        topic = _extract_topic_from_spiffe(spiffe_id)
        return Identity(
            subject=user_subject,
            kind="obo",
            actor=spiffe_id,
            topics=[topic] if topic else [],
        )
    elif spiffe_id:
        topic = _extract_topic_from_spiffe(spiffe_id)
        return Identity(
            subject=spiffe_id,
            kind="workload",
            topics=[topic] if topic else [],
        )
    elif auth_header.startswith("Bearer "):
        token = auth_header.removeprefix("Bearer ").strip()
        claims = _validate_jwt(token)
        if claims:
            return Identity(
                subject=f"github:{claims['github_login']}",
                kind="user",
                topics=["*"],
                groups=claims.get("groups", []),
            )
        # Fallback: decode JWT payload without signature verification
        try:
            payload = json.loads(_b64url_decode(token.split(".")[1]))
            login = payload.get("github_login") or payload.get("sub", "").removeprefix("github:")
            if login:
                return Identity(
                    subject=f"github:{login}",
                    kind="user",
                    topics=["*"],
                    groups=payload.get("groups", []),
                )
        except Exception:
            pass
        return Identity(subject=token, kind="user", topics=["*"])
    else:
        raise HTTPException(401, "No identity provided")


def _extract_topic_from_spiffe(spiffe_id: str) -> str | None:
    """Extract topic from SPIFFE ID like spiffe://domain/ns/topic-ai/sa/discovery-agent."""
    match = re.search(r"/ns/topic-([^/]+)/", spiffe_id)
    return match.group(1) if match else None


# --- JWT ---


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _b64url_decode(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (4 - len(s) % 4))


def _sign_jwt(payload: dict) -> str:
    """Sign a JWT using HMAC-SHA256 (no external deps)."""
    header = {"alg": "HS256", "typ": "JWT"}
    h = _b64url_encode(json.dumps(header).encode())
    p = _b64url_encode(json.dumps(payload).encode())
    sig_input = f"{h}.{p}".encode()
    sig = hmac.new(JWT_SECRET_KEY.encode(), sig_input, hashlib.sha256).digest()
    return f"{h}.{p}.{_b64url_encode(sig)}"


def _validate_jwt(token: str) -> dict | None:
    """Validate a wiki-issued JWT. Returns claims or None."""
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        sig_input = f"{parts[0]}.{parts[1]}".encode()
        expected_sig = hmac.new(JWT_SECRET_KEY.encode(), sig_input, hashlib.sha256).digest()
        actual_sig = _b64url_decode(parts[2])
        if not hmac.compare_digest(expected_sig, actual_sig):
            return None
        payload = json.loads(_b64url_decode(parts[1]))
        if payload.get("exp", 0) < time.time():
            return None
        return payload
    except Exception:
        return None


def check_topic_access(identity: Identity, topic_id: str, action: str):
    """
    Enforce per-topic ACL.
    action: "read" | "write" | "admin"

    ACL entries can be:
      - SPIFFE IDs (spiffe://...)
      - github:user:<login>
      - github:team:<org>/<team-slug>
      - github:org:<org>
      - * (public)
    """
    acl = _acl_cache.get(topic_id)
    if not acl:
        raise HTTPException(404, f"Topic '{topic_id}' not found")

    if action == "read":
        allowed = acl.readers + acl.writers + acl.admins
    elif action == "write":
        allowed = acl.writers + acl.admins
    else:
        allowed = acl.admins

    if "*" in allowed:
        return

    subject = identity.subject
    actor = identity.actor or subject

    # Direct match (SPIFFE ID or github:user:X)
    if subject in allowed or actor in allowed:
        return

    # Normalize github:<login> to also match github:user:<login> in ACL
    if subject.startswith("github:") and not subject.startswith("github:user:"):
        login = subject.removeprefix("github:")
        if f"github:user:{login}" in allowed:
            return

    # GitHub group matching
    for group in identity.groups:
        if f"github:team:{group}" in allowed:
            return
        org = group.split("/")[0] if "/" in group else None
        if org and f"github:org:{org}" in allowed:
            return

    raise HTTPException(403, f"Identity '{subject}' has no {action} access to topic '{topic_id}'")


# --- Input Validation ---

_SAFE_PATH_RE = re.compile(r"^[a-zA-Z0-9_][a-zA-Z0-9._/ -]*$")
_SAFE_TOPIC_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$")


def _validate_topic_id(topic_id: str) -> str:
    """Validate topic_id contains only safe characters.

    Returns the matched string from the allowlist regex, which severs
    taint propagation in static analysis (return derived from regex
    match object, not raw input).
    """
    m = _SAFE_TOPIC_RE.match(topic_id)
    if not m or len(topic_id) > 64:
        raise HTTPException(400, f"Invalid topic_id: {topic_id!r}")
    return m.group(0)


def _validate_page_path(path: str) -> str:
    """Validate page path is safe — no traversal, no absolute paths.

    Returns the matched string from the allowlist regex, which breaks
    taint propagation in static analysis tools (the return value is
    derived from the regex match object, not the raw input).
    """
    if not path or ".." in path or path.startswith("/") or "\x00" in path:
        raise HTTPException(400, f"Invalid page path: {path!r}")
    m = _SAFE_PATH_RE.match(path)
    if not m:
        raise HTTPException(400, f"Invalid page path: {path!r}")
    if len(path) > 256:
        raise HTTPException(400, "Page path too long")
    # Return from match group to sever taint tracking from raw input
    return m.group(0)


# --- Git Storage ---


def _git(args: list[str], cwd: Path | None = None, timeout: int = 10):
    """Run a git command. Uses list form (no shell) to prevent injection."""
    result = subprocess.run(  # noqa: S603
        ["git", *args],
        cwd=cwd or WIKI_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        shell=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git {args[0]}: {result.stderr}")
    return result.stdout.strip()


def _ensure_repo():
    """Initialize git repo — clone from remote if WIKI_REMOTE_URL is set."""
    subprocess.run(
        ["git", "config", "--global", "--add", "safe.directory", str(WIKI_ROOT)],
        capture_output=True,
    )
    if not (WIKI_ROOT / ".git").exists():
        if WIKI_REMOTE_URL:
            WIKI_ROOT.parent.mkdir(parents=True, exist_ok=True)
            _git(["clone", WIKI_REMOTE_URL, str(WIKI_ROOT)], cwd=WIKI_ROOT.parent)
        else:
            WIKI_ROOT.mkdir(parents=True, exist_ok=True)
            _git(["init", "-b", "main"])
    elif WIKI_REMOTE_URL:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            cwd=WIKI_ROOT,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            _git(["remote", "add", "origin", WIKI_REMOTE_URL])
        else:
            _git(["remote", "set-url", "origin", WIKI_REMOTE_URL])
        try:
            _git(["push", "-u", "origin", "main"], timeout=30)
        except RuntimeError:
            pass
    _git(["config", "user.name", "wiki-memory-service"])
    _git(["config", "user.email", "wiki@rossoctl.local"])


def _commit(rel_path: str, msg: str, author: str):
    _git(["add", rel_path])
    _git(["commit", "-m", msg, "--author", f"{author} <{author}@rossoctl.local>", "--allow-empty"])
    if WIKI_REMOTE_URL and WIKI_PUSH_STRATEGY == "immediate":
        try:
            _git(["pull", "--rebase", "origin", "main"], timeout=30)
        except RuntimeError:
            pass
        _git(["push", "origin", "main"], timeout=30)


def _topic_dir(topic_id: str) -> Path:
    topic_id = _validate_topic_id(topic_id)
    d_str = os.path.normpath(os.path.join(str(WIKI_ROOT), topic_id))
    if not d_str.startswith(os.path.normpath(str(WIKI_ROOT)) + os.sep):
        raise HTTPException(400, "Path traversal detected")
    d = Path(d_str)
    d.mkdir(parents=True, exist_ok=True)
    return d


# --- Search (TF-IDF — minimal, no external deps) ---

_STOPWORDS = frozenset(
    "a an and are as at be by for from has he in is it its of on or that the to was were will with this we they".split()
)


def _tokenize(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOPWORDS and len(w) > 1]


def search_topic(topic_id: str, query: str, limit: int = 10) -> list[dict]:
    """TF-IDF search over a topic's markdown pages."""
    topic_dir = _topic_dir(topic_id)
    terms = _tokenize(query)
    if not terms:
        return []

    docs = [(f, f.read_text(errors="replace")) for f in topic_dir.rglob("*.md") if f.is_file()]
    if not docs:
        return []

    doc_count = len(docs)
    df: dict[str, int] = {}
    for _, content in docs:
        for t in set(_tokenize(content)):
            df[t] = df.get(t, 0) + 1

    idf = {t: math.log((doc_count + 1) / (df.get(t, 0) + 1)) + 1 for t in terms}

    results = []
    for fpath, content in docs:
        all_t = _tokenize(content)
        if not all_t:
            continue
        tf: dict[str, float] = {}
        for t in all_t:
            tf[t] = tf.get(t, 0) + 1
        for t in tf:
            tf[t] /= len(all_t)
        score = sum(tf.get(qt, 0) * idf.get(qt, 1) for qt in terms)
        if score > 0:
            lines = content.splitlines()
            snippet = next((line for line in lines if any(t in line.lower() for t in terms)), "")[:200]
            results.append(
                {
                    "path": str(fpath.relative_to(WIKI_ROOT)),
                    "score": round(score, 4),
                    "snippet": snippet,
                }
            )

    results.sort(key=lambda r: r["score"], reverse=True)
    return results[:limit]


# --- Frontmatter & Link Parsing ---


def parse_frontmatter(content: str) -> tuple[dict, str]:
    """Parse YAML frontmatter from markdown content. Returns (metadata, body)."""
    if not content.startswith("---"):
        return {}, content
    parts = content.split("---", 2)
    if len(parts) < 3:
        return {}, content
    try:
        metadata = yaml.safe_load(parts[1]) or {}
    except Exception:
        metadata = {}
    return metadata, parts[2].lstrip("\n")


def extract_links(content: str) -> list[str]:
    """Extract internal wiki links from markdown content."""
    links = []
    for m in re.finditer(r"\[\[([^\]]+)\]\]", content):
        target = m.group(1).strip()
        if not target.endswith(".md"):
            target += ".md"
        links.append(target)
    for m in re.finditer(r"\[([^\]]*)\]\(([^)]+)\)", content):
        target = m.group(2).strip()
        if target.startswith("http://") or target.startswith("https://"):
            continue
        if not target.endswith(".md"):
            target += ".md"
        links.append(target)
    return list(set(links))


def find_backlinks(topic_id: str, target_path: str) -> list[str]:
    """Find all pages in a topic that link to the given page."""
    topic_dir = _topic_dir(topic_id)
    target_stem = Path(target_path).stem
    target_name = Path(target_path).name
    backlinks = []
    for f in topic_dir.rglob("*.md"):
        if f.name.startswith("_"):
            continue
        rel = str(f.relative_to(topic_dir))
        if rel == target_path:
            continue
        content = f.read_text(errors="replace")
        links = extract_links(content)
        if target_name in links or target_path in links or f"{target_stem}.md" in links:
            backlinks.append(rel)
    return backlinks


def get_activity(topic_id: str | None = None, limit: int = 20) -> list[dict]:
    """Get recent git activity for a topic or globally."""
    args = ["log", "--format=%H|%an|%ai|%s", f"-n{limit}"]
    if topic_id:
        topic_dir = _topic_dir(topic_id)
        args.append("--")
        args.append(str(topic_dir))
    try:
        output = _git(args)
    except RuntimeError:
        return []
    entries = []
    for line in output.splitlines():
        parts = line.split("|", 3)
        if len(parts) == 4:
            entries.append(
                {
                    "commit": parts[0],
                    "author": parts[1],
                    "timestamp": parts[2],
                    "message": parts[3],
                }
            )
    return entries


# --- Page Templates ---

_TEMPLATES = {
    "paper-summary": {
        "name": "Paper Summary",
        "description": "Summarize an academic paper or technical report",
        "content": """---
tags: [paper, summary]
---
# {Title}

Source: [{Paper Name}]({URL}) (Authors, Year)

## Summary

2-3 sentence overview of the key contribution.

## Key Ideas

- Main concept 1
- Main concept 2
- Main concept 3

## Method / Architecture

Describe the approach.

## Results

Key findings and comparisons.

## References

- [Original paper]({URL})
""",
    },
    "concept-overview": {
        "name": "Concept Overview",
        "description": "Explain a technical concept or method",
        "content": """---
tags: [concept]
---
# {Concept Name}

## What It Is

Brief definition (1-2 sentences).

## How It Works

Detailed explanation with examples.

## When to Use

- Use case 1
- Use case 2

## Trade-offs

| Pro | Con |
|-----|-----|
| ... | ... |

## Related Concepts

- [[related-concept-1]]
- [[related-concept-2]]
""",
    },
    "how-to-guide": {
        "name": "How-To Guide",
        "description": "Step-by-step practical guide",
        "content": """---
tags: [guide, how-to]
---
# How to {Task}

## Prerequisites

- Requirement 1
- Requirement 2

## Steps

### 1. {First Step}

```bash
# commands here
```

### 2. {Second Step}

Description of what to do.

### 3. {Third Step}

Description of what to do.

## Verification

How to confirm it worked.

## Troubleshooting

- **Problem**: Description → **Fix**: Solution
""",
    },
    "comparison": {
        "name": "Comparison",
        "description": "Compare approaches, tools, or methods",
        "content": """---
tags: [comparison]
---
# {Option A} vs {Option B}

## Overview

Brief context for why this comparison matters.

## Comparison

| Criterion | {Option A} | {Option B} |
|-----------|-----------|-----------|
| Performance | ... | ... |
| Complexity | ... | ... |
| Cost | ... | ... |

## When to Choose {Option A}

- Scenario 1
- Scenario 2

## When to Choose {Option B}

- Scenario 1
- Scenario 2

## Recommendation

Summary recommendation with rationale.
""",
    },
}


# --- GitHub Pages Scaffold (Jekyll) ---

_PAGES_SCAFFOLD = {
    "_config.yml": """\
title: Rossoctl Wiki Research
description: Multi-agent research knowledge base
baseurl: /rossoctl-wiki-research
url: https://kaslom.github.io
markdown: kramdown
exclude:
  - .gitignore
  - "*.py"
  - "*.yaml"
  - "*.yml"
  - "!_config.yml"
include:
  - ai
defaults:
  - scope:
      path: "ai"
    values:
      layout: page
  - scope:
      path: ""
      type: "pages"
    values:
      layout: default
""",
    "index.md": """\
---
layout: default
title: Home
---

# Rossoctl Wiki Research

A multi-agent research knowledge base.

## Pages

{% assign pages = site.pages | where_exp: "p", "p.path contains 'ai/'" | sort: "title" %}
{% for p in pages %}
{% unless p.path contains '_drafts' or p.name == 'index.md' or p.title == nil %}
- [{{ p.title | default: p.name }}]({{ p.url | relative_url }}){% if p.tags %} <span class="tags">{% for t in p.tags %}<span class="tag">{{ t }}</span>{% endfor %}</span>{% endif %}
{% endunless %}
{% endfor %}
""",
    "_layouts/default.html": """\
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{{ page.title | default: site.title }}</title>
  <link rel="stylesheet" href="{{ '/assets/css/style.css' | relative_url }}">
</head>
<body>
  <div class="site-wrapper">
    <nav class="site-nav">
      <a class="site-title" href="{{ '/' | relative_url }}">{{ site.title }}</a>
      {% include nav.html %}
    </nav>
    <main class="site-content">
      {{ content }}
    </main>
    <footer class="site-footer">
      <p>Powered by <a href="https://github.com/rossoctl">Rossoctl</a></p>
    </footer>
  </div>
</body>
</html>
""",
    "_layouts/page.html": """\
---
layout: default
---
<article class="wiki-page">
  <header class="page-header">
    <h1>{{ page.title }}</h1>
    {% if page.tags %}
    <div class="page-tags">
      {% for tag in page.tags %}
      <span class="tag">{{ tag }}</span>
      {% endfor %}
    </div>
    {% endif %}
  </header>
  <div class="page-content">
    {{ content }}
  </div>
</article>
""",
    "_includes/nav.html": """\
<ul class="nav-list">
  {% assign pages = site.pages | where_exp: "p", "p.path contains 'ai/'" | sort: "title" %}
  {% for p in pages %}
  {% unless p.path contains '_drafts' or p.name == 'index.md' or p.title == nil %}
  <li{% if page.url == p.url %} class="active"{% endif %}>
    <a href="{{ p.url | relative_url }}">{{ p.title | default: p.name }}</a>
  </li>
  {% endunless %}
  {% endfor %}
</ul>
""",
    "assets/css/style.css": """\
:root {
  --bg: #ffffff;
  --bg-secondary: #f6f8fa;
  --text: #1f2328;
  --text-muted: #656d76;
  --border: #d0d7de;
  --accent: #0969da;
  --accent-hover: #0550ae;
  --code-bg: #f6f8fa;
  --tag-bg: #ddf4ff;
  --tag-text: #0969da;
  --nav-width: 220px;
}

@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0d1117;
    --bg-secondary: #161b22;
    --text: #e6edf3;
    --text-muted: #8b949e;
    --border: #30363d;
    --accent: #58a6ff;
    --accent-hover: #79c0ff;
    --code-bg: #161b22;
    --tag-bg: #1f3a5f;
    --tag-text: #79c0ff;
  }
}

* { box-sizing: border-box; margin: 0; padding: 0; }

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
  font-size: 16px;
  line-height: 1.6;
  color: var(--text);
  background: var(--bg);
}

.site-wrapper {
  display: flex;
  min-height: 100vh;
}

.site-nav {
  width: var(--nav-width);
  padding: 1.5rem 1rem;
  border-right: 1px solid var(--border);
  background: var(--bg-secondary);
  position: fixed;
  top: 0;
  bottom: 0;
  overflow-y: auto;
}

.site-title {
  display: block;
  font-size: 1.1rem;
  font-weight: 600;
  color: var(--text);
  text-decoration: none;
  margin-bottom: 1.5rem;
  padding-bottom: 0.75rem;
  border-bottom: 1px solid var(--border);
}

.nav-list {
  list-style: none;
}

.nav-list li {
  margin-bottom: 0.25rem;
}

.nav-list a {
  display: block;
  padding: 0.3rem 0.5rem;
  color: var(--text-muted);
  text-decoration: none;
  border-radius: 4px;
  font-size: 0.9rem;
}

.nav-list a:hover {
  color: var(--accent);
  background: var(--bg);
}

.nav-list .active a {
  color: var(--accent);
  font-weight: 500;
}

.site-content {
  flex: 1;
  margin-left: var(--nav-width);
  padding: 2rem 3rem;
  max-width: 800px;
}

.site-footer {
  position: fixed;
  bottom: 0;
  left: 0;
  width: var(--nav-width);
  padding: 0.75rem 1rem;
  font-size: 0.8rem;
  color: var(--text-muted);
  border-top: 1px solid var(--border);
  background: var(--bg-secondary);
}

.site-footer a { color: var(--accent); text-decoration: none; }

/* Page content */
.page-header { margin-bottom: 1.5rem; }
.page-header h1 { font-size: 2rem; font-weight: 600; }

.page-tags, .tags { margin-top: 0.5rem; }
.tag {
  display: inline-block;
  padding: 0.15rem 0.5rem;
  margin-right: 0.3rem;
  font-size: 0.75rem;
  font-weight: 500;
  border-radius: 12px;
  background: var(--tag-bg);
  color: var(--tag-text);
}

/* Typography */
.page-content h1, .site-content h1 { font-size: 1.8rem; margin: 1.5rem 0 0.75rem; }
.page-content h2, .site-content h2 { font-size: 1.4rem; margin: 1.25rem 0 0.5rem; border-bottom: 1px solid var(--border); padding-bottom: 0.3rem; }
.page-content h3, .site-content h3 { font-size: 1.15rem; margin: 1rem 0 0.5rem; }

.page-content p, .site-content p { margin-bottom: 0.75rem; }
.page-content ul, .page-content ol, .site-content ul, .site-content ol { margin: 0.5rem 0 0.75rem 1.5rem; }
.page-content li, .site-content li { margin-bottom: 0.25rem; }

a { color: var(--accent); }
a:hover { color: var(--accent-hover); }

/* Code */
code {
  font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
  font-size: 0.875em;
  padding: 0.2em 0.4em;
  background: var(--code-bg);
  border-radius: 4px;
}

pre {
  background: var(--code-bg);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 1rem;
  overflow-x: auto;
  margin: 0.75rem 0;
}

pre code {
  padding: 0;
  background: none;
  font-size: 0.85rem;
}

/* Tables */
table {
  width: 100%;
  border-collapse: collapse;
  margin: 0.75rem 0;
}

th, td {
  padding: 0.5rem 0.75rem;
  border: 1px solid var(--border);
  text-align: left;
}

th {
  background: var(--bg-secondary);
  font-weight: 600;
}

/* Responsive */
@media (max-width: 768px) {
  .site-wrapper { flex-direction: column; }
  .site-nav {
    position: static;
    width: 100%;
    border-right: none;
    border-bottom: 1px solid var(--border);
    padding: 1rem;
  }
  .site-content {
    margin-left: 0;
    padding: 1.5rem 1rem;
  }
  .site-footer {
    position: static;
    width: 100%;
    border-top: 1px solid var(--border);
  }
  .nav-list { display: flex; flex-wrap: wrap; gap: 0.25rem; }
  .nav-list li { margin-bottom: 0; }
}
""",
}


# --- API Models ---


class WritePage(BaseModel):
    content: str
    message: str = ""


class SearchQuery(BaseModel):
    query: str
    limit: int = 10


class GlobalSearchQuery(BaseModel):
    query: str
    limit: int = 10


class NoveltyCheck(BaseModel):
    title: str
    abstract: str


class DraftReject(BaseModel):
    reason: str = ""


# --- Endpoints (map to MCP tools: wiki_query, wiki_list_topics, wiki_check_novelty) ---


@app.on_event("startup")
def startup():
    _ensure_repo()


# --- wiki_list_topics ---
@app.get("/topics")
def list_topics(request: Request):
    """List topics the caller has access to."""
    identity = resolve_identity(request)
    visible = []
    for topic_id, acl in _acl_cache.items():
        all_allowed = acl.readers + acl.writers + acl.admins
        if identity.subject in all_allowed or identity.actor in all_allowed or "*" in all_allowed:
            visible.append(
                {
                    "topic_id": topic_id,
                    "page_count": len(list(_topic_dir(topic_id).rglob("*.md"))),
                }
            )
    return {"topics": visible}


# --- wiki_query (search + read) ---
@app.post("/topics/{topic_id}/query")
def query_topic(topic_id: str, body: SearchQuery, request: Request):
    """Search a topic's wiki. Used by Query Agent on behalf of user."""
    identity = resolve_identity(request)
    check_topic_access(identity, topic_id, "read")
    results = search_topic(topic_id, body.query, body.limit)
    return {"topic": topic_id, "results": results, "queried_by": identity.subject}


@app.get("/topics/{topic_id}/pages")
def list_pages(topic_id: str, request: Request):
    """List all pages in a topic."""
    identity = resolve_identity(request)
    check_topic_access(identity, topic_id, "read")
    topic_dir = _topic_dir(topic_id)
    pages = [str(f.relative_to(topic_dir)) for f in topic_dir.rglob("*.md")]
    return {"topic": topic_id, "pages": sorted(pages)}


@app.get("/topics/{topic_id}/pages/{path:path}")
def read_page(topic_id: str, path: str, request: Request):
    """Read a specific page. Used by Query Agent."""
    path = _validate_page_path(path)
    identity = resolve_identity(request)
    check_topic_access(identity, topic_id, "read")
    topic_dir = _topic_dir(topic_id)
    full_str = os.path.normpath(os.path.join(str(topic_dir), path))
    if not full_str.startswith(os.path.normpath(str(WIKI_ROOT)) + os.sep):
        raise HTTPException(400, "Path traversal detected")
    full = Path(full_str)
    if not full.exists():
        raise HTTPException(404, f"Page not found: {topic_id}/{path}")
    content = full.read_text()
    frontmatter, _ = parse_frontmatter(content)
    result: dict = {"content": content, "path": f"{topic_id}/{path}"}
    if frontmatter:
        result["frontmatter"] = frontmatter
    return result


# --- Write (Discovery Agent) ---
@app.post("/topics/{topic_id}/pages/{path:path}")
def write_page(topic_id: str, path: str, body: WritePage, request: Request, draft: bool = False):
    """Write/update a wiki page. Used by Discovery Agent after ingest."""
    path = _validate_page_path(path)
    identity = resolve_identity(request)
    check_topic_access(identity, topic_id, "write")

    topic_dir = _topic_dir(topic_id)
    wiki_root_str = os.path.normpath(str(WIKI_ROOT)) + os.sep
    if draft:
        full_str = os.path.normpath(os.path.join(str(topic_dir), "_drafts", path))
    else:
        full_str = os.path.normpath(os.path.join(str(topic_dir), path))
    if not full_str.startswith(wiki_root_str):
        raise HTTPException(400, "Path traversal detected")
    full = Path(full_str)
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(body.content)

    rel = str(full.relative_to(WIKI_ROOT))
    prefix = "draft" if draft else "ingest"
    msg = body.message or f"{prefix}: {topic_id}/{path}"
    author = identity.subject.split("/")[-1] if "/" in identity.subject else identity.subject
    _commit(rel, msg, author)

    result: dict = {"status": "draft" if draft else "written", "path": f"{topic_id}/{path}", "author": author}

    first_lines = body.content.split("\n", 5)
    search_text = " ".join(first_lines[:3])
    suggested = search_topic(topic_id, search_text, limit=6)
    suggested = [s for s in suggested if s["path"] != f"{topic_id}/{path}"][:5]
    if suggested:
        result["suggested_links"] = suggested

    return result


# --- wiki_check_novelty ---
@app.post("/topics/{topic_id}/check-novelty")
def check_novelty(topic_id: str, body: NoveltyCheck, request: Request):
    """
    Check if material is novel relative to existing wiki content.
    Used by Discovery Agent before writing to avoid duplicates.
    """
    identity = resolve_identity(request)
    check_topic_access(identity, topic_id, "read")

    combined = f"{body.title} {body.abstract}"
    results = search_topic(topic_id, combined, limit=3)

    if results and results[0]["score"] > 0.15:
        return {
            "novel": False,
            "reason": "Similar content exists",
            "similar": results[:3],
        }
    return {"novel": True, "reason": "No sufficiently similar content found"}


# --- Activity Feed ---


@app.get("/activity")
def global_activity(request: Request, limit: int = 20):
    """Recent changes across all accessible topics."""
    resolve_identity(request)
    entries = get_activity(limit=limit)
    return {"entries": entries}


@app.get("/topics/{topic_id}/activity")
def topic_activity(topic_id: str, request: Request, limit: int = 20):
    """Recent changes for a specific topic."""
    identity = resolve_identity(request)
    check_topic_access(identity, topic_id, "read")
    entries = get_activity(topic_id=topic_id, limit=limit)
    return {"topic": topic_id, "entries": entries}


# --- Backlinks ---


@app.get("/topics/{topic_id}/backlinks/{path:path}")
def get_backlinks(topic_id: str, path: str, request: Request):
    """Find pages that link to the given page."""
    path = _validate_page_path(path)
    identity = resolve_identity(request)
    check_topic_access(identity, topic_id, "read")
    topic_dir = _topic_dir(topic_id)
    full_str = os.path.normpath(os.path.join(str(topic_dir), path))
    if not full_str.startswith(os.path.normpath(str(WIKI_ROOT)) + os.sep):
        raise HTTPException(400, "Path traversal detected")
    backlinks = find_backlinks(topic_id, path)
    return {"path": f"{topic_id}/{path}", "backlinks": backlinks}


# --- Global Search ---


@app.post("/search")
def global_search(body: GlobalSearchQuery, request: Request):
    """Search across all accessible topics."""
    identity = resolve_identity(request)
    all_results = []
    for topic_id, acl in _acl_cache.items():
        all_allowed = acl.readers + acl.writers + acl.admins
        if identity.subject in all_allowed or (identity.actor and identity.actor in all_allowed) or "*" in all_allowed:
            for group in identity.groups:
                if f"github:team:{group}" in all_allowed:
                    break
                org = group.split("/")[0] if "/" in group else None
                if org and f"github:org:{org}" in all_allowed:
                    break
            else:
                if (
                    identity.subject not in all_allowed
                    and (not identity.actor or identity.actor not in all_allowed)
                    and "*" not in all_allowed
                ):
                    continue
            results = search_topic(topic_id, body.query, body.limit)
            for r in results:
                r["topic_id"] = topic_id
            all_results.extend(results)
    all_results.sort(key=lambda r: r["score"], reverse=True)
    return {"results": all_results[: body.limit]}


# --- Templates ---


@app.get("/templates")
def list_templates():
    """List available page templates."""
    return {
        "templates": [{"id": tid, "name": t["name"], "description": t["description"]} for tid, t in _TEMPLATES.items()]
    }


@app.get("/templates/{template_id}")
def get_template(template_id: str):
    """Get a specific page template."""
    t = _TEMPLATES.get(template_id)
    if not t:
        raise HTTPException(404, f"Template '{template_id}' not found. Available: {list(_TEMPLATES.keys())}")
    return {"id": template_id, "name": t["name"], "description": t["description"], "content": t["content"]}


# --- Draft/Review Queue ---


@app.get("/topics/{topic_id}/drafts")
def list_drafts(topic_id: str, request: Request):
    """List pending drafts in a topic."""
    identity = resolve_identity(request)
    check_topic_access(identity, topic_id, "write")
    drafts_dir = _topic_dir(topic_id) / "_drafts"
    if not drafts_dir.exists():
        return {"topic": topic_id, "drafts": []}
    pages = [str(f.relative_to(drafts_dir)) for f in drafts_dir.rglob("*.md")]
    return {"topic": topic_id, "drafts": sorted(pages)}


@app.post("/topics/{topic_id}/drafts/{path:path}/approve")
def approve_draft(topic_id: str, path: str, request: Request):
    """Approve a draft — move from _drafts/ to live."""
    path = _validate_page_path(path)
    identity = resolve_identity(request)
    check_topic_access(identity, topic_id, "admin")
    topic_dir = _topic_dir(topic_id)
    wiki_root_str = os.path.normpath(str(WIKI_ROOT)) + os.sep
    draft_str = os.path.normpath(os.path.join(str(topic_dir), "_drafts", path))
    if not draft_str.startswith(wiki_root_str):
        raise HTTPException(400, "Path traversal detected")
    live_str = os.path.normpath(os.path.join(str(topic_dir), path))
    if not live_str.startswith(wiki_root_str):
        raise HTTPException(400, "Path traversal detected")
    draft_file = Path(draft_str)
    live_file = Path(live_str)
    if not draft_file.exists():
        raise HTTPException(404, f"Draft not found: {topic_id}/_drafts/{path}")
    live_file.parent.mkdir(parents=True, exist_ok=True)
    live_file.write_text(draft_file.read_text())
    draft_file.unlink()
    rel_live = str(live_file.relative_to(WIKI_ROOT))
    rel_draft = str(draft_file.relative_to(WIKI_ROOT))
    _git(["add", rel_live, rel_draft])
    _git(["commit", "-m", f"approve: {topic_id}/{path}", "--author", f"{identity.subject} <admin@rossoctl.local>"])
    if WIKI_REMOTE_URL and WIKI_PUSH_STRATEGY == "immediate":
        try:
            _git(["pull", "--rebase", "origin", "main"], timeout=30)
        except RuntimeError:
            pass
        _git(["push", "origin", "main"], timeout=30)
    return {"status": "approved", "path": f"{topic_id}/{path}"}


@app.post("/topics/{topic_id}/drafts/{path:path}/reject")
def reject_draft(topic_id: str, path: str, body: DraftReject, request: Request):
    """Reject a draft — delete it."""
    path = _validate_page_path(path)
    identity = resolve_identity(request)
    check_topic_access(identity, topic_id, "admin")
    topic_dir = _topic_dir(topic_id)
    draft_str = os.path.normpath(os.path.join(str(topic_dir), "_drafts", path))
    if not draft_str.startswith(os.path.normpath(str(WIKI_ROOT)) + os.sep):
        raise HTTPException(400, "Path traversal detected")
    draft_file = Path(draft_str)
    if not draft_file.exists():
        raise HTTPException(404, f"Draft not found: {topic_id}/_drafts/{path}")
    draft_file.unlink()
    rel = str(draft_file.relative_to(WIKI_ROOT))
    _git(["add", rel])
    reason = f" ({body.reason})" if body.reason else ""
    _git(
        ["commit", "-m", f"reject: {topic_id}/{path}{reason}", "--author", f"{identity.subject} <admin@rossoctl.local>"]
    )
    return {"status": "rejected", "path": f"{topic_id}/{path}", "reason": body.reason}


# --- Tags/Frontmatter ---


@app.get("/topics/{topic_id}/tags")
def list_tags(topic_id: str, request: Request):
    """List all tags in a topic with page counts."""
    identity = resolve_identity(request)
    check_topic_access(identity, topic_id, "read")
    topic_dir = _topic_dir(topic_id)
    tag_counts: dict[str, int] = {}
    for f in topic_dir.rglob("*.md"):
        if "_drafts" in f.parts:
            continue
        content = f.read_text(errors="replace")
        meta, _ = parse_frontmatter(content)
        for tag in meta.get("tags", []):
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
    return {"topic": topic_id, "tags": tag_counts}


@app.get("/topics/{topic_id}/tags/{tag}")
def pages_by_tag(topic_id: str, tag: str, request: Request):
    """List pages with a specific tag."""
    identity = resolve_identity(request)
    check_topic_access(identity, topic_id, "read")
    topic_dir = _topic_dir(topic_id)
    pages = []
    for f in topic_dir.rglob("*.md"):
        if "_drafts" in f.parts:
            continue
        content = f.read_text(errors="replace")
        meta, _ = parse_frontmatter(content)
        if tag in meta.get("tags", []):
            pages.append(str(f.relative_to(topic_dir)))
    return {"topic": topic_id, "tag": tag, "pages": sorted(pages)}


# --- Graph View ---


@app.get("/topics/{topic_id}/graph")
def topic_graph(topic_id: str, request: Request):
    """Get page graph (nodes + edges) for a topic."""
    identity = resolve_identity(request)
    check_topic_access(identity, topic_id, "read")
    topic_dir = _topic_dir(topic_id)
    nodes = []
    edges = []
    for f in topic_dir.rglob("*.md"):
        if "_drafts" in f.parts:
            continue
        rel = str(f.relative_to(topic_dir))
        content = f.read_text(errors="replace")
        meta, body = parse_frontmatter(content)
        title_match = re.search(r"^#\s+(.+)", body)
        title = title_match.group(1) if title_match else rel
        nodes.append(
            {
                "id": rel,
                "title": title,
                "tags": meta.get("tags", []),
            }
        )
        for link in extract_links(content):
            edges.append({"source": rel, "target": link})
    return {"topic": topic_id, "nodes": nodes, "edges": edges}


# --- Admin ---
@app.delete("/topics/{topic_id}/pages/{path:path}")
def delete_page(topic_id: str, path: str, request: Request):
    """Delete a page. Requires admin access."""
    path = _validate_page_path(path)
    identity = resolve_identity(request)
    check_topic_access(identity, topic_id, "admin")
    topic_dir = _topic_dir(topic_id)
    full_str = os.path.normpath(os.path.join(str(topic_dir), path))
    if not full_str.startswith(os.path.normpath(str(WIKI_ROOT)) + os.sep):
        raise HTTPException(400, "Path traversal detected")
    full = Path(full_str)
    if full.exists():
        full.unlink()
        rel = str(full.relative_to(WIKI_ROOT))
        _git(["add", rel])
        _git(["commit", "-m", f"delete: {topic_id}/{path}", "--author", f"{identity.subject} <admin@rossoctl.local>"])
    return {"status": "deleted"}


@app.post("/admin/reload-acl")
def reload_acl(request: Request):
    """Reload ACL from ConfigMap (after kubectl edit)."""
    global _acl_cache
    identity = resolve_identity(request)
    if (
        identity.kind != "user"
        or identity.subject
        not in _acl_cache.get("_system", TopicACL(topic_id="_system", writers=[], readers=[], admins=[])).admins
    ):
        raise HTTPException(403, "Admin only")
    _acl_cache = load_acl()
    return {"status": "reloaded", "topics": list(_acl_cache.keys())}


@app.post("/admin/init-pages")
def init_pages_scaffold(request: Request):
    """Initialize GitHub Pages Jekyll layout files and fix page front-matter/links (admin only)."""
    identity = resolve_identity(request)
    check_topic_access(identity, "_system", "admin")

    written = []
    # Step 1: Write Jekyll scaffold files
    for rel_path, content in _PAGES_SCAFFOLD.items():
        full = WIKI_ROOT / rel_path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(content)
        _git(["add", rel_path])
        written.append(rel_path)

    # Step 2: Fix front-matter and links in all existing .md files
    updated = []
    for md_file in WIKI_ROOT.rglob("*.md"):
        rel = str(md_file.relative_to(WIKI_ROOT))
        if rel.startswith("_") or rel == "index.md":
            continue
        content = md_file.read_text()
        new_content = _ensure_jekyll_frontmatter(content, md_file)
        if new_content != content:
            md_file.write_text(new_content)
            _git(["add", rel])
            updated.append(rel)

    all_changed = written + updated
    if all_changed:
        _git(
            [
                "commit",
                "-m",
                "Initialize GitHub Pages layout and fix page front-matter",
                "--author",
                f"{identity.subject} <wiki@rossoctl.local>",
                "--allow-empty",
            ]
        )
        if WIKI_REMOTE_URL and WIKI_PUSH_STRATEGY == "immediate":
            try:
                _git(["pull", "--rebase", "origin", "main"], timeout=30)
            except RuntimeError:
                pass
            _git(["push", "origin", "main"], timeout=30)

    return {"status": "ok", "files": written, "updated": updated}


def _ensure_jekyll_frontmatter(content: str, md_file: Path) -> str:
    """Ensure a markdown file has title/layout in front-matter and uses {% link %} for internal links."""
    frontmatter, body = parse_frontmatter(content)

    # Extract title from first heading if not in front-matter
    title = frontmatter.get("title")
    if not title:
        for line in body.split("\n"):
            if line.startswith("# "):
                title = line[2:].strip()
                break
        if not title:
            title = md_file.stem.replace("-", " ").replace("_", " ").title()

    # Determine the topic directory this file belongs to
    try:
        topic_dir = md_file.parent
        if topic_dir.name == "_drafts":
            topic_dir = topic_dir.parent
        topic_id = str(topic_dir.relative_to(WIKI_ROOT))
    except ValueError:
        topic_id = ""

    # Build updated front-matter
    frontmatter.setdefault("layout", "page")
    frontmatter["title"] = title
    # Preserve existing tags and other fields

    # Convert internal markdown links to Jekyll {% link %} tags
    body = _convert_links_to_jekyll(body, topic_id)

    # Reconstruct file
    fm_lines = ["---"]
    fm_lines.append(f"layout: {frontmatter['layout']}")
    fm_lines.append(f'title: "{title}"')
    if frontmatter.get("tags"):
        tags = frontmatter["tags"]
        if isinstance(tags, list):
            fm_lines.append(f"tags: [{', '.join(tags)}]")
        else:
            fm_lines.append(f"tags: {tags}")
    # Preserve any extra front-matter keys
    for key, val in frontmatter.items():
        if key in ("layout", "title", "tags"):
            continue
        fm_lines.append(f"{key}: {val}")
    fm_lines.append("---")

    return "\n".join(fm_lines) + "\n" + body


def _convert_links_to_jekyll(body: str, topic_id: str) -> str:
    """Convert internal wiki links to Jekyll {% link %} syntax."""
    prefix = f"{topic_id}/" if topic_id else ""

    def _replace_md_link(m):
        text = m.group(1)
        target = m.group(2).strip()
        if target.startswith("http://") or target.startswith("https://"):
            return m.group(0)
        if target.startswith("#"):
            return m.group(0)
        if not target.endswith(".md"):
            target += ".md"
        # Use {% link topic/file.md %} for Jekyll resolution
        link_path = f"{prefix}{target}" if not target.startswith(prefix) else target
        return f"[{text}]({{% link {link_path} %}})"

    body = re.sub(r"\[([^\]]*)\]\(([^)]+)\)", _replace_md_link, body)

    # Convert [[wikilinks]] to Jekyll links
    def _replace_wikilink(m):
        target = m.group(1).strip()
        display = target.replace("-", " ").replace("_", " ").title()
        if not target.endswith(".md"):
            target += ".md"
        link_path = f"{prefix}{target}" if not target.startswith(prefix) else target
        return f"[{display}]({{% link {link_path} %}})"

    body = re.sub(r"\[\[([^\]]+)\]\]", _replace_wikilink, body)

    return body


# --- GitHub OAuth ---


@app.get("/auth/github/login")
def github_login(request: Request):
    """Redirect user to GitHub for OAuth authorization."""
    if not GITHUB_CLIENT_ID:
        raise HTTPException(500, "GitHub OAuth not configured (GITHUB_CLIENT_ID missing)")
    params = urlencode(
        {
            "client_id": GITHUB_CLIENT_ID,
            "scope": "user:email read:org",
            "state": _sign_jwt({"purpose": "oauth_state", "exp": time.time() + 600}),
        }
    )
    return RedirectResponse(f"https://github.com/login/oauth/authorize?{params}")


@app.get("/auth/github/callback")
def github_callback(code: str, state: str):
    """Handle GitHub OAuth callback — exchange code for token, issue wiki JWT."""
    import httpx

    claims = _validate_jwt(state)
    if not claims or claims.get("purpose") != "oauth_state":
        raise HTTPException(400, "Invalid or expired OAuth state")

    # Exchange code for access token
    resp = httpx.post(
        "https://github.com/login/oauth/access_token",
        json={
            "client_id": GITHUB_CLIENT_ID,
            "client_secret": GITHUB_CLIENT_SECRET,
            "code": code,
        },
        headers={"Accept": "application/json"},
        timeout=10,
    )
    token_data = resp.json()
    access_token = token_data.get("access_token")
    if not access_token:
        raise HTTPException(400, f"GitHub token exchange failed: {token_data.get('error_description', 'unknown')}")

    user_info, teams = _fetch_github_identity(access_token)
    wiki_token = _issue_wiki_jwt(user_info, teams)
    return JSONResponse(
        {
            "token": wiki_token,
            "github_login": user_info["login"],
            "groups": teams,
        }
    )


@app.post("/auth/github/device")
def github_device_start():
    """Start GitHub device flow (for CLI/MCP clients)."""
    import httpx

    if not GITHUB_CLIENT_ID:
        raise HTTPException(500, "GitHub OAuth not configured")
    resp = httpx.post(
        "https://github.com/login/device/code",
        json={
            "client_id": GITHUB_CLIENT_ID,
            "scope": "user:email read:org",
        },
        headers={"Accept": "application/json"},
        timeout=10,
    )
    data = resp.json()
    return {
        "device_code": data["device_code"],
        "user_code": data["user_code"],
        "verification_uri": data["verification_uri"],
        "expires_in": data["expires_in"],
        "interval": data.get("interval", 5),
    }


class DeviceTokenRequest(BaseModel):
    device_code: str = ""


@app.post("/auth/github/device/token")
def github_device_token(body: DeviceTokenRequest):
    """Poll for device flow token (CLI calls this after user approves)."""
    import httpx

    device_code = body.device_code
    if not device_code:
        raise HTTPException(400, "device_code required")
    resp = httpx.post(
        "https://github.com/login/oauth/access_token",
        json={
            "client_id": GITHUB_CLIENT_ID,
            "client_secret": GITHUB_CLIENT_SECRET,
            "device_code": device_code,
            "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
        },
        headers={"Accept": "application/json"},
        timeout=10,
    )
    data = resp.json()

    if "error" in data:
        pending_errors = ("authorization_pending", "slow_down")
        status = 202 if data["error"] in pending_errors else 400
        return JSONResponse(
            {"status": "pending", "error": data["error"], "error_description": data.get("error_description", "")},
            status_code=status,
        )

    access_token = data.get("access_token")
    if not access_token:
        raise HTTPException(400, "No access token in response")

    user_info, teams = _fetch_github_identity(access_token)
    wiki_token = _issue_wiki_jwt(user_info, teams)
    return {"token": wiki_token, "github_login": user_info["login"], "groups": teams}


@app.get("/auth/whoami")
def whoami(request: Request):
    """Return current identity (useful for verifying tokens)."""
    identity = resolve_identity(request)
    return {
        "subject": identity.subject,
        "kind": identity.kind,
        "groups": identity.groups,
        "actor": identity.actor,
    }


@app.get("/auth/permissions")
def get_permissions(request: Request):
    """Return per-topic permissions with explanations of why access is granted."""
    identity = resolve_identity(request)
    subject = identity.subject
    login = subject.removeprefix("github:") if subject.startswith("github:") else None
    permissions: dict = {}
    for topic_id, acl in _acl_cache.items():
        if topic_id.startswith("_"):
            continue
        all_readers = acl.readers + acl.writers + acl.admins
        all_writers = acl.writers + acl.admins

        def _match_reason(allowed: list[str]) -> str | None:
            if "*" in allowed:
                return "*"
            if subject in allowed:
                return subject
            if login and f"github:user:{login}" in allowed:
                return f"github:user:{login}"
            for group in identity.groups:
                if f"github:team:{group}" in allowed:
                    return f"github:team:{group}"
                org = group.split("/")[0] if "/" in group else None
                if org and f"github:org:{org}" in allowed:
                    return f"github:org:{org}"
            return None

        topic_access: dict = {}
        reason = _match_reason(all_readers)
        if reason:
            topic_access["read"] = reason
        reason = _match_reason(all_writers)
        if reason:
            topic_access["write"] = reason
        reason = _match_reason(acl.admins)
        if reason:
            topic_access["admin"] = reason
        if topic_access:
            permissions[topic_id] = topic_access
    return {"subject": subject, "groups": identity.groups, "permissions": permissions}


@app.post("/auth/renew")
def renew_token(request: Request):
    """Renew a token that is still valid or expired within the last 24h."""
    auth_header = request.headers.get("authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(401, "No token provided")
    token = auth_header.removeprefix("Bearer ").strip()
    parts = token.split(".")
    if len(parts) != 3:
        raise HTTPException(401, "Invalid token format")
    try:
        payload = json.loads(_b64url_decode(parts[1]))
    except Exception:
        raise HTTPException(401, "Cannot decode token")
    exp = payload.get("exp", 0)
    grace_window = 7 * 24 * 3600
    if exp < time.time() - grace_window:
        raise HTTPException(401, "Token too old to renew. Login again: kwiki login")
    login = payload.get("github_login") or payload.get("sub", "").removeprefix("github:")
    if not login:
        raise HTTPException(401, "Cannot identify user from token")
    new_payload = {
        "sub": f"github:{login}",
        "github_login": login,
        "email": payload.get("email"),
        "groups": payload.get("groups", []),
        "iss": "wiki-memory-service",
        "iat": int(time.time()),
        "exp": int(time.time()) + JWT_EXPIRY_HOURS * 3600,
    }
    new_token = _sign_jwt(new_payload)
    return {"token": new_token, "github_login": login, "expires_in": JWT_EXPIRY_HOURS * 3600}


WIKI_GITHUB_ORG = os.environ.get("WIKI_GITHUB_ORG", "kaslomorg")


def _fetch_github_identity(access_token: str) -> tuple[dict, list[str]]:
    """Fetch user profile and team membership from GitHub API."""
    import httpx

    headers = {"Authorization": f"Bearer {access_token}", "Accept": "application/json"}
    client = httpx.Client(timeout=10)

    user_resp = client.get("https://api.github.com/user", headers=headers)
    user_resp.raise_for_status()
    user_info = user_resp.json()
    username = user_info["login"]

    teams: list[str] = []

    # Method 1: GET /user/teams (works when OAuth app has org access)
    try:
        url: str | None = "https://api.github.com/user/teams?per_page=100"
        while url:
            teams_resp = client.get(url, headers=headers)
            logger.info("GET /user/teams -> %d", teams_resp.status_code)
            if teams_resp.status_code != 200:
                break
            for team in teams_resp.json():
                org = team.get("organization", {}).get("login", "")
                slug = team.get("slug", "")
                if org and slug:
                    teams.append(f"{org}/{slug}")
            link = teams_resp.headers.get("link", "")
            url = None
            if 'rel="next"' in link:
                for part in link.split(","):
                    if 'rel="next"' in part:
                        url = part.split("<")[1].split(">")[0]
    except Exception as e:
        logger.warning("Failed /user/teams: %s", e)

    # Method 2: Use GraphQL to query team membership
    if not teams:
        acl_teams: set[str] = set()
        for acl in _acl_cache.values():
            for entry in acl.writers + acl.readers + acl.admins:
                if entry.startswith(f"github:team:{WIKI_GITHUB_ORG}/"):
                    team_slug = entry.removeprefix(f"github:team:{WIKI_GITHUB_ORG}/")
                    acl_teams.add(team_slug)
        logger.info("Checking via GraphQL for user=%s org=%s teams=%s", username, WIKI_GITHUB_ORG, acl_teams)
        for team_slug in acl_teams:
            try:
                query = """
                query($org: String!, $team: String!, $user: String!) {
                  organization(login: $org) {
                    team(slug: $team) {
                      members(query: $user, first: 1) {
                        nodes { login }
                      }
                    }
                  }
                }
                """
                gql_resp = client.post(
                    "https://api.github.com/graphql",
                    json={"query": query, "variables": {"org": WIKI_GITHUB_ORG, "team": team_slug, "user": username}},
                    headers=headers,
                )
                logger.debug("GraphQL %s -> %d", team_slug, gql_resp.status_code)
                if gql_resp.status_code == 200:
                    data = gql_resp.json()
                    team_data = (data.get("data") or {}).get("organization", {}).get("team")
                    if team_data:
                        members = team_data.get("members", {}).get("nodes", [])
                        if any(m.get("login", "").lower() == username.lower() for m in members):
                            teams.append(f"{WIKI_GITHUB_ORG}/{team_slug}")
                            logger.info("Team %s: user %s is MEMBER", team_slug, username)
                        else:
                            logger.debug("Team %s: user %s not a member", team_slug, username)
                    else:
                        errors = data.get("errors", [])
                        logger.warning("Team %s: no team data, errors=%s", team_slug, errors)
            except Exception as e:
                logger.warning("Failed team %s: %s", team_slug, e)

    logger.info("Resolved github identity for user=%s teams=%s", username, teams)
    return user_info, teams


def _issue_wiki_jwt(user_info: dict, teams: list[str]) -> str:
    """Issue a wiki-service JWT with GitHub identity and groups."""
    payload = {
        "sub": f"github:{user_info['login']}",
        "github_login": user_info["login"],
        "email": user_info.get("email", ""),
        "groups": teams,
        "iss": "wiki-memory-service",
        "iat": int(time.time()),
        "exp": int(time.time()) + JWT_EXPIRY_HOURS * 3600,
    }
    return _sign_jwt(payload)


# --- Health ---
@app.get("/healthz")
def health():
    return {"status": "ok", "version": __version__, "topics": len(_acl_cache), "root": str(WIKI_ROOT)}
