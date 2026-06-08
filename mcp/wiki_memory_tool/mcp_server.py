"""
MCP Server for Wiki Memory Service.

Exposes wiki operations as MCP tools for Claude Code and other MCP clients.

Two modes:
  - Local (default): imports wiki_service.py directly, no ACL enforcement
  - Remote (WIKI_SERVICE_URL set): calls wiki service HTTP API with cached auth token

Supports two transports:
  - stdio: for Claude Code subprocess integration (default)
  - streamable-http: for remote MCP clients (set MCP_TRANSPORT=streamable-http)
"""

import json
import logging
import os
from pathlib import Path

import httpx

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

MCP_TRANSPORT = os.environ.get("MCP_TRANSPORT", "stdio")
MCP_PORT = int(os.environ.get("MCP_PORT", "8322"))
MCP_HOST = os.environ.get("MCP_HOST", "0.0.0.0")
WIKI_SERVICE_URL = os.environ.get("WIKI_SERVICE_URL", "")

mcp = FastMCP("wiki-memory", host=MCP_HOST, port=MCP_PORT)

TOKEN_FILE = Path.home() / ".wiki-memory" / "token.json"


def _load_token() -> str | None:
    if TOKEN_FILE.exists():
        data = json.loads(TOKEN_FILE.read_text())
        return data.get("token")
    return None


def _remote_client() -> httpx.Client:
    headers = {}
    token = _load_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    insecure = os.environ.get("WIKI_INSECURE_TLS") == "1"
    if insecure:
        logger.warning("WIKI_INSECURE_TLS=1 — TLS verification disabled (dev only)")
    return httpx.Client(base_url=WIKI_SERVICE_URL, headers=headers, timeout=30, verify=not insecure)


def _get_service():
    """Lazy import wiki_service to ensure env vars are set before import."""
    import wiki_service as ws

    return ws


@mcp.tool(
    name="wiki_list_topics",
    description="List all available wiki topics and their page counts.",
)
async def wiki_list_topics() -> str:
    if WIKI_SERVICE_URL:
        client = _remote_client()
        resp = client.get("/topics")
        if resp.status_code != 200:
            return f"Error: {resp.status_code} {resp.text}"
        topics = resp.json().get("topics", [])
        if not topics:
            return "No topics found."
        return "Topics:\n" + "\n".join(f"- {t['topic_id']} ({t['page_count']} pages)" for t in topics)

    ws = _get_service()
    topics = []
    for topic_id in ws._acl_cache:
        if topic_id.startswith("_"):
            continue
        page_count = len(list(ws._topic_dir(topic_id).rglob("*.md")))
        topics.append(f"- {topic_id} ({page_count} pages)")
    if not topics:
        return "No topics found."
    return "Topics:\n" + "\n".join(topics)


@mcp.tool(
    name="wiki_query",
    description="Search a wiki topic for pages matching a query. Returns ranked results with snippets.",
)
async def wiki_query(topic_id: str, query: str, limit: int = 10) -> str:
    if WIKI_SERVICE_URL:
        client = _remote_client()
        resp = client.post(f"/topics/{topic_id}/query", json={"query": query, "limit": limit})
        if resp.status_code != 200:
            return f"Error: {resp.status_code} {resp.text}"
        results = resp.json().get("results", [])
        if not results:
            return f"No results for '{query}' in topic '{topic_id}'."
        lines = [f"Search results for '{query}' in '{topic_id}':"]
        for r in results:
            lines.append(f"- {r['path']} (score={r['score']})")
            if r.get("snippet"):
                lines.append(f"  {r['snippet'][:150]}")
        return "\n".join(lines)

    ws = _get_service()
    results = ws.search_topic(topic_id, query, limit)
    if not results:
        return f"No results for '{query}' in topic '{topic_id}'."
    lines = [f"Search results for '{query}' in '{topic_id}':"]
    for r in results:
        lines.append(f"- {r['path']} (score={r['score']})")
        if r["snippet"]:
            lines.append(f"  {r['snippet'][:150]}")
    return "\n".join(lines)


@mcp.tool(
    name="wiki_read",
    description="Read the full content of a wiki page. Provide topic_id and the page path (e.g. 'transformers.md').",
)
async def wiki_read(topic_id: str, path: str) -> str:
    if WIKI_SERVICE_URL:
        client = _remote_client()
        resp = client.get(f"/topics/{topic_id}/pages/{path}")
        if resp.status_code != 200:
            return f"Error: {resp.status_code} {resp.text}"
        return resp.json().get("content", "")

    ws = _get_service()
    full = ws._topic_dir(topic_id) / path
    if not full.exists():
        return f"Page not found: {topic_id}/{path}"
    return full.read_text()


@mcp.tool(
    name="wiki_write",
    description="Write or update a wiki page. Commits to git (and pushes to remote if configured). Set draft=True to submit for review.",
)
async def wiki_write(topic_id: str, path: str, content: str, message: str = "", draft: bool = False) -> str:
    if WIKI_SERVICE_URL:
        client = _remote_client()
        body = {"content": content, "message": message or f"mcp-write: {topic_id}/{path}"}
        url = f"/topics/{topic_id}/pages/{path}"
        if draft:
            url += "?draft=true"
        resp = client.post(url, json=body)
        if resp.status_code != 200:
            return f"Error: {resp.status_code} {resp.text}"
        data = resp.json()
        result = f"{'Draft' if draft else 'Written'}: {topic_id}/{path}"
        if data.get("suggested_links"):
            result += "\nSuggested links:\n" + "\n".join(
                f"- {s['path']} (score={s['score']})" for s in data["suggested_links"]
            )
        return result

    ws = _get_service()
    topic_dir = ws._topic_dir(topic_id)
    if draft:
        full = topic_dir / "_drafts" / path
    else:
        full = topic_dir / path
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(content)

    rel = str(full.relative_to(ws.WIKI_ROOT))
    msg = message or f"mcp-write: {topic_id}/{path}"
    ws._commit(rel, msg, "mcp-client")
    return f"{'Draft' if draft else 'Written'}: {topic_id}/{path}"


@mcp.tool(
    name="wiki_check_novelty",
    description="Check if content is novel relative to existing wiki pages in a topic. Returns whether similar content already exists.",
)
async def wiki_check_novelty(topic_id: str, title: str, abstract: str) -> str:
    if WIKI_SERVICE_URL:
        client = _remote_client()
        resp = client.post(f"/topics/{topic_id}/check-novelty", json={"title": title, "abstract": abstract})
        if resp.status_code != 200:
            return f"Error: {resp.status_code} {resp.text}"
        return json.dumps(resp.json())

    ws = _get_service()
    combined = f"{title} {abstract}"
    results = ws.search_topic(topic_id, combined, limit=3)

    if results and results[0]["score"] > 0.15:
        similar = [r["path"] for r in results[:3]]
        return json.dumps({"novel": False, "reason": "Similar content exists", "similar": similar})
    return json.dumps({"novel": True, "reason": "No sufficiently similar content found"})


@mcp.tool(
    name="wiki_activity",
    description="Get recent changes (git log) for a topic or globally. Returns commit history.",
)
async def wiki_activity(topic_id: str = "", limit: int = 20) -> str:
    if WIKI_SERVICE_URL:
        client = _remote_client()
        url = f"/topics/{topic_id}/activity?limit={limit}" if topic_id else f"/activity?limit={limit}"
        resp = client.get(url)
        if resp.status_code != 200:
            return f"Error: {resp.status_code} {resp.text}"
        entries = resp.json().get("entries", [])
        if not entries:
            return "No recent activity."
        lines = ["Recent activity:"]
        for e in entries:
            lines.append(f"- {e['timestamp']} {e['author']}: {e['message']}")
        return "\n".join(lines)

    ws = _get_service()
    entries = ws.get_activity(topic_id=topic_id or None, limit=limit)
    if not entries:
        return "No recent activity."
    lines = ["Recent activity:"]
    for e in entries:
        lines.append(f"- {e['timestamp']} {e['author']}: {e['message']}")
    return "\n".join(lines)


@mcp.tool(
    name="wiki_backlinks",
    description="Find pages that link to a given page (backlinks/inbound references).",
)
async def wiki_backlinks(topic_id: str, path: str) -> str:
    if WIKI_SERVICE_URL:
        client = _remote_client()
        resp = client.get(f"/topics/{topic_id}/backlinks/{path}")
        if resp.status_code != 200:
            return f"Error: {resp.status_code} {resp.text}"
        backlinks = resp.json().get("backlinks", [])
        if not backlinks:
            return f"No pages link to {topic_id}/{path}."
        return f"Pages linking to {topic_id}/{path}:\n" + "\n".join(f"- {b}" for b in backlinks)

    ws = _get_service()
    backlinks = ws.find_backlinks(topic_id, path)
    if not backlinks:
        return f"No pages link to {topic_id}/{path}."
    return f"Pages linking to {topic_id}/{path}:\n" + "\n".join(f"- {b}" for b in backlinks)


@mcp.tool(
    name="wiki_search_all",
    description="Search across all wiki topics for pages matching a query. Returns results from all accessible topics.",
)
async def wiki_search_all(query: str, limit: int = 10) -> str:
    if WIKI_SERVICE_URL:
        client = _remote_client()
        resp = client.post("/search", json={"query": query, "limit": limit})
        if resp.status_code != 200:
            return f"Error: {resp.status_code} {resp.text}"
        results = resp.json().get("results", [])
        if not results:
            return f"No results for '{query}' across all topics."
        lines = [f"Global search results for '{query}':"]
        for r in results:
            lines.append(f"- [{r.get('topic_id', '?')}] {r['path']} (score={r['score']})")
            if r.get("snippet"):
                lines.append(f"  {r['snippet'][:150]}")
        return "\n".join(lines)

    ws = _get_service()
    all_results = []
    for topic_id in ws._acl_cache:
        if topic_id.startswith("_"):
            continue
        results = ws.search_topic(topic_id, query, limit)
        for r in results:
            r["topic_id"] = topic_id
        all_results.extend(results)
    all_results.sort(key=lambda r: r["score"], reverse=True)
    all_results = all_results[:limit]
    if not all_results:
        return f"No results for '{query}' across all topics."
    lines = [f"Global search results for '{query}':"]
    for r in all_results:
        lines.append(f"- [{r['topic_id']}] {r['path']} (score={r['score']})")
        if r.get("snippet"):
            lines.append(f"  {r['snippet'][:150]}")
    return "\n".join(lines)


@mcp.tool(
    name="wiki_get_template",
    description="Get a page template for structured content creation. Available: paper-summary, concept-overview, how-to-guide, comparison. Omit template_id to list all.",
)
async def wiki_get_template(template_id: str = "") -> str:
    if WIKI_SERVICE_URL:
        client = _remote_client()
        url = f"/templates/{template_id}" if template_id else "/templates"
        resp = client.get(url)
        if resp.status_code != 200:
            return f"Error: {resp.status_code} {resp.text}"
        data = resp.json()
        if "templates" in data:
            lines = ["Available templates:"]
            for t in data["templates"]:
                lines.append(f"- {t['id']}: {t['name']} — {t['description']}")
            return "\n".join(lines)
        return f"Template: {data['name']}\n\n{data['content']}"

    from wiki_service import _TEMPLATES

    if not template_id:
        lines = ["Available templates:"]
        for tid, t in _TEMPLATES.items():
            lines.append(f"- {tid}: {t['name']} — {t['description']}")
        return "\n".join(lines)
    t = _TEMPLATES.get(template_id)
    if not t:
        return f"Template '{template_id}' not found. Available: {list(_TEMPLATES.keys())}"
    return f"Template: {t['name']}\n\n{t['content']}"


@mcp.tool(
    name="wiki_list_drafts",
    description="List pending drafts in a topic that need review/approval.",
)
async def wiki_list_drafts(topic_id: str) -> str:
    if WIKI_SERVICE_URL:
        client = _remote_client()
        resp = client.get(f"/topics/{topic_id}/drafts")
        if resp.status_code != 200:
            return f"Error: {resp.status_code} {resp.text}"
        drafts = resp.json().get("drafts", [])
        if not drafts:
            return f"No pending drafts in '{topic_id}'."
        return f"Drafts in '{topic_id}':\n" + "\n".join(f"- {d}" for d in drafts)

    ws = _get_service()
    drafts_dir = ws._topic_dir(topic_id) / "_drafts"
    if not drafts_dir.exists():
        return f"No pending drafts in '{topic_id}'."
    pages = [str(f.relative_to(drafts_dir)) for f in drafts_dir.rglob("*.md")]
    if not pages:
        return f"No pending drafts in '{topic_id}'."
    return f"Drafts in '{topic_id}':\n" + "\n".join(f"- {p}" for p in sorted(pages))


@mcp.tool(
    name="wiki_approve_draft",
    description="Approve a draft and publish it as a live wiki page. Requires admin access.",
)
async def wiki_approve_draft(topic_id: str, path: str) -> str:
    if WIKI_SERVICE_URL:
        client = _remote_client()
        resp = client.post(f"/topics/{topic_id}/drafts/{path}/approve")
        if resp.status_code != 200:
            return f"Error: {resp.status_code} {resp.text}"
        return f"Approved: {topic_id}/{path}"

    ws = _get_service()
    topic_dir = ws._topic_dir(topic_id)
    draft_file = topic_dir / "_drafts" / path
    if not draft_file.exists():
        return f"Draft not found: {topic_id}/_drafts/{path}"
    live_file = topic_dir / path
    live_file.parent.mkdir(parents=True, exist_ok=True)
    live_file.write_text(draft_file.read_text())
    draft_file.unlink()
    rel_live = str(live_file.relative_to(ws.WIKI_ROOT))
    ws._commit(rel_live, f"approve: {topic_id}/{path}", "mcp-admin")
    return f"Approved: {topic_id}/{path}"


@mcp.tool(
    name="wiki_list_tags",
    description="List all tags in a topic with page counts.",
)
async def wiki_list_tags(topic_id: str) -> str:
    if WIKI_SERVICE_URL:
        client = _remote_client()
        resp = client.get(f"/topics/{topic_id}/tags")
        if resp.status_code != 200:
            return f"Error: {resp.status_code} {resp.text}"
        tags = resp.json().get("tags", {})
        if not tags:
            return f"No tags in '{topic_id}'."
        lines = [f"Tags in '{topic_id}':"]
        for tag, count in sorted(tags.items()):
            lines.append(f"- {tag} ({count} pages)")
        return "\n".join(lines)

    ws = _get_service()
    topic_dir = ws._topic_dir(topic_id)
    tag_counts: dict[str, int] = {}
    for f in topic_dir.rglob("*.md"):
        if "_drafts" in f.parts:
            continue
        content = f.read_text(errors="replace")
        meta, _ = ws.parse_frontmatter(content)
        for tag in meta.get("tags", []):
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
    if not tag_counts:
        return f"No tags in '{topic_id}'."
    lines = [f"Tags in '{topic_id}':"]
    for tag, count in sorted(tag_counts.items()):
        lines.append(f"- {tag} ({count} pages)")
    return "\n".join(lines)


@mcp.tool(
    name="wiki_pages_by_tag",
    description="List all pages in a topic that have a specific tag.",
)
async def wiki_pages_by_tag(topic_id: str, tag: str) -> str:
    if WIKI_SERVICE_URL:
        client = _remote_client()
        resp = client.get(f"/topics/{topic_id}/tags/{tag}")
        if resp.status_code != 200:
            return f"Error: {resp.status_code} {resp.text}"
        pages = resp.json().get("pages", [])
        if not pages:
            return f"No pages with tag '{tag}' in '{topic_id}'."
        return f"Pages tagged '{tag}' in '{topic_id}':\n" + "\n".join(f"- {p}" for p in pages)

    ws = _get_service()
    topic_dir = ws._topic_dir(topic_id)
    pages = []
    for f in topic_dir.rglob("*.md"):
        if "_drafts" in f.parts:
            continue
        content = f.read_text(errors="replace")
        meta, _ = ws.parse_frontmatter(content)
        if tag in meta.get("tags", []):
            pages.append(str(f.relative_to(topic_dir)))
    if not pages:
        return f"No pages with tag '{tag}' in '{topic_id}'."
    return f"Pages tagged '{tag}' in '{topic_id}':\n" + "\n".join(f"- {p}" for p in sorted(pages))


@mcp.tool(
    name="wiki_graph",
    description="Get the page link graph for a topic. Returns nodes (pages) and edges (links between them).",
)
async def wiki_graph(topic_id: str) -> str:
    if WIKI_SERVICE_URL:
        client = _remote_client()
        resp = client.get(f"/topics/{topic_id}/graph")
        if resp.status_code != 200:
            return f"Error: {resp.status_code} {resp.text}"
        return json.dumps(resp.json())

    ws = _get_service()
    import re as _re

    topic_dir = ws._topic_dir(topic_id)
    nodes = []
    edges = []
    for f in topic_dir.rglob("*.md"):
        if "_drafts" in f.parts:
            continue
        rel = str(f.relative_to(topic_dir))
        content = f.read_text(errors="replace")
        meta, body = ws.parse_frontmatter(content)
        title_match = _re.search(r"^#\s+(.+)", body)
        title = title_match.group(1) if title_match else rel
        nodes.append({"id": rel, "title": title, "tags": meta.get("tags", [])})
        for link in ws.extract_links(content):
            edges.append({"source": rel, "target": link})
    return json.dumps({"topic": topic_id, "nodes": nodes, "edges": edges})


def run_mcp_server():
    """Start the MCP server with configured transport."""
    mcp.run(transport=MCP_TRANSPORT)


if __name__ == "__main__":
    run_mcp_server()
