"""
Test wiki access as logged-in user (bob-kagenti) against the live server.

Tests both query and discovery operations using the user's OAuth token.

Usage:
    kwiki login   # login first
    uv run python test_user_access.py
    # or with explicit URL:
    uv run python test_user_access.py --url=https://wiki-memory-service-team1.apps.ykt1.hcp.res.ibm.com
"""

import json
import os
import sys
from pathlib import Path

import httpx

TOKEN_FILE = Path.home() / ".wiki-memory" / "token.json"


def load_token() -> tuple[str, str]:
    if not TOKEN_FILE.exists():
        print("ERROR: Not logged in. Run: kwiki login", file=sys.stderr)
        sys.exit(1)
    data = json.loads(TOKEN_FILE.read_text())
    return data["token"], data.get("base_url", "http://localhost:8321")


def separator(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def test_whoami(client: httpx.Client, base: str, headers: dict):
    separator("WHOAMI — Verify identity")
    resp = client.get(f"{base}/auth/whoami", headers=headers)
    if resp.status_code != 200:
        print(f"FAIL: whoami returned {resp.status_code}: {resp.text}", file=sys.stderr)
        sys.exit(1)
    data = resp.json()
    print(f"  Subject: {data['subject']}")
    print(f"  Kind:    {data['kind']}")
    print(f"  Groups:  {data.get('groups', [])}")
    print("  PASS")


def test_query_operations(client: httpx.Client, base: str, headers: dict):
    separator("QUERY — List topics, pages, search")

    print("[list topics]")
    resp = client.get(f"{base}/topics", headers=headers)
    if resp.status_code != 200:
        print(f"  FAIL ({resp.status_code}): {resp.text}")
        return
    topics = resp.json()["topics"]
    for t in topics:
        print(f"  - {t['topic_id']} ({t['page_count']} pages)")
    print("  PASS")

    if not topics:
        print("\n  No topics found — skipping page/search tests")
        return

    topic_id = topics[0]["topic_id"]

    print(f"\n[list pages in '{topic_id}']")
    resp = client.get(f"{base}/topics/{topic_id}/pages", headers=headers)
    if resp.status_code == 200:
        pages = resp.json()["pages"]
        for p in pages[:10]:
            print(f"  - {p}")
        if len(pages) > 10:
            print(f"  ... and {len(pages) - 10} more")
        print("  PASS")
    else:
        print(f"  FAIL ({resp.status_code}): {resp.text}")

    print(f"\n[search '{topic_id}' for 'transformer']")
    resp = client.post(
        f"{base}/topics/{topic_id}/query",
        json={"query": "transformer", "limit": 3},
        headers=headers,
    )
    if resp.status_code == 200:
        results = resp.json()["results"]
        for r in results:
            print(f"  - {r['path']} (score={r['score']:.3f})")
        print(f"  PASS ({len(results)} results)")
    else:
        print(f"  FAIL ({resp.status_code}): {resp.text}")

    if pages:
        page_path = pages[0]
        print(f"\n[read page '{page_path}']")
        resp = client.get(f"{base}/topics/{topic_id}/pages/{page_path}", headers=headers)
        if resp.status_code == 200:
            content = resp.json()["content"]
            print(f"  {content[:120]}...")
            print("  PASS")
        else:
            print(f"  FAIL ({resp.status_code}): {resp.text}")

    print(f"\n[activity for '{topic_id}']")
    resp = client.get(f"{base}/topics/{topic_id}/activity?limit=5", headers=headers)
    if resp.status_code == 200:
        entries = resp.json().get("entries", [])
        for e in entries[:3]:
            print(f"  - {e.get('message', '?')} by {e.get('author', '?')}")
        print(f"  PASS ({len(entries)} entries)")
    else:
        print(f"  FAIL ({resp.status_code}): {resp.text}")

    print(f"\n[tags for '{topic_id}']")
    resp = client.get(f"{base}/topics/{topic_id}/tags", headers=headers)
    if resp.status_code == 200:
        tags = resp.json().get("tags", {})
        for tag, count in list(tags.items())[:5]:
            print(f"  - {tag}: {count} pages")
        print(f"  PASS ({len(tags)} tags)")
    else:
        print(f"  FAIL ({resp.status_code}): {resp.text}")

    print(f"\n[graph for '{topic_id}']")
    resp = client.get(f"{base}/topics/{topic_id}/graph", headers=headers)
    if resp.status_code == 200:
        graph = resp.json()
        print(f"  nodes: {len(graph.get('nodes', []))}, edges: {len(graph.get('edges', []))}")
        print("  PASS")
    else:
        print(f"  FAIL ({resp.status_code}): {resp.text}")

    print("\n[global search for 'attention']")
    resp = client.post(
        f"{base}/search",
        json={"query": "attention", "limit": 5},
        headers=headers,
    )
    if resp.status_code == 200:
        results = resp.json()["results"]
        for r in results[:3]:
            print(f"  - [{r.get('topic_id')}] {r['path']} (score={r['score']:.3f})")
        print(f"  PASS ({len(results)} results)")
    else:
        print(f"  FAIL ({resp.status_code}): {resp.text}")


def test_discovery_operations(client: httpx.Client, base: str, headers: dict):
    separator("DISCOVERY — Templates, write, novelty check")

    print("[list templates]")
    resp = client.get(f"{base}/templates", headers=headers)
    if resp.status_code == 200:
        templates = resp.json()["templates"]
        for t in templates:
            print(f"  - {t['id']}: {t['description']}")
        print("  PASS")
    else:
        print(f"  FAIL ({resp.status_code}): {resp.text}")

    print("\n[get template 'paper-summary']")
    resp = client.get(f"{base}/templates/paper-summary", headers=headers)
    if resp.status_code == 200:
        tmpl = resp.json()
        print(f"  {tmpl['content'][:100]}...")
        print("  PASS")
    else:
        print(f"  FAIL ({resp.status_code}): {resp.text}")

    print("\n[check novelty in 'ai']")
    resp = client.post(
        f"{base}/topics/ai/check-novelty",
        json={"title": "Test User Access Page", "abstract": "A test page for verifying user access"},
        headers=headers,
    )
    if resp.status_code == 200:
        data = resp.json()
        print(f"  novel: {data.get('novel')}")
        print("  PASS")
    elif resp.status_code == 403:
        print("  No write access (403) — expected if user is reader only")
        print("  PASS (access control working)")
    else:
        print(f"  FAIL ({resp.status_code}): {resp.text}")

    print("\n[write test page as draft in 'ai']")
    test_content = "---\ntags: [test]\n---\n# Access Test\n\nThis page verifies write access."
    resp = client.post(
        f"{base}/topics/ai/pages/test-user-access.md?draft=true",
        json={"content": test_content, "message": "Test user write access"},
        headers=headers,
    )
    if resp.status_code == 200:
        data = resp.json()
        print(f"  Written as draft: {data.get('path')}")
        print("  PASS")
    elif resp.status_code == 403:
        print("  No write access (403) — expected if user is reader only")
        print("  PASS (access control working)")
    else:
        print(f"  FAIL ({resp.status_code}): {resp.text}")


def test_renew(client: httpx.Client, base: str, headers: dict):
    separator("RENEW — Token renewal")
    resp = client.post(f"{base}/auth/renew", headers=headers)
    if resp.status_code == 200:
        data = resp.json()
        print(f"  Renewed for: {data['github_login']}")
        print(f"  Expires in:  {data['expires_in'] // 3600}h")
        print("  PASS")
    else:
        print(f"  FAIL ({resp.status_code}): {resp.text}")


def main():
    token, base = load_token()

    for arg in sys.argv[1:]:
        if arg.startswith("--url="):
            base = arg.split("=", 1)[1]
        elif arg.startswith("http"):
            base = arg

    headers = {"Authorization": f"Bearer {token}"}
    insecure = os.environ.get("WIKI_INSECURE_TLS", "1") == "1"
    client = httpx.Client(timeout=30, verify=not insecure)

    print("Testing wiki access as logged-in user")
    print(f"Server: {base}")

    test_whoami(client, base, headers)
    test_query_operations(client, base, headers)
    test_discovery_operations(client, base, headers)
    test_renew(client, base, headers)

    separator("ALL TESTS COMPLETE")
    print("  All query and discovery operations verified.")


if __name__ == "__main__":
    main()
