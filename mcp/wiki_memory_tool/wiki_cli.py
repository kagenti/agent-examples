"""
Wiki Memory Service CLI — testing tool for Discovery and Query agent operations.

Authenticates via SPIFFE headers (simulated), GitHub OAuth token, or user token.

Usage:
    # GitHub login (device flow):
    uv run python wiki_cli.py login
    uv run python wiki_cli.py whoami
    uv run python wiki_cli.py logout

    # As Discovery Agent (writer):
    uv run python wiki_cli.py discover write ai transformers.md --file content.md
    uv run python wiki_cli.py discover write ai transformers.md --content "# Title\nBody"
    uv run python wiki_cli.py discover novelty ai "Transformers" "Attention mechanisms paper"

    # As Query Agent (reader, on behalf of user):
    uv run python wiki_cli.py query list-topics
    uv run python wiki_cli.py query list-pages ai
    uv run python wiki_cli.py query search ai "attention mechanism"
    uv run python wiki_cli.py query read ai transformers.md

    # Override defaults:
    uv run python wiki_cli.py --base-url http://localhost:8321 --agent discovery --topic ai discover write ...
"""

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path

import httpx

DEFAULT_BASE_URL = "http://localhost:8321"
DEFAULT_TRUST_DOMAIN = "kagenti.example.com"
TOKEN_DIR = Path.home() / ".wiki-memory"
TOKEN_FILE = TOKEN_DIR / "token.json"

DEVICE_POLL_INTERVAL = 5


def load_cached_token() -> dict | None:
    if TOKEN_FILE.exists():
        data = json.loads(TOKEN_FILE.read_text())
        if data.get("token"):
            return data
    return None


def save_token(token: str, base_url: str):
    TOKEN_DIR.mkdir(parents=True, exist_ok=True)
    TOKEN_FILE.write_text(json.dumps({"token": token, "base_url": base_url}))
    TOKEN_FILE.chmod(0o600)


def delete_token():
    if TOKEN_FILE.exists():
        TOKEN_FILE.unlink()


def make_auth_headers() -> dict | None:
    cached = load_cached_token()
    if cached:
        return {"Authorization": f"Bearer {cached['token']}"}
    return None


def cmd_login(client: httpx.Client, base: str, args):
    resp = client.post(f"{base}/auth/github/device")
    if resp.status_code != 200:
        print(f"ERROR: Failed to start device flow: {resp.text}", file=sys.stderr)
        sys.exit(1)

    data = resp.json()
    user_code = data["user_code"]
    verification_uri = data["verification_uri"]
    expires_in = data.get("expires_in", 900)

    print("\n" + "=" * 50)
    print("  GitHub Device Authorization")
    print("=" * 50)
    print("\n  1. Open this URL in your browser:\n")
    print(f"     {verification_uri}\n")
    print("  2. Enter this code:\n")
    print(f"     {user_code}")
    print(f"\n  Code expires in {expires_in // 60} minutes.")
    print("=" * 50)
    print("\nWaiting for authorization...", end="", flush=True)

    device_code = data["device_code"]
    interval = data.get("interval", DEVICE_POLL_INTERVAL)

    while True:
        time.sleep(interval)
        print(".", end="", flush=True)
        try:
            poll_resp = client.post(
                f"{base}/auth/github/device/token",
                json={"device_code": device_code},
            )
        except httpx.RequestError as e:
            print(f"\n\nERROR: Connection failed: {e}", file=sys.stderr)
            sys.exit(1)

        if poll_resp.status_code == 200:
            token_data = poll_resp.json()
            save_token(token_data["token"], base)
            login = token_data.get("github_login", token_data.get("login", "unknown"))
            groups = token_data.get("groups", [])
            print(f"\n\nLogged in as {login}")
            if groups:
                print(f"Groups: {', '.join(groups)}")
            return
        elif poll_resp.status_code == 202:
            poll_data = poll_resp.json()
            error = poll_data.get("error", "")
            if error == "slow_down":
                interval += 5
            elif error == "expired_token":
                print("\n\nDevice code expired. Please run login again.", file=sys.stderr)
                sys.exit(1)
            elif error == "access_denied":
                print("\n\nAuthorization denied by user.", file=sys.stderr)
                sys.exit(1)
            continue
        else:
            try:
                detail = poll_resp.json().get("detail", poll_resp.text)
            except Exception:
                detail = poll_resp.text
            print(f"\n\nERROR: {detail}", file=sys.stderr)
            sys.exit(1)


def cmd_logout(args):
    if TOKEN_FILE.exists():
        delete_token()
        print("Logged out (token removed).")
    else:
        print("Not logged in.")


def _decode_jwt_payload(token: str) -> dict | None:
    """Decode JWT payload without signature verification."""
    parts = token.split(".")
    if len(parts) != 3:
        return None
    try:
        return json.loads(base64.urlsafe_b64decode(parts[1] + "=="))
    except Exception:
        return None


def _resolve_display_name(subject: str) -> str | None:
    """Extract a human-readable name from the subject. Returns None if unresolvable."""
    if subject.startswith("github:"):
        return subject.removeprefix("github:")
    if subject.startswith("spiffe://"):
        return subject
    payload = _decode_jwt_payload(subject)
    if payload:
        login = payload.get("github_login") or payload.get("sub", "").removeprefix("github:")
        if login:
            return login
    return None


def cmd_whoami(client: httpx.Client, base: str, args):
    headers = make_auth_headers()
    if not headers:
        print("Not logged in. Run: kwiki login", file=sys.stderr)
        sys.exit(1)

    cached = load_cached_token()
    token = cached["token"] if cached else ""
    payload = _decode_jwt_payload(token)

    resp = client.get(f"{base}/auth/whoami", headers=headers)
    if resp.status_code == 200:
        data = resp.json()
        subject = data.get("subject", "")
        display = _resolve_display_name(subject)
        if not display and payload:
            display = payload.get("github_login") or payload.get("sub", "").removeprefix("github:")
        if not display:
            print("ERROR: Could not resolve identity from token.", file=sys.stderr)
            print("Try logging in again: kwiki login", file=sys.stderr)
            sys.exit(1)
        exp = payload.get("exp") if payload else None
        if exp and int(exp - time.time()) <= 0:
            print("ERROR: Token expired.", file=sys.stderr)
            print("Renew token: kwiki renew", file=sys.stderr)
            sys.exit(1)
        if exp:
            remaining = int(exp - time.time())
            if remaining < 3600:
                status = f"expires in {remaining // 60}m"
            else:
                status = f"expires in {remaining // 3600}h {(remaining % 3600) // 60}m"
        else:
            status = "active"
        print(f"  User:    {display}")
        print(f"  Status:  {status}")
        groups = data.get("groups", [])
        if groups:
            print(f"  Groups:  {', '.join(groups)}")
        else:
            print("  Groups:  (none)")
        print(f"  Server:  {base}")

        # Verify org membership
        org_groups = [g for g in groups if g.startswith("kaslomorg/")]
        if not org_groups:
            print("")
            print("  WARNING: No kaslomorg teams in token.", file=sys.stderr)
            print("  If you are a member of kaslomorg teams, re-login to refresh:", file=sys.stderr)
            print("    kwiki login", file=sys.stderr)

        # Fetch permissions
        perm_resp = client.get(f"{base}/auth/permissions", headers=headers)
        if perm_resp.status_code == 200:
            perms = perm_resp.json().get("permissions", {})
            if perms:
                print("  Access:")
                for topic, access_map in sorted(perms.items()):
                    roles = list(access_map.keys())
                    print(f"    {topic}: {', '.join(roles)}")
                    for role, reason in access_map.items():
                        print(f"      {role} <- {reason}")
            else:
                print("  Access:  none")
    elif resp.status_code == 401:
        print("ERROR: Token expired or invalid.", file=sys.stderr)
        print("Login again: kwiki login", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"ERROR ({resp.status_code}): {resp.text}", file=sys.stderr)
        sys.exit(1)


def cmd_renew(client: httpx.Client, base: str):
    headers = make_auth_headers()
    if not headers:
        print("Not logged in. Run: kwiki login", file=sys.stderr)
        sys.exit(1)
    resp = client.post(f"{base}/auth/renew", headers=headers)
    if resp.status_code == 200:
        data = resp.json()
        save_token(data["token"], base)
        print(f"  Token renewed for {data['github_login']}")
        hours = data.get("expires_in", 0) // 3600
        print(f"  Expires in {hours}h")
    elif resp.status_code == 404:
        print("  Server does not support /auth/renew, starting login flow...")
        cmd_login(client, base, None)
    elif resp.status_code == 401:
        print("ERROR: Token too old to renew.", file=sys.stderr)
        print("Login again: kwiki login", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"ERROR ({resp.status_code}): {resp.text}", file=sys.stderr)
        sys.exit(1)


def make_discovery_headers(topic: str, trust_domain: str) -> dict:
    return {
        "X-Spiffe-Id": f"spiffe://{trust_domain}/ns/topic-{topic}/sa/discovery-agent",
    }


def make_query_headers(user: str, trust_domain: str) -> dict:
    return {
        "X-Spiffe-Id": f"spiffe://{trust_domain}/ns/wiki-system/sa/query-agent",
        "X-Original-Subject": user,
    }


def cmd_discover_write(client: httpx.Client, base: str, headers: dict, args):
    topic = args.topic
    path = args.path
    if args.file:
        content = open(args.file).read()
    elif args.content:
        content = args.content
    else:
        content = sys.stdin.read()

    url = f"{base}/topics/{topic}/pages/{path}"
    if getattr(args, "draft", False):
        url += "?draft=true"
    resp = client.post(
        url,
        json={"content": content, "message": args.message or f"cli: write {topic}/{path}"},
        headers=headers,
    )
    if resp.status_code == 200:
        data = resp.json()
        status = data.get("status", "written")
        print(f"{status.capitalize()}: {data['path']} by {data['author']}")
        if data.get("suggested_links"):
            print("Suggested links:")
            for s in data["suggested_links"]:
                print(f"  {s['path']} (score={s['score']})")
    else:
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text
        print(f"ERROR ({resp.status_code}): {detail}", file=sys.stderr)
        sys.exit(1)


def cmd_discover_novelty(client: httpx.Client, base: str, headers: dict, args):
    topic = args.topic
    resp = client.post(
        f"{base}/topics/{topic}/check-novelty",
        json={"title": args.title, "abstract": args.abstract},
        headers=headers,
    )
    if resp.status_code == 200:
        data = resp.json()
        if data["novel"]:
            print(f"NOVEL: {data['reason']}")
        else:
            print(f"NOT NOVEL: {data['reason']}")
            for s in data.get("similar", []):
                print(f"  similar: {s['path']} (score={s['score']})")
    else:
        print(f"ERROR ({resp.status_code}): {resp.json().get('detail', resp.text)}", file=sys.stderr)
        sys.exit(1)


def cmd_query_list_topics(client: httpx.Client, base: str, headers: dict, args):
    resp = client.get(f"{base}/topics", headers=headers)
    if resp.status_code == 200:
        for t in resp.json()["topics"]:
            print(f"  {t['topic_id']} ({t['page_count']} pages)")
    else:
        print(f"ERROR ({resp.status_code}): {resp.text}", file=sys.stderr)
        sys.exit(1)


def cmd_query_list_pages(client: httpx.Client, base: str, headers: dict, args):
    resp = client.get(f"{base}/topics/{args.topic}/pages", headers=headers)
    if resp.status_code == 200:
        for p in resp.json()["pages"]:
            print(f"  {p}")
    else:
        print(f"ERROR ({resp.status_code}): {resp.json().get('detail', resp.text)}", file=sys.stderr)
        sys.exit(1)


def cmd_query_search(client: httpx.Client, base: str, headers: dict, args):
    resp = client.post(
        f"{base}/topics/{args.topic}/query",
        json={"query": args.query, "limit": args.limit},
        headers=headers,
    )
    if resp.status_code == 200:
        results = resp.json()["results"]
        if not results:
            print("No results.")
            return
        for r in results:
            print(f"  {r['path']} (score={r['score']})")
            if r.get("snippet"):
                print(f"    {r['snippet'][:120]}")
    else:
        print(f"ERROR ({resp.status_code}): {resp.json().get('detail', resp.text)}", file=sys.stderr)
        sys.exit(1)


def cmd_query_read(client: httpx.Client, base: str, headers: dict, args):
    resp = client.get(f"{base}/topics/{args.topic}/pages/{args.path}", headers=headers)
    if resp.status_code == 200:
        data = resp.json()
        print(data["content"])
        if data.get("frontmatter"):
            print(f"\n--- Frontmatter: {json.dumps(data['frontmatter'])}")
    else:
        print(f"ERROR ({resp.status_code}): {resp.json().get('detail', resp.text)}", file=sys.stderr)
        sys.exit(1)


def cmd_query_activity(client: httpx.Client, base: str, headers: dict, args):
    topic = getattr(args, "topic", None)
    if topic:
        url = f"{base}/topics/{topic}/activity?limit={args.limit}"
    else:
        url = f"{base}/activity?limit={args.limit}"
    resp = client.get(url, headers=headers)
    if resp.status_code == 200:
        for e in resp.json()["entries"]:
            print(f"  {e['timestamp']} {e['author']}: {e['message']}")
    else:
        print(f"ERROR ({resp.status_code}): {resp.text}", file=sys.stderr)
        sys.exit(1)


def cmd_query_backlinks(client: httpx.Client, base: str, headers: dict, args):
    resp = client.get(f"{base}/topics/{args.topic}/backlinks/{args.path}", headers=headers)
    if resp.status_code == 200:
        backlinks = resp.json()["backlinks"]
        if not backlinks:
            print("No backlinks found.")
            return
        for b in backlinks:
            print(f"  {b}")
    else:
        print(f"ERROR ({resp.status_code}): {resp.text}", file=sys.stderr)
        sys.exit(1)


def cmd_query_search_all(client: httpx.Client, base: str, headers: dict, args):
    resp = client.post(
        f"{base}/search",
        json={"query": args.query, "limit": args.limit},
        headers=headers,
    )
    if resp.status_code == 200:
        results = resp.json()["results"]
        if not results:
            print("No results.")
            return
        for r in results:
            print(f"  [{r.get('topic_id', '?')}] {r['path']} (score={r['score']})")
            if r.get("snippet"):
                print(f"    {r['snippet'][:120]}")
    else:
        print(f"ERROR ({resp.status_code}): {resp.text}", file=sys.stderr)
        sys.exit(1)


def cmd_query_tags(client: httpx.Client, base: str, headers: dict, args):
    resp = client.get(f"{base}/topics/{args.topic}/tags", headers=headers)
    if resp.status_code == 200:
        tags = resp.json()["tags"]
        if not tags:
            print("No tags found.")
            return
        for tag, count in sorted(tags.items()):
            print(f"  {tag} ({count} pages)")
    else:
        print(f"ERROR ({resp.status_code}): {resp.text}", file=sys.stderr)
        sys.exit(1)


def cmd_query_tag(client: httpx.Client, base: str, headers: dict, args):
    resp = client.get(f"{base}/topics/{args.topic}/tags/{args.tag}", headers=headers)
    if resp.status_code == 200:
        pages = resp.json()["pages"]
        if not pages:
            print(f"No pages with tag '{args.tag}'.")
            return
        for p in pages:
            print(f"  {p}")
    else:
        print(f"ERROR ({resp.status_code}): {resp.text}", file=sys.stderr)
        sys.exit(1)


def cmd_query_graph(client: httpx.Client, base: str, headers: dict, args):
    resp = client.get(f"{base}/topics/{args.topic}/graph", headers=headers)
    if resp.status_code == 200:
        data = resp.json()
        print(f"Nodes ({len(data['nodes'])}):")
        for n in data["nodes"]:
            tags = f" [{', '.join(n['tags'])}]" if n.get("tags") else ""
            print(f"  {n['id']}: {n['title']}{tags}")
        if data["edges"]:
            print(f"\nEdges ({len(data['edges'])}):")
            for e in data["edges"]:
                print(f"  {e['source']} -> {e['target']}")
    else:
        print(f"ERROR ({resp.status_code}): {resp.text}", file=sys.stderr)
        sys.exit(1)


def cmd_query_drafts(client: httpx.Client, base: str, headers: dict, args):
    resp = client.get(f"{base}/topics/{args.topic}/drafts", headers=headers)
    if resp.status_code == 200:
        drafts = resp.json()["drafts"]
        if not drafts:
            print("No pending drafts.")
            return
        for d in drafts:
            print(f"  {d}")
    else:
        print(f"ERROR ({resp.status_code}): {resp.text}", file=sys.stderr)
        sys.exit(1)


def cmd_discover_template(client: httpx.Client, base: str, headers: dict, args):
    if args.template_id:
        resp = client.get(f"{base}/templates/{args.template_id}")
    else:
        resp = client.get(f"{base}/templates")
    if resp.status_code == 200:
        data = resp.json()
        if "templates" in data:
            for t in data["templates"]:
                print(f"  {t['id']}: {t['name']} — {t['description']}")
        else:
            print(f"--- {data['name']} ---\n")
            print(data["content"])
    else:
        print(f"ERROR ({resp.status_code}): {resp.text}", file=sys.stderr)
        sys.exit(1)


def cmd_admin_approve(client: httpx.Client, base: str, headers: dict, args):
    resp = client.post(f"{base}/topics/{args.topic}/drafts/{args.path}/approve", headers=headers)
    if resp.status_code == 200:
        print(f"Approved: {resp.json()['path']}")
    else:
        print(f"ERROR ({resp.status_code}): {resp.text}", file=sys.stderr)
        sys.exit(1)


def cmd_admin_reject(client: httpx.Client, base: str, headers: dict, args):
    resp = client.post(
        f"{base}/topics/{args.topic}/drafts/{args.path}/reject",
        json={"reason": args.reason or ""},
        headers=headers,
    )
    if resp.status_code == 200:
        print(f"Rejected: {resp.json()['path']}")
    else:
        print(f"ERROR ({resp.status_code}): {resp.text}", file=sys.stderr)
        sys.exit(1)


def cmd_admin_clean_test_pages(client: httpx.Client, base: str, headers: dict, args):
    """Delete pages matching test patterns across all topics."""
    path_prefixes = ("_drafts/test-",)
    basename_prefixes = ("test-", "_deploy-validation-")
    resp = client.get(f"{base}/topics", headers=headers)
    if resp.status_code != 200:
        print(f"ERROR ({resp.status_code}): {resp.text}", file=sys.stderr)
        sys.exit(1)
    topics = resp.json()["topics"]
    deleted = 0
    for topic in topics:
        tid = topic["topic_id"]
        pages_resp = client.get(f"{base}/topics/{tid}/pages", headers=headers)
        if pages_resp.status_code != 200:
            continue
        for page in pages_resp.json()["pages"]:
            name = page.split("/")[-1]
            if any(page.startswith(p) for p in path_prefixes) or any(name.startswith(p) for p in basename_prefixes):
                if args.dry_run:
                    print(f"  [dry-run] would delete: {tid}/{page}")
                else:
                    del_resp = client.delete(f"{base}/topics/{tid}/pages/{page}", headers=headers)
                    if del_resp.status_code == 200:
                        print(f"  Deleted: {tid}/{page}")
                        deleted += 1
                    else:
                        print(f"  FAILED ({del_resp.status_code}): {tid}/{page}")
    if args.dry_run:
        print("\nDry run — no pages deleted. Remove --dry-run to delete.")
    else:
        print(f"\nCleaned {deleted} test page(s).")


def main():
    parser = argparse.ArgumentParser(description="Wiki Memory Service CLI")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Service base URL")
    parser.add_argument("--trust-domain", default=DEFAULT_TRUST_DOMAIN)
    parser.add_argument("--user", default="alice@example.com", help="User identity for query agent OBO")
    parser.add_argument("--topic", default="ai", help="Default topic for discovery agent")

    sub = parser.add_subparsers(dest="mode", required=True)

    # --- auth commands ---
    sub.add_parser("login", help="Login via GitHub device flow")
    sub.add_parser("logout", help="Remove cached token")
    sub.add_parser("whoami", help="Show current identity")
    sub.add_parser("renew", help="Renew expired or expiring token")

    # --- discover mode ---
    discover = sub.add_parser("discover", help="Discovery Agent operations (write)")
    discover_sub = discover.add_subparsers(dest="action", required=True)

    write_p = discover_sub.add_parser("write", help="Write a wiki page")
    write_p.add_argument("topic", help="Topic ID")
    write_p.add_argument("path", help="Page path (e.g. transformers.md)")
    write_p.add_argument("--content", help="Page content (string)")
    write_p.add_argument("--file", help="Read content from file")
    write_p.add_argument("--message", help="Commit message")
    write_p.add_argument("--draft", action="store_true", help="Submit as draft for review")

    novelty_p = discover_sub.add_parser("novelty", help="Check content novelty")
    novelty_p.add_argument("topic", help="Topic ID")
    novelty_p.add_argument("title", help="Content title")
    novelty_p.add_argument("abstract", help="Content abstract/summary")

    template_p = discover_sub.add_parser("template", help="List or get page templates")
    template_p.add_argument("template_id", nargs="?", help="Template ID (omit to list all)")

    # --- query mode ---
    query = sub.add_parser("query", help="Query Agent operations (read/search)")
    query_sub = query.add_subparsers(dest="action", required=True)

    query_sub.add_parser("list-topics", help="List all topics")

    lp = query_sub.add_parser("list-pages", help="List pages in a topic")
    lp.add_argument("topic", help="Topic ID")

    sp = query_sub.add_parser("search", help="Search a topic")
    sp.add_argument("topic", help="Topic ID")
    sp.add_argument("query", help="Search query")
    sp.add_argument("--limit", type=int, default=10)

    rp = query_sub.add_parser("read", help="Read a page")
    rp.add_argument("topic", help="Topic ID")
    rp.add_argument("path", help="Page path")

    act_p = query_sub.add_parser("activity", help="Recent changes")
    act_p.add_argument("topic", nargs="?", help="Topic ID (omit for global)")
    act_p.add_argument("--limit", type=int, default=20)

    bl_p = query_sub.add_parser("backlinks", help="Pages linking to a page")
    bl_p.add_argument("topic", help="Topic ID")
    bl_p.add_argument("path", help="Page path")

    sa_p = query_sub.add_parser("search-all", help="Search across all topics")
    sa_p.add_argument("query", help="Search query")
    sa_p.add_argument("--limit", type=int, default=10)

    tags_p = query_sub.add_parser("tags", help="List tags in a topic")
    tags_p.add_argument("topic", help="Topic ID")

    tag_p = query_sub.add_parser("tag", help="Pages with a specific tag")
    tag_p.add_argument("topic", help="Topic ID")
    tag_p.add_argument("tag", help="Tag name")

    graph_p = query_sub.add_parser("graph", help="Page graph (nodes + edges)")
    graph_p.add_argument("topic", help="Topic ID")

    drafts_p = query_sub.add_parser("drafts", help="List pending drafts")
    drafts_p.add_argument("topic", help="Topic ID")

    # --- admin mode ---
    admin = sub.add_parser("admin", help="Admin operations")
    admin_sub = admin.add_subparsers(dest="action", required=True)

    approve_p = admin_sub.add_parser("approve", help="Approve a draft")
    approve_p.add_argument("topic", help="Topic ID")
    approve_p.add_argument("path", help="Draft page path")

    reject_p = admin_sub.add_parser("reject", help="Reject a draft")
    reject_p.add_argument("topic", help="Topic ID")
    reject_p.add_argument("path", help="Draft page path")
    reject_p.add_argument("--reason", help="Rejection reason")

    admin_sub.add_parser("init-pages", help="Initialize GitHub Pages layout")

    clean_p = admin_sub.add_parser("clean-test-pages", help="Delete accumulated test pages")
    clean_p.add_argument("--dry-run", action="store_true", help="Show what would be deleted without deleting")

    args = parser.parse_args()
    insecure = os.environ.get("WIKI_INSECURE_TLS") == "1"
    client = httpx.Client(timeout=30, verify=not insecure)

    if args.mode == "login":
        cached = load_cached_token()
        base = cached["base_url"] if cached else args.base_url
        cmd_login(client, base, args)

    elif args.mode == "logout":
        cmd_logout(args)

    elif args.mode == "whoami":
        cached = load_cached_token()
        base = cached["base_url"] if cached else args.base_url
        cmd_whoami(client, base, args)

    elif args.mode == "renew":
        cached = load_cached_token()
        base = cached["base_url"] if cached else args.base_url
        cmd_renew(client, base)

    elif args.mode == "discover":
        auth_headers = make_auth_headers()
        if auth_headers:
            headers = auth_headers
        else:
            headers = make_discovery_headers(args.topic if hasattr(args, "topic") else "ai", args.trust_domain)
        if args.action == "write":
            cmd_discover_write(client, args.base_url, headers, args)
        elif args.action == "novelty":
            cmd_discover_novelty(client, args.base_url, headers, args)
        elif args.action == "template":
            cmd_discover_template(client, args.base_url, headers, args)

    elif args.mode == "query":
        auth_headers = make_auth_headers()
        if auth_headers:
            headers = auth_headers
        else:
            headers = make_query_headers(args.user, args.trust_domain)
        if args.action == "list-topics":
            cmd_query_list_topics(client, args.base_url, headers, args)
        elif args.action == "list-pages":
            cmd_query_list_pages(client, args.base_url, headers, args)
        elif args.action == "search":
            cmd_query_search(client, args.base_url, headers, args)
        elif args.action == "read":
            cmd_query_read(client, args.base_url, headers, args)
        elif args.action == "activity":
            cmd_query_activity(client, args.base_url, headers, args)
        elif args.action == "backlinks":
            cmd_query_backlinks(client, args.base_url, headers, args)
        elif args.action == "search-all":
            cmd_query_search_all(client, args.base_url, headers, args)
        elif args.action == "tags":
            cmd_query_tags(client, args.base_url, headers, args)
        elif args.action == "tag":
            cmd_query_tag(client, args.base_url, headers, args)
        elif args.action == "graph":
            cmd_query_graph(client, args.base_url, headers, args)
        elif args.action == "drafts":
            cmd_query_drafts(client, args.base_url, headers, args)

    elif args.mode == "admin":
        auth_headers = make_auth_headers()
        if auth_headers:
            headers = auth_headers
        else:
            headers = make_discovery_headers(getattr(args, "topic", "system"), args.trust_domain)
        if args.action == "approve":
            cmd_admin_approve(client, args.base_url, headers, args)
        elif args.action == "reject":
            cmd_admin_reject(client, args.base_url, headers, args)
        elif args.action == "init-pages":
            resp = client.post(f"{args.base_url}/admin/init-pages", headers=headers)
            if resp.status_code == 200:
                data = resp.json()
                print(f"GitHub Pages initialized ({len(data['files'])} files):")
                for f in data["files"]:
                    print(f"  {f}")
            else:
                print(f"ERROR ({resp.status_code}): {resp.json().get('detail', resp.text)}")
        elif args.action == "clean-test-pages":
            cmd_admin_clean_test_pages(client, args.base_url, headers, args)


if __name__ == "__main__":
    main()
