"""
Test all wiki skills as user bob-rossoctl using FastAPI TestClient.

Verifies query and discovery operations work correctly with OAuth user identity.

Run with: uv run --with pytest python -m pytest test_user_skills.py -v
"""

import os
import shutil
import sys
import tempfile
import time

import pytest

os.environ["WIKI_ROOT"] = tempfile.mkdtemp()
os.environ["ACL_FILE"] = os.path.join(os.path.dirname(__file__), "test_acl.yaml")
os.environ["JWT_SECRET_KEY"] = "test-secret-key"
os.environ["JWT_EXPIRY_HOURS"] = "8"

# Use httpx for TestClient compatibility
from starlette.testclient import TestClient  # noqa: E402
from wiki_service import WIKI_ROOT, _ensure_repo, _sign_jwt, app  # noqa: E402

client = TestClient(app)

USER_LOGIN = "bob-rossoctl"
USER_TOKEN = _sign_jwt(
    {
        "sub": f"github:{USER_LOGIN}",
        "github_login": USER_LOGIN,
        "email": "bob@rossoctl.io",
        "groups": ["rossoctl/ml-team"],
        "iss": "wiki-memory-service",
        "iat": int(time.time()),
        "exp": int(time.time()) + 8 * 3600,
    }
)

AUTH_HEADERS = {"Authorization": f"Bearer {USER_TOKEN}"}

DISCOVERY_HEADERS = {
    "X-Spiffe-Id": "spiffe://rossoctl.example.com/ns/topic-ai/sa/discovery-agent",
}


@pytest.fixture(autouse=True, scope="session")
def seed_wiki_data():
    """Seed test pages once before the entire test session."""
    setup_test_data()


def separator(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def setup_test_data():
    """Write test pages using discovery agent so user can query them."""
    _ensure_repo()
    pages = {
        "transformers.md": "---\ntags: [paper, architecture]\n---\n# Transformer Architecture\n\nSelf-attention mechanisms for parallel sequence processing.\n\n## Links\n- See also [rag-patterns.md](rag-patterns.md)\n",
        "rag-patterns.md": "---\ntags: [paper, retrieval]\n---\n# RAG Patterns\n\nRetrieval-Augmented Generation combines retrievers with generators.\n\n## Links\n- Based on [[transformers]]\n",
        "fine-tuning.md": "---\ntags: [technique, training]\n---\n# Fine-Tuning\n\nLoRA, QLoRA, and full fine-tuning approaches.\n",
        "evaluation.md": "---\ntags: [metrics]\n---\n# LLM Evaluation\n\nPerplexity, BLEU, ROUGE, and human preference ratings.\n",
    }
    for path, content in pages.items():
        resp = client.post(
            f"/topics/ai/pages/{path}",
            json={"content": content, "message": f"Add {path}"},
            headers=DISCOVERY_HEADERS,
        )
        assert resp.status_code == 200, f"Setup failed for {path}: {resp.text}"


def test_whoami():
    separator("TEST: whoami")
    resp = client.get("/auth/whoami", headers=AUTH_HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    assert data["subject"] == f"github:{USER_LOGIN}"
    assert data["kind"] == "user"
    print(f"  User:   {data['subject']}")
    print(f"  Kind:   {data['kind']}")
    print(f"  Groups: {data['groups']}")
    print("  PASS")


def test_query_list_topics():
    separator("TEST: query list-topics")
    resp = client.get("/topics", headers=AUTH_HEADERS)
    assert resp.status_code == 200
    topics = resp.json()["topics"]
    print(f"  Topics: {[t['topic_id'] for t in topics]}")
    assert any(t["topic_id"] == "ai" for t in topics)
    print("  PASS")


def test_query_list_pages():
    separator("TEST: query list-pages")
    resp = client.get("/topics/ai/pages", headers=AUTH_HEADERS)
    assert resp.status_code == 200
    pages = resp.json()["pages"]
    print(f"  Pages: {pages}")
    assert "transformers.md" in pages
    print("  PASS")


def test_query_search():
    separator("TEST: query search")
    resp = client.post(
        "/topics/ai/query",
        json={"query": "attention transformer", "limit": 3},
        headers=AUTH_HEADERS,
    )
    assert resp.status_code == 200
    results = resp.json()["results"]
    print(f"  Results: {len(results)}")
    for r in results:
        print(f"    - {r['path']} (score={r['score']:.3f})")
    assert len(results) > 0
    print("  PASS")


def test_query_search_all():
    separator("TEST: query search-all (global)")
    resp = client.post(
        "/search",
        json={"query": "retrieval", "limit": 5},
        headers=AUTH_HEADERS,
    )
    assert resp.status_code == 200
    results = resp.json()["results"]
    print(f"  Results: {len(results)}")
    for r in results:
        print(f"    - [{r.get('topic_id')}] {r['path']} (score={r['score']:.3f})")
    assert len(results) > 0
    print("  PASS")


def test_query_read():
    separator("TEST: query read")
    resp = client.get("/topics/ai/pages/transformers.md", headers=AUTH_HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    assert "Transformer" in data["content"]
    assert data.get("frontmatter", {}).get("tags") == ["paper", "architecture"]
    print(f"  Content: {data['content'][:80]}...")
    print(f"  Frontmatter: {data['frontmatter']}")
    print("  PASS")


def test_query_activity():
    separator("TEST: query activity")
    resp = client.get("/topics/ai/activity?limit=5", headers=AUTH_HEADERS)
    assert resp.status_code == 200
    entries = resp.json()["entries"]
    print(f"  Entries: {len(entries)}")
    for e in entries[:3]:
        print(f"    - {e['message']} by {e['author']}")
    assert len(entries) > 0
    print("  PASS")


def test_query_backlinks():
    separator("TEST: query backlinks")
    resp = client.get("/topics/ai/backlinks/transformers.md", headers=AUTH_HEADERS)
    assert resp.status_code == 200
    backlinks = resp.json()["backlinks"]
    print(f"  Backlinks to transformers.md: {backlinks}")
    assert "rag-patterns.md" in backlinks
    print("  PASS")


def test_query_tags():
    separator("TEST: query tags")
    resp = client.get("/topics/ai/tags", headers=AUTH_HEADERS)
    assert resp.status_code == 200
    tags = resp.json()["tags"]
    print(f"  Tags: {tags}")
    assert "paper" in tags
    print("  PASS")

    print("\n  [pages with tag 'paper']")
    resp = client.get("/topics/ai/tags/paper", headers=AUTH_HEADERS)
    assert resp.status_code == 200
    pages = resp.json()["pages"]
    print(f"  Pages: {pages}")
    assert len(pages) >= 2
    print("  PASS")


def test_query_graph():
    separator("TEST: query graph")
    resp = client.get("/topics/ai/graph", headers=AUTH_HEADERS)
    assert resp.status_code == 200
    graph = resp.json()
    nodes = graph["nodes"]
    edges = graph["edges"]
    print(f"  Nodes: {len(nodes)}, Edges: {len(edges)}")
    for n in nodes[:3]:
        print(f"    - {n['id']} (tags={n.get('tags', [])})")
    for e in edges[:3]:
        print(f"    - {e['source']} -> {e['target']}")
    assert len(nodes) >= 4
    assert len(edges) >= 1
    print("  PASS")


def test_discovery_templates():
    separator("TEST: discovery templates")
    resp = client.get("/templates", headers=AUTH_HEADERS)
    assert resp.status_code == 200
    templates = resp.json()["templates"]
    print(f"  Templates: {[t['id'] for t in templates]}")
    assert len(templates) >= 4
    print("  PASS")

    print("\n  [get paper-summary template]")
    resp = client.get("/templates/paper-summary", headers=AUTH_HEADERS)
    assert resp.status_code == 200
    print(f"  Content: {resp.json()['content'][:100]}...")
    print("  PASS")


def test_discovery_novelty():
    separator("TEST: discovery novelty check")
    resp = client.post(
        "/topics/ai/check-novelty",
        json={"title": "Transformer Architecture", "abstract": "Self-attention for sequences"},
        headers=AUTH_HEADERS,
    )
    assert resp.status_code == 200
    data = resp.json()
    print(f"  Novel: {data['novel']} (existing page should not be novel)")
    assert data["novel"] is False
    print("  PASS")

    resp = client.post(
        "/topics/ai/check-novelty",
        json={"title": "Quantum Computing for ML", "abstract": "Quantum advantage in optimization"},
        headers=AUTH_HEADERS,
    )
    assert resp.status_code == 200
    data = resp.json()
    print(f"  Novel: {data['novel']} (new topic should be novel)")
    assert data["novel"] is True
    print("  PASS")


def test_discovery_write_draft():
    separator("TEST: discovery write (draft mode)")
    content = "---\ntags: [test]\n---\n# Test Draft\n\nWritten by discovery agent as draft."
    resp = client.post(
        "/topics/ai/pages/test-draft.md?draft=true",
        json={"content": content, "message": "Test draft write"},
        headers=DISCOVERY_HEADERS,
    )
    assert resp.status_code == 200, f"Got {resp.status_code}: {resp.text}"
    data = resp.json()
    print(f"  Path: {data['path']}")
    print("  PASS")

    print("\n  [list drafts as discovery agent (write access required)]")
    resp = client.get("/topics/ai/drafts", headers=DISCOVERY_HEADERS)
    assert resp.status_code == 200
    drafts = resp.json()["drafts"]
    print(f"  Drafts: {drafts}")
    assert len(drafts) >= 1
    print("  PASS")


def test_discovery_write_with_suggested_links():
    separator("TEST: discovery write (suggested links)")
    content = "---\ntags: [technique]\n---\n# Attention Mechanisms\n\nMulti-head attention in transformers."
    resp = client.post(
        "/topics/ai/pages/attention.md",
        json={"content": content, "message": "Add attention page"},
        headers=DISCOVERY_HEADERS,
    )
    assert resp.status_code == 200, f"Got {resp.status_code}: {resp.text}"
    data = resp.json()
    links = data.get("suggested_links", [])
    print(f"  Suggested links: {links}")
    print("  PASS")


def test_user_write_blocked():
    separator("TEST: user write blocked (ACL enforcement)")
    resp = client.post(
        "/topics/ai/pages/unauthorized.md",
        json={"content": "# Should fail", "message": "unauthorized"},
        headers=AUTH_HEADERS,
    )
    assert resp.status_code == 403, f"Expected 403 but got {resp.status_code}"
    print(f"  Correctly blocked: {resp.json()['detail']}")
    print("  PASS")


def test_token_renew():
    separator("TEST: token renew")
    resp = client.post("/auth/renew", headers=AUTH_HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    print(f"  Renewed for: {data['github_login']}")
    print(f"  Expires in:  {data['expires_in'] // 3600}h")
    assert data["github_login"] == USER_LOGIN
    print("  PASS")


def main():
    print(f"Testing wiki skills as user: {USER_LOGIN}")
    print(f"Wiki root: {WIKI_ROOT}")

    setup_test_data()

    tests = [
        test_whoami,
        test_query_list_topics,
        test_query_list_pages,
        test_query_search,
        test_query_search_all,
        test_query_read,
        test_query_activity,
        test_query_backlinks,
        test_query_tags,
        test_query_graph,
        test_discovery_templates,
        test_discovery_novelty,
        test_discovery_write_draft,
        test_discovery_write_with_suggested_links,
        test_user_write_blocked,
        test_token_renew,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  FAIL: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            failed += 1

    separator("RESULTS")
    print(f"  Passed: {passed}/{passed + failed}")
    if failed:
        print(f"  Failed: {failed}")
        sys.exit(1)
    print("  All wiki skills verified for user bob-rossoctl!")

    # Cleanup
    shutil.rmtree(WIKI_ROOT, ignore_errors=True)


if __name__ == "__main__":
    main()
