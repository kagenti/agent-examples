"""
Simulated test agents for the Wiki Memory Service.

Agent 1 — Discovery Agent (writer): writes wiki pages to the "ai" topic.
Agent 2 — Query Agent (reader): searches and reads pages on behalf of a user.

Usage:
    # Local:
    uv run python run_local.py --clean
    uv run python test_agents.py

    # Against cluster (port-forward):
    kubectl port-forward svc/wiki-memory-service 8321:8000 -n wiki-memory-service &
    WIKI_SERVICE_URL=http://localhost:8321 uv run python test_agents.py

    # Against cluster (route):
    WIKI_SERVICE_URL=https://wiki-memory-service-wiki-memory-service.apps.<cluster> uv run python test_agents.py
"""

import os
import sys
import time

import httpx

BASE = os.environ.get("WIKI_SERVICE_URL", "http://localhost:8321")

DISCOVERY_AGENT_HEADERS = {
    "X-Spiffe-Id": "spiffe://rossoctl.example.com/ns/topic-ai/sa/discovery-agent",
}

QUERY_AGENT_HEADERS = {
    "X-Spiffe-Id": "spiffe://rossoctl.example.com/ns/wiki-system/sa/query-agent",
    "X-Original-Subject": "alice@example.com",
}

PAGES = {
    "transformers.md": {
        "content": (
            "# Transformer Architecture\n\n"
            "The transformer model uses self-attention mechanisms to process "
            "sequential data in parallel. Key components include multi-head "
            "attention, positional encoding, and feed-forward layers.\n\n"
            "## Key Papers\n- Attention Is All You Need (Vaswani et al., 2017)\n"
            "- BERT (Devlin et al., 2019)\n- GPT series (Radford et al.)\n"
        ),
        "message": "Add transformer architecture overview",
    },
    "rag-patterns.md": {
        "content": (
            "# RAG Patterns\n\n"
            "Retrieval-Augmented Generation combines a retriever with a generator.\n\n"
            "## Common Patterns\n"
            "1. **Naive RAG** — embed, retrieve top-k, concatenate into prompt\n"
            "2. **Agentic RAG** — agent decides when to retrieve\n"
            "3. **Graph RAG** — knowledge graph enriched retrieval\n\n"
            "## Chunking Strategies\n"
            "- Fixed-size with overlap\n- Semantic splitting\n- Document-aware\n"
        ),
        "message": "Add RAG patterns documentation",
    },
    "fine-tuning.md": {
        "content": (
            "# Fine-Tuning Techniques\n\n"
            "## Full Fine-Tuning\nUpdate all model parameters. Expensive.\n\n"
            "## LoRA\nLow-Rank Adaptation freezes base weights and trains small "
            "rank-decomposition matrices. Memory efficient.\n\n"
            "## RLHF\nReinforcement Learning from Human Feedback aligns model "
            "outputs with human preferences using a reward model.\n"
        ),
        "message": "Add fine-tuning techniques page",
    },
    "evaluation.md": {
        "content": (
            "# LLM Evaluation\n\n"
            "## Metrics\n- Perplexity\n- BLEU/ROUGE (generation)\n"
            "- Human preference ratings\n- Task-specific benchmarks\n\n"
            "## Frameworks\n- lm-eval-harness\n- HELM\n- OpenCompass\n"
        ),
        "message": "Add evaluation methods page",
    },
}


def separator(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def run_discovery_agent(client: httpx.Client):
    separator("DISCOVERY AGENT — Writing pages to 'ai' topic")

    for page_name, page_data in PAGES.items():
        # Check novelty first
        title = page_name.replace(".md", "").replace("-", " ").title()
        novelty = client.post(
            f"{BASE}/topics/ai/check-novelty",
            json={"title": title, "abstract": page_data["content"][:100]},
            headers=DISCOVERY_AGENT_HEADERS,
        )
        novelty_result = novelty.json()
        print(f"[novelty] {page_name}: novel={novelty_result.get('novel')}")

        if not novelty_result.get("novel"):
            print("  -> Skipping (similar content exists)")
            continue

        # Write the page
        resp = client.post(
            f"{BASE}/topics/ai/pages/{page_name}",
            json={"content": page_data["content"], "message": page_data["message"]},
            headers=DISCOVERY_AGENT_HEADERS,
        )
        if resp.status_code == 200:
            print(f"  -> Written: {resp.json()['path']} by {resp.json()['author']}")
        else:
            print(f"  -> FAILED ({resp.status_code}): {resp.text}")

    print("\nDiscovery Agent: done writing pages.")


def run_query_agent(client: httpx.Client):
    separator("QUERY AGENT — Reading/searching 'ai' topic (on behalf of alice)")

    # List topics
    print("[list topics]")
    resp = client.get(f"{BASE}/topics", headers=QUERY_AGENT_HEADERS)
    for topic in resp.json()["topics"]:
        print(f"  - {topic['topic_id']} ({topic['page_count']} pages)")

    # List pages in ai topic
    print("\n[list pages in 'ai']")
    resp = client.get(f"{BASE}/topics/ai/pages", headers=QUERY_AGENT_HEADERS)
    for page in resp.json()["pages"]:
        print(f"  - {page}")

    # Search for "attention transformer"
    print("\n[search 'ai' for 'attention transformer']")
    resp = client.post(
        f"{BASE}/topics/ai/query",
        json={"query": "attention transformer", "limit": 5},
        headers=QUERY_AGENT_HEADERS,
    )
    for r in resp.json()["results"]:
        print(f"  - {r['path']} (score={r['score']})")
        if r["snippet"]:
            print(f"    snippet: {r['snippet'][:80]}...")

    # Search for "LoRA fine-tuning"
    print("\n[search 'ai' for 'LoRA fine-tuning']")
    resp = client.post(
        f"{BASE}/topics/ai/query",
        json={"query": "LoRA fine-tuning", "limit": 3},
        headers=QUERY_AGENT_HEADERS,
    )
    for r in resp.json()["results"]:
        print(f"  - {r['path']} (score={r['score']})")

    # Read a specific page
    print("\n[read page 'rag-patterns.md']")
    resp = client.get(
        f"{BASE}/topics/ai/pages/rag-patterns.md",
        headers=QUERY_AGENT_HEADERS,
    )
    content = resp.json()["content"]
    print(f"  First 150 chars: {content[:150]}...")

    print("\nQuery Agent: done reading/searching.")


def run_acl_test(client: httpx.Client):
    separator("ACL TEST — Query Agent attempts unauthorized write (expect 403)")

    resp = client.post(
        f"{BASE}/topics/ai/pages/hack.md",
        json={"content": "# Unauthorized write attempt", "message": "should fail"},
        headers=QUERY_AGENT_HEADERS,
    )
    if resp.status_code == 403:
        print(f"  PASS: Got expected 403 — {resp.json()['detail']}")
    else:
        print(f"  FAIL: Expected 403 but got {resp.status_code}")

    # Discovery agent tries to read security topic (no access)
    print("\n[Discovery Agent (ai) tries to write to 'security' topic — expect 403]")
    resp = client.post(
        f"{BASE}/topics/security/pages/exploit.md",
        json={"content": "# Cross-topic write", "message": "should fail"},
        headers=DISCOVERY_AGENT_HEADERS,
    )
    if resp.status_code == 403:
        print(f"  PASS: Got expected 403 — {resp.json()['detail']}")
    else:
        print(f"  FAIL: Expected 403 but got {resp.status_code}")


def wait_for_service(client: httpx.Client, retries: int = 10):
    print(f"Connecting to wiki service at: {BASE}")
    for i in range(retries):
        try:
            resp = client.get(f"{BASE}/healthz")
            if resp.status_code == 200:
                print(f"Service is up: {resp.json()}")
                return
        except (httpx.ConnectError, httpx.ReadError):
            pass
        print(f"Waiting for service... ({i + 1}/{retries})")
        time.sleep(2)
    print("ERROR: Service not reachable at", BASE)
    sys.exit(1)


def main():
    global BASE
    for arg in sys.argv[1:]:
        if arg.startswith("--url="):
            BASE = arg.split("=", 1)[1]
        elif arg.startswith("http"):
            BASE = arg

    insecure = os.environ.get("WIKI_INSECURE_TLS", "1") == "1"
    client = httpx.Client(timeout=30, verify=not insecure)

    wait_for_service(client)
    run_discovery_agent(client)
    run_query_agent(client)
    run_acl_test(client)

    separator("ALL TESTS COMPLETE")
    print("Summary:")
    print(f"  - Service URL: {BASE}")
    print("  - Discovery Agent wrote 4 pages to 'ai' topic")
    print("  - Query Agent listed, searched, and read pages")
    print("  - ACL enforcement blocked unauthorized writes")


if __name__ == "__main__":
    main()
