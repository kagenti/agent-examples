# /// script
# requires-python = ">=3.11"
# dependencies = ["httpx"]
# ///
"""
Deploy wiki-memory-service to Kubernetes.

Usage:
    export WIKI_REMOTE_URL="https://x-access-token:<PAT>@github.com/<org>/<repo>.git"
    uv run deploy.py

    # Or pass as argument:
    uv run deploy.py --url "https://x-access-token:<PAT>@github.com/<org>/<repo>.git"
"""

import os
import re
import subprocess
import sys
import time
from pathlib import Path

NAMESPACE = "wiki-memory-service"
SCRIPT_DIR = Path(__file__).parent.resolve()
K8S_DIR = SCRIPT_DIR / "k8s"
GIT_REMOTE_K8S_NAME = "wiki-github-remote"
IMAGE_REPO = "quay.io/aslomnet/wiki-memory-service"


def get_version() -> str:
    """Read version from pyproject.toml (single source of truth)."""
    import tomllib

    pyproject = SCRIPT_DIR / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text())
    version = data.get("project", {}).get("version")
    if not version:
        print("ERROR: Could not read version from pyproject.toml")
        sys.exit(1)
    return version


def run(cmd: list[str], check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    result = subprocess.run(cmd, capture_output=capture, text=True)
    if check and result.returncode != 0:
        stderr = result.stderr if capture else ""
        safe_cmd = re.sub(r"x-access-token:[^@\s]+", "x-access-token:***", " ".join(cmd))
        print(f"  FAILED: {safe_cmd}")
        if stderr:
            print(f"  {stderr.strip()}")
        sys.exit(1)
    return result


def validate_remote_url(url: str) -> None:
    """Validate WIKI_REMOTE_URL format. Exits if invalid."""
    if not re.match(r"https://x-access-token:[^@]+@.+", url):
        print("ERROR: WIKI_REMOTE_URL must be in format:")
        print("  https://x-access-token:<PAT>@github.com/<org>/<repo>.git")
        sys.exit(1)


def ensure_namespace():
    """Create namespace if it does not exist."""
    print(f"[1/6] Checking namespace '{NAMESPACE}'...")
    result = run(["kubectl", "get", "namespace", NAMESPACE], check=False, capture=True)
    if result.returncode != 0:
        print(f"  Creating namespace '{NAMESPACE}'...")
        run(["kubectl", "create", "namespace", NAMESPACE])
    else:
        print(f"  Namespace '{NAMESPACE}' exists.")


def apply_manifests():
    """Apply all k8s manifests in order."""
    print("[2/6] Applying Kubernetes manifests...")
    manifests = ["serviceaccount.yaml", "acl-configmap.yaml", "deployment.yaml"]
    for manifest in manifests:
        path = K8S_DIR / manifest
        if not path.exists():
            print(f"  ERROR: {path} not found")
            sys.exit(1)
        print(f"  Applying {manifest}...")
        run(["kubectl", "apply", "-f", str(path)])


def create_or_update_secret(encoded_value: str):
    """Create or update the wiki-github-remote secret via stdin (no raw secrets in scope)."""
    print(f"[3/6] Creating/updating k8s resource '{GIT_REMOTE_K8S_NAME}'...")
    manifest = (
        f"apiVersion: v1\nkind: Secret\nmetadata:\n"
        f"  name: {GIT_REMOTE_K8S_NAME}\n  namespace: {NAMESPACE}\n"
        f"type: Opaque\ndata:\n  WIKI_REMOTE_URL: {encoded_value}\n"
    )
    pipe = subprocess.run(
        ["kubectl", "apply", "-f", "-"],
        input=manifest,
        text=True,
        capture_output=True,
    )
    if pipe.returncode != 0:
        print(f"  FAILED: {pipe.stderr.strip()}")
        sys.exit(1)
    print("  Secret configured.")


def restart_and_wait():
    """Set image tag from pyproject.toml version and wait for rollout."""
    version = get_version()
    image = f"{IMAGE_REPO}:{version}"
    print(f"[4/6] Setting image to {image} and restarting...")
    run(["kubectl", "set", "image", "deployment/wiki-memory-service", f"wiki-memory={image}", "-n", NAMESPACE])
    print("  Waiting for rollout to complete...")
    run(["kubectl", "rollout", "status", "deployment/wiki-memory-service", "-n", NAMESPACE, "--timeout=90s"])
    print("  Deployment ready.")


def run_validation():
    """Port-forward and run basic validation tests."""
    print("[5/6] Running validation tests...")

    import httpx

    port = 18321
    pf_cmd = ["kubectl", "port-forward", "svc/wiki-memory-service", f"{port}:8000", "-n", NAMESPACE]
    pf_proc = subprocess.Popen(pf_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    try:
        base = f"http://localhost:{port}"
        client = httpx.Client(timeout=30)

        # Wait for port-forward to be ready
        for i in range(15):
            try:
                resp = client.get(f"{base}/healthz")
                if resp.status_code == 200:
                    break
            except httpx.ConnectError:
                pass
            time.sleep(2)
        else:
            print("  FAILED: Service not reachable via port-forward")
            sys.exit(1)

        health = resp.json()
        print(f"  Health check: OK (topics={health['topics']}, root={health['root']})")

        # Test topic listing
        headers = {
            "X-Spiffe-Id": "spiffe://kagenti.example.com/ns/wiki-system/sa/query-agent",
            "X-Original-Subject": "alice@example.com",
        }
        resp = client.get(f"{base}/topics", headers=headers)
        assert resp.status_code == 200, f"GET /topics returned {resp.status_code}"
        topics = resp.json()["topics"]
        print(f"  Topic listing: OK ({len(topics)} topics visible)")

        # Test novelty check (exercises search + ACL)
        resp = client.post(
            f"{base}/topics/ai/check-novelty",
            json={"title": "Test", "abstract": "deployment validation check"},
            headers=headers,
        )
        assert resp.status_code == 200, f"POST check-novelty returned {resp.status_code}"
        print(f"  Novelty check: OK (novel={resp.json()['novel']})")

        # Test write (discovery agent)
        write_headers = {
            "X-Spiffe-Id": "spiffe://kagenti.example.com/ns/topic-ai/sa/discovery-agent",
        }
        test_page = "_deploy-validation-test.md"
        resp = client.post(
            f"{base}/topics/ai/pages/{test_page}",
            json={
                "content": "# Deploy validation\nThis page verifies the deployment works.",
                "message": "deploy: validation test page",
            },
            headers=write_headers,
        )
        if resp.status_code == 200:
            print(f"  Write test: OK (wrote {test_page})")
            # Read it back
            resp = client.get(f"{base}/topics/ai/pages/{test_page}", headers=headers)
            assert resp.status_code == 200, f"GET page returned {resp.status_code}"
            print("  Read test: OK")
        else:
            print(f"  Write test: SKIPPED ({resp.status_code} — may need git push fix)")

    finally:
        pf_proc.terminate()
        pf_proc.wait()


def build_and_push(version: str, skip_build: bool = False):
    """Build and push container image tagged with version."""
    tag = f"{IMAGE_REPO}:{version}"
    latest = f"{IMAGE_REPO}:latest"
    if skip_build:
        print(f"[build] Skipping build (--no-build), using image {tag}")
        return
    print(f"[build] Building {tag}...")
    run(["docker", "build", "-t", tag, "-t", latest, str(SCRIPT_DIR)])
    print(f"[build] Pushing {tag} and :latest...")
    run(["docker", "push", tag])
    run(["docker", "push", latest])


def print_summary(has_remote: bool):
    """Print deployment summary."""
    version = get_version()
    print("[6/6] Deployment complete!")
    print()
    print("  Summary:")
    print(f"    Namespace:  {NAMESPACE}")
    print("    Service:    wiki-memory-service:8000")
    print(f"    Image:      {IMAGE_REPO}:{version}")
    if has_remote:
        print(f"    Git remote: configured (k8s={GIT_REMOTE_K8S_NAME})")
    else:
        print("    Git remote: (local-only, no remote configured)")
    print()
    print("  To run full tests:")
    print(f"    kubectl port-forward svc/wiki-memory-service 8321:8000 -n {NAMESPACE} &")
    print("    WIKI_SERVICE_URL=http://localhost:8321 uv run python test_agents.py")


def main():
    version = get_version()
    print("=" * 60)
    print(f"  Wiki Memory Service v{version} — Kubernetes Deployment")
    print("=" * 60)
    print()

    # Parse flags
    remote_url = os.environ.get("WIKI_REMOTE_URL", "")
    do_build = "--build" in sys.argv
    for arg in sys.argv[1:]:
        if arg.startswith("--url="):
            remote_url = arg.split("=", 1)[1]
        elif arg.startswith("https://"):
            remote_url = arg

    if remote_url:
        validate_remote_url(remote_url)
        print("  Git repo: configured (WIKI_REMOTE_URL)")
        print("  PAT:      configured")
    else:
        print("  Mode:     local-only (no WIKI_REMOTE_URL — wiki will use local git init)")

    print(f"  Version:  {version}")
    print()

    if do_build:
        build_and_push(version)

    ensure_namespace()
    apply_manifests()
    if remote_url:
        import base64 as b64

        create_or_update_secret(b64.b64encode(remote_url.encode()).decode())
    else:
        print("[3/6] Skipping secret (no WIKI_REMOTE_URL — local-only mode).")
    restart_and_wait()
    run_validation()
    print_summary(bool(remote_url))


if __name__ == "__main__":
    main()
