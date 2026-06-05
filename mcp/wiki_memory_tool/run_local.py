"""
Local runner for Wiki Memory Service — no container, no k8s.
Starts both the REST API (FastAPI on :8321) and MCP server (streamable-http on :8322).

Usage: python run_local.py [--clean] [--remote]
"""

import os
import shutil
import sys
import threading
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "data" / "wiki"

os.environ["WIKI_ROOT"] = str(DATA_DIR)
os.environ["ACL_FILE"] = str(SCRIPT_DIR / "test_acl.yaml")
os.environ["SPIFFE_TRUST_DOMAIN"] = "kagenti.example.com"
os.environ.setdefault("JWT_SECRET_KEY", "local-dev-secret-do-not-use-in-production")
os.environ["MCP_TRANSPORT"] = "streamable-http"
os.environ["MCP_PORT"] = "8322"

if "--clean" in sys.argv:
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
        print(f"Cleaned {DATA_DIR}")

if "--remote" in sys.argv:
    remote_url = os.environ.get("WIKI_REMOTE_URL", "")
    if not remote_url:
        print("ERROR: --remote requires WIKI_REMOTE_URL env var to be set")
        sys.exit(1)
    print(f"Remote:    {remote_url.split('@')[0]}@***")
else:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    import uvicorn

    print(f"Wiki root: {DATA_DIR}")
    print(f"ACL file:  {os.environ['ACL_FILE']}")
    print("Starting wiki-memory-service REST API on http://localhost:8321")
    print("Starting wiki-memory-service MCP server on http://localhost:8322/mcp")

    def start_mcp():
        from mcp_server import run_mcp_server

        run_mcp_server()

    mcp_thread = threading.Thread(target=start_mcp, daemon=True)
    mcp_thread.start()

    uvicorn.run("wiki_service:app", host="0.0.0.0", port=8321, reload=False)
