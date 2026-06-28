"""Xquik MCP tool example."""

import json
import logging
import os
import sys
from typing import Any

import requests
from fastmcp import FastMCP

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    stream=sys.stdout,
    format="%(levelname)s: %(message)s",
)
logging.getLogger("urllib3").setLevel(logging.INFO)

XQUIK_API_KEY = os.getenv("XQUIK_API_KEY")
XQUIK_BASE_URL = os.getenv("XQUIK_BASE_URL", "https://xquik.com").rstrip("/")
REQUEST_TIMEOUT = int(os.getenv("XQUIK_REQUEST_TIMEOUT", "20"))

mcp = FastMCP("Xquik X Data")


def _bounded_int(value: int, *, minimum: int, maximum: int) -> int:
    """Clamp an integer to the allowed API range."""
    return max(minimum, min(value, maximum))


def _fetch_json(path: str, params: dict[str, Any]) -> dict[str, Any]:
    """Call a Xquik REST route and return parsed JSON."""
    if not XQUIK_API_KEY:
        return {"error": "XQUIK_API_KEY is not configured"}

    headers = {
        "accept": "application/json",
        "x-api-key": XQUIK_API_KEY,
    }

    try:
        response = requests.get(
            f"{XQUIK_BASE_URL}{path}",
            headers=headers,
            params=params,
            timeout=REQUEST_TIMEOUT,
        )
        if response.status_code in {401, 403}:
            return {"error": "Xquik authentication failed"}
        if response.status_code == 429:
            return {"error": "Xquik rate limit reached"}

        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict):
            return data
        return {"data": data}
    except requests.RequestException as exc:
        logger.warning("Xquik request failed for %s: %s", path, exc)
        return {"error": "Xquik request failed"}
    except ValueError:
        logger.warning("Xquik returned non-JSON data for %s", path)
        return {"error": "Xquik returned non-JSON data"}


@mcp.tool(annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True})
def search_x_posts(
    query: str,
    query_type: str = "Latest",
    limit: int = 20,
    since_time: str | None = None,
    until_time: str | None = None,
) -> str:
    """Search public X posts with Xquik."""
    params: dict[str, Any] = {
        "q": query,
        "queryType": query_type,
        "limit": _bounded_int(limit, minimum=1, maximum=100),
    }
    if since_time:
        params["sinceTime"] = since_time
    if until_time:
        params["untilTime"] = until_time

    return json.dumps(_fetch_json("/api/v1/x/tweets/search", params), ensure_ascii=False)


@mcp.tool(annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True})
def search_x_users(query: str, cursor: str | None = None) -> str:
    """Search public X users with Xquik."""
    params: dict[str, Any] = {"q": query}
    if cursor:
        params["cursor"] = cursor
    return json.dumps(_fetch_json("/api/v1/x/users/search", params), ensure_ascii=False)


@mcp.tool(annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True})
def get_x_trends(woeid: int = 1, count: int = 30) -> str:
    """Fetch public X trends with Xquik."""
    params = {
        "woeid": woeid,
        "count": _bounded_int(count, minimum=1, maximum=50),
    }
    return json.dumps(_fetch_json("/api/v1/x/trends", params), ensure_ascii=False)


def run_server() -> None:
    """Run the MCP server."""
    if not XQUIK_API_KEY:
        logger.warning("Please configure XQUIK_API_KEY before invoking Xquik tools")
    transport = os.getenv("MCP_TRANSPORT", "streamable-http")
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    mcp.run(transport=transport, host=host, port=port)


if __name__ == "__main__":
    run_server()
