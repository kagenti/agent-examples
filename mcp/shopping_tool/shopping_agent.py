"""Shopping Agent MCP Tool - Uses SerpAPI for product search"""

import argparse
import os
import sys
import json
import logging
from fastmcp import FastMCP
from serpapi import GoogleSearch

logger = logging.getLogger(__name__)


def _env_flag(name: str, default: str = "false") -> bool:
    """Parse environment flag strings like 1/true/on into booleans."""
    value = os.getenv(name)
    if value is None:
        value = default
    return value.strip().lower() in {"1", "true", "yes", "on"}

# Environment variable for API key
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

# Initialize FastMCP
mcp = FastMCP("Shopping Agent")

# Public agent card for A2A discovery
AGENT_CARD = {
    "name": "shopping-agent",
    "version": "1.0.0",
    "description": "Shopping MCP tool using SerpAPI for product search",
    "capabilities": {
        "mcp": {
            # Using cluster DNS; adjust if you front this with ingress
            "endpoints": [
                {
                    "type": "http",
                    "url": os.getenv(
                        "AGENT_CARD_MCP_URL",
                        "http://shopping-agent:8000/mcp",
                    ),
                }
            ]
        }
    },
}


@mcp.tool(annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True})
def recommend_products(query: str, max_results: int = 10) -> str:
    """
    Recommend products based on natural language query (e.g., "good curtains under $40")
    
    This tool searches Google Shopping via SerpAPI and returns structured product data
    including titles, prices, and descriptions.
    
    Args:
        query: Natural language product request
        max_results: Maximum number of product recommendations to return (default 10, max 20)
    
    Returns:
        JSON string containing product search results with names, prices, descriptions, and links.
    """
    # Validate query input
    if not isinstance(query, str) or not query.strip():
        return json.dumps({"error": "Query must not be empty."})
    if len(query) > 256:
        return json.dumps({"error": "Query is too long (max 256 characters)."})
    logger.info(f"Searching products for query: '{query}'")
    
    if not SERPAPI_API_KEY:
        return json.dumps({"error": "SERPAPI_API_KEY not configured"})
    
    # Limit max_results
    max_results = min(max_results, 20)
    
    try:
        # Configure SerpAPI Google Shopping search
        params = {
            "api_key": SERPAPI_API_KEY,
            "engine": "google_shopping",
            "q": query,
            "google_domain": "google.com",
            "gl": "us",
            "hl": "en",
            "num": max_results
        }
        
        logger.debug(f"Searching with params: {json.dumps(params, default=str)}")
        search = GoogleSearch(params)
        results = search.get_dict()
        
        if "error" in results:
            return json.dumps({"error": results["error"]})
            
        shopping_results = results.get("shopping_results", [])
        
        # Format products
        products = []
        for item in shopping_results:
            product = {
                "name": item.get("title"),
                "price": item.get("price"),
                "description": item.get("snippet") or item.get("description") or "No description available",
                "url": item.get("link"),
                "thumbnail": item.get("thumbnail"),
                "source": item.get("source"),
                "rating": item.get("rating"),
                "reviews": item.get("reviews")
            }
            products.append(product)
            
        # Fallback to regular search if no shopping results found
        if not products and "organic_results" in results:
            logger.info("No shopping results found, falling back to organic results")
            # This might happen if we switch engine to 'google' or if shopping has no results
            # But with engine='google_shopping', we should get shopping_results
            pass
            
        return json.dumps({
            "query": query,
            "products": products[:max_results],
            "count": len(products[:max_results])
        }, indent=2)
        
    except Exception as e:
        logger.error(f"Error in recommend_products: {e}", exc_info=True)
        return json.dumps({"error": str(e)})


@mcp.tool(annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True})
def search_products(query: str, max_results: int = 10) -> str:
    """
    Search for products using standard Google Search (internal tool)
    
    Args:
        query: Product search query
        max_results: Maximum number of results to return (default 10, max 100)
    
    Returns:
        JSON string containing search results
    """
    # Validate query input
    if not isinstance(query, str) or not query.strip():
        return json.dumps({"error": "Query parameter must be a non-empty string."})
    if len(query) > 256:
        return json.dumps({"error": "Query parameter is too long (max 256 characters)."})
    logger.info(f"Searching products for query: '{query}'")
    
    if not SERPAPI_API_KEY:
        return json.dumps({"error": "SERPAPI_API_KEY not configured"})
    
    # Limit max_results
    max_results = min(max_results, 100)
    
    try:
        # Use standard Google Search for broader context
        params = {
            "api_key": SERPAPI_API_KEY,
            "engine": "google",
            "q": query,
            "google_domain": "google.com",
            "gl": "us",
            "hl": "en",
            "num": max_results
        }
        
        search = GoogleSearch(params)
        results = search.get_dict()
        
        if "error" in results:
            return json.dumps({"error": results["error"]})

        return json.dumps(
            {
                "query": query,
                "organic_results": results.get("organic_results", [])[:max_results],
                "shopping_results": results.get("shopping_results", [])[:max_results],
            },
            indent=2,
        )
        
    except Exception as e:
        logger.error(f"Error in search_products: {e}", exc_info=True)
        return json.dumps({"error": str(e)})


def run_server(
    transport: str | None = None,
    host: str | None = None,
    port: int | str | None = None,
    json_response: bool | None = None,
    stateless_http: bool | None = None,
) -> None:
    """Run the MCP server with optional overrides from CLI or environment."""
    if transport is None:
        transport = os.getenv("MCP_TRANSPORT", "http")
    if host is None:
        host = os.getenv("HOST", "0.0.0.0")
    if port is None:
        port = int(os.getenv("PORT", "8000"))
    else:
        port = int(port)
    if json_response is None:
        json_response = _env_flag("MCP_JSON_RESPONSE", "true")
    if stateless_http is None:
        stateless_http = _env_flag("MCP_STATELESS_HTTP", "false")

    logger.info(
        "Starting MCP server transport=%s host=%s port=%s json_response=%s stateless_http=%s",
        transport,
        host,
        port,
        json_response,
        stateless_http,
    )
    mcp.run(
        transport=transport,
        host=host,
        port=port,
        json_response=json_response,
        stateless_http=stateless_http,
    )


# Attach agent card route if FastMCP exposes the FastAPI app
app = getattr(mcp, "app", None)
if app:
    @app.get("/.well-known/agent.json")
    def agent_card() -> Dict[str, Any]:
        return AGENT_CARD


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Shopping Agent MCP Server")
    parser.add_argument(
        "--transport",
        dest="transport",
        default=None,
        help="Transport to use for FastMCP (default: env MCP_TRANSPORT or http)",
    )
    parser.add_argument(
        "--host",
        dest="host",
        default=None,
        help="Host interface to bind (default: env HOST or 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        dest="port",
        type=int,
        default=None,
        help="Port to bind (default: env PORT or 8000)",
    )
    parser.add_argument(
        "--json-response",
        dest="json_response",
        action="store_true",
        help="Force JSON responses (overrides env MCP_JSON_RESPONSE)",
    )
    parser.add_argument(
        "--no-json-response",
        dest="json_response",
        action="store_false",
        help="Disable JSON responses (overrides env MCP_JSON_RESPONSE)",
    )
    parser.add_argument(
        "--stateless-http",
        dest="stateless_http",
        action="store_true",
        help="Enable stateless HTTP transport mode",
    )
    parser.add_argument(
        "--stateful-http",
        dest="stateless_http",
        action="store_false",
        help="Disable stateless HTTP transport mode",
    )
    parser.set_defaults(json_response=None, stateless_http=None)
    return parser.parse_args()


def main() -> int:
    # Configure logging only when run as a script, not on import
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        stream=sys.stdout,
        format="%(levelname)s: %(message)s",
    )

    args = _parse_args()

    if SERPAPI_API_KEY is None:
        logger.error("Please configure the SERPAPI_API_KEY environment variable before running the server")
        return 1
    
    logger.info("Starting Shopping Agent MCP Server with SerpAPI")
    logger.info("Note: This server provides search results. The calling agent provides reasoning.")
    
    run_server(
        transport=args.transport,
        host=args.host,
        port=args.port,
        json_response=args.json_response,
        stateless_http=args.stateless_http,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
