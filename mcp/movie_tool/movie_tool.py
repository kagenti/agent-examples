import os
import json
import requests
import sys
from fastmcp import FastMCP
import logging
from typing import Any

logger = logging.getLogger(__name__)
logging.basicConfig(level=os.getenv("LOG_LEVEL", "DEBUG"), stream=sys.stdout, format='%(levelname)s: %(message)s')
logging.getLogger("urllib3").setLevel(logging.INFO)

OMDB_API_KEY = os.getenv("OMDB_API_KEY")

mcp = FastMCP("Movie Review")

def _fetch_json(params: dict[str, Any], timeout: int = 10) -> dict[str, Any]:
    """
    Helper to perform a GET request and parse the JSON response from the OMDb API.

    Args:
        params (dict[str, Any]): Dictionary of URL parameters (e.g., {"t": "movie title"}).
        timeout (int, optional): Timeout for the request in seconds. Defaults to 10.

    Returns:
        dict[str, Any]: The parsed JSON data on success, or a dictionary containing an "Error" key on failure.
    """
    if OMDB_API_KEY is None:
        return {"Error": "OMDB_API_KEY is not configured"}
    base_url = f"http://www.omdbapi.com/?apikey={OMDB_API_KEY}&"
    try:
        resp = requests.get(base_url, params=params, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.error("Error fetching data: %s", e)
        return {"Error": "Error fetching data"}

@mcp.tool(annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True})
def get_full_plot(movie_title: str) -> str:
    """Get full plot summary of a movie from OMDb API."""
    
    logger.debug("Requesting OMDb with t=%s plot=%s", movie_title, "full")
    params = {"t": movie_title, "plot": "full"}
    data = _fetch_json(params=params)
        
    if "Error" in data:
        return data["Error"]
    
    if "Response" in data and data["Response"] == "True" and "Plot" in data:
        return data["Plot"]
    
    return "Movie not found"

@mcp.tool(annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True})
def get_movie_details(movie_title: str) -> str:
    """Get full details (awards, actors, short plot, and ratings, etc.) of a movie from OMDb API."""

    logger.debug("Requesting OMDb with t=%s plot=%s", movie_title, "short")
    params = {"t": movie_title, "plot": "short"}
    data = _fetch_json(params=params)
    
    if "Error" in data:
        return data["Error"]

    if "Response" in data and data["Response"] == "True":
        data.pop("Poster", None)
        del data["Response"]
        return json.dumps(data)

    return "Movie not found"

# host can be specified with HOST env variable
# transport can be specified with MCP_TRANSPORT env variable (defaults to streamable-http)
def run_server():
    "Run the MCP server"
    transport = os.getenv("MCP_TRANSPORT", "streamable-http")
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    mcp.run(transport=transport, host=host, port=port)

if __name__ == "__main__":
    if OMDB_API_KEY is None:
        logger.warning("Please configure the OMDB_API_KEY environment variable before running the server")
    run_server()
