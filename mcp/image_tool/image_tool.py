# Image MCP tool - returns images from picsum.photos.

import base64
import logging
import os
import requests
import sys
from fastmcp import FastMCP

mcp = FastMCP("Image")
logger = logging.getLogger(__name__)
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), stream=sys.stdout, format='%(levelname)s: %(message)s')


@mcp.tool(annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True})
def get_image(height: int, width: int) -> dict:
    """Fetch a random image from picsum.photos API and return it as base64-encoded data.

    Parameters:
    - height: image height in pixels (must be positive integer)
    - width: image width in pixels (must be positive integer)

    Returns a dict containing:
    - image_base64: base64-encoded image data (string)
    - url: the source URL of the image (string)
    
    Example return value:
    {"image_base64": "/9j/4AAQSkZJRg...", "url": "https://picsum.photos/200/300"}
    """
    try:
        h = int(height)
        w = int(width)
        if h <= 0 or w <= 0:
            return {"error": "height and width must be positive integers"}
    except (ValueError, TypeError):
        return {"error": "height and width must be integers"}
    url = f"https://picsum.photos/{w}/{h}"

    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        img_b = resp.content
        img_b64 = base64.b64encode(img_b).decode("ascii")
        logger.info(f"Successfully fetched and encoded {h}x{w} image, base64 length={len(img_b64)}")
        return {"image_base64": img_b64, "url": url}
    except requests.RequestException as e:
        logger.error("failed to fetch image: %s", e)
        return {"error": str(e), "url": url}


# host can be specified with HOST env variable
# transport can be specified with MCP_TRANSPORT env variable (defaults to streamable-http)
def run_server():
    """Run the MCP server"""
    transport = os.getenv("MCP_TRANSPORT", "streamable-http")
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    mcp.run(transport=transport, host=host, port=port)


if __name__ == "__main__":
    run_server()
