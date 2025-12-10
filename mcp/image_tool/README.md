# Image MCP tool

Small MCP server that returns images from the https://picsum.photos service.

Tools
- `get_image(height, width)`
  - Returns `{"image_base64": "<base64>", "url": "https://picsum.photos/<h>/<w>"}`.


Run locally

```bash
cd mcp/image_tool
# activate the .venv if you use one, then:
python image_tool.py
```

Environment
- `HOST` (default `0.0.0.0`)
- `PORT` (default `8000`)
- `MCP_TRANSPORT` (default `streamable-http`)
- `LOG_LEVEL` (default `INFO`)
