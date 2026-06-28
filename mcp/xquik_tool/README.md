# MCP Xquik tool

This tool demonstrates a small MCP server that queries public X data through Xquik. It implements read-only tools for post search, user search, and trends.

## Tools

- `search_x_posts(query, query_type, limit, since_time, until_time)` searches public X posts.
- `search_x_users(query, cursor)` searches public X users.
- `get_x_trends(woeid, count)` fetches public X trends.

## Requirements

- `XQUIK_API_KEY` environment variable

## Test the MCP server locally

Run locally:

```bash
cd mcp/xquik_tool
export XQUIK_API_KEY="your-key"
uv run --no-sync xquik_tool.py
```

## Deploy the MCP server to Kagenti

### Deploy using the Kagenti UI

- Browse to http://kagenti-ui.localtest.me:8080/tools
- Import Tool
- Deploy from Source
  - Select xquik tool

### Deploy using Docker

```bash
cd mcp/xquik_tool
docker build -t xquik-mcp-tool .
docker run -p 8000:8000 -e XQUIK_API_KEY="$XQUIK_API_KEY" xquik-mcp-tool
```

## Notes

- This tool only reads public X data.
- Keep `XQUIK_API_KEY` in the environment. Do not write it into source files, prompts, logs, or Docker images.
- Use narrow queries and cite returned post URLs, handles, timestamps, and query terms when summarizing results.
