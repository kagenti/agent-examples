# MCP Weather tool

This tool demonstrates a small MCP server.  The server implements a `get_weather` tool that returns the current weather for a city using https://open-meteo.com/en/docs/geocoding-api .

## Test the MCP server locally

Run locally

```bash
cd mcp/weather_tool
uv run --no-sync weather_tool.py
```

## Deploy the MCP server to Kagenti

### Deploy using the Kagenti UI

- Browse to http://kagenti-ui.localtest.me:8080/tools
- Import Tool
- Deploy from Source
  - Select weather tool

### Deploy using a Kubernetes deployment descriptor

Alternately, you can deploy a pre-built image using Kubernetes

- `kubectl apply -f mcp/weather_tool/deployment/k8s.yaml`

## Test the MCP server using Kagenti

- Visit http://kagenti-ui.localtest.me:8080/tools/team1/weather-tool
- Click "Connect & list tools"
- Expand "get_weather"
- Click "invoke tool"
- Enter the name of a city
- Click "invoke"
