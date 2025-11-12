# Shopping Agent MCP Tool

A Model Context Protocol (MCP) server that provides intelligent shopping recommendations using LangChain, LangGraph, OpenAI, and SerpAPI.

## Features

- **AI-Powered Recommendations**: Uses OpenAI's GPT-4 to understand natural language queries and generate personalized product recommendations
- **Real-time Search**: Leverages SerpAPI to search across multiple retailers for the best products
- **LangGraph Agent**: Implements a sophisticated multi-step agent workflow with:
  - Query parsing and understanding
  - Product search across retailers
  - Recommendation generation with reasoning
- **Configurable Results**: Limit recommendations (default 5, max 20) based on your needs

## Tools

### 1. `recommend_products`

Recommends products based on a natural language query with budget and preferences.

**Parameters:**
- `query` (string, required): Natural language product request (e.g., "I want to buy a scarf for 40 dollars")
- `maxResults` (integer, optional): Maximum number of recommendations (default: 5, max: 20)

**Returns:**
```json
{
  "query": "I want to buy a scarf for 40 dollars",
  "recommendations": [
    {
      "name": "Cashmere Blend Scarf",
      "price": "$35.99",
      "description": "Soft and warm cashmere blend scarf in multiple colors",
      "url": "https://example.com/product",
      "reason": "High quality within budget with excellent reviews"
    }
  ],
  "count": 5
}
```

### 2. `search_products`

Search for products across retailers (lower-level tool for raw search results).

**Parameters:**
- `query` (string, required): Product search query
- `maxResults` (integer, optional): Maximum results to return (default: 10, max: 100)

**Returns:**
Raw search results from SerpAPI.

## Setup

### Prerequisites

- Python 3.10 or higher
- [uv](https://docs.astral.sh/uv/) package manager
- OpenAI API key
- SerpAPI key

### Installation

1. **Get API Keys:**
   - OpenAI API key: https://platform.openai.com/api-keys
   - SerpAPI key: https://serpapi.com/manage-api-key

2. **Install Dependencies:**

```bash
cd mcp/shopping_agent
uv pip install -e .
```

### Configuration

Set the required environment variables:

```bash
export OPENAI_API_KEY="your-openai-api-key"
export SERPAPI_API_KEY="your-serpapi-key"
```

Optional configuration:
```bash
export HOST="0.0.0.0"                    # Server host (default: 0.0.0.0)
export PORT="8000"                        # Server port (default: 8000)
export MCP_TRANSPORT="http"               # Transport type (default: http, Inspector-ready)
export MCP_JSON_RESPONSE="1"              # Force JSON responses (default: enabled)
export LOG_LEVEL="INFO"                   # Logging level (default: INFO)
```

## Running the Server

### Development Mode

```bash
cd mcp/shopping_agent
export OPENAI_API_KEY="your-key"
export SERPAPI_API_KEY="your-key"
python shopping_agent.py
```

The server will start on `http://0.0.0.0:8000` by default.

### Command-line options

You can override server behaviour with CLI flags:

```bash
uv run shopping_agent.py --json-response --port 8020
```

- `--json-response` / `--no-json-response`: toggle JSON responses without touching `MCP_JSON_RESPONSE`
- `--stateless-http` / `--stateful-http`: control FastMCP stateless HTTP mode
- `--host`, `--port`, `--transport`: override bind settings (fall back to environment variables when omitted)

### MCP Inspector Demo (HTTP Transport)

Follow these steps to debug the shopping agent with the official MCP Inspector UI:

1. Start the server on its own port using HTTP transport:
   ```bash
   cd mcp/shopping_agent
   export OPENAI_API_KEY="your-key"
   export SERPAPI_API_KEY="your-key"
   MCP_TRANSPORT=http PORT=8001 python shopping_agent.py
   ```
2. In a new terminal (requires Node.js ≥18), launch the inspector:
   ```bash
   npx @modelcontextprotocol/inspector
   ```
3. In the Inspector UI choose **Add server**, then supply:
   - Name: `Shopping Agent (HTTP)`
   - Transport: `HTTP` (or `Streamable HTTP` on older Inspector releases)
   - URL: `http://localhost:8001`
4. Click **Connect**, open the **Tools** tab, and invoke `recommend_products` or `search_products`. Responses stream in the right-hand panel.

Tip: run the `movie_tool` server on a different port (for example `PORT=8002 MCP_TRANSPORT=http python ../movie_tool/movie_tool.py`) to compare both MCP servers side by side inside the inspector.

### Using Docker

```bash
cd mcp/shopping_agent

# Build the image
docker build -t shopping-agent-mcp .

# Run the container
docker run -p 8000:8000 \
  -e OPENAI_API_KEY="your-openai-api-key" \
  -e SERPAPI_API_KEY="your-serpapi-key" \
  shopping-agent-mcp
```

## Architecture

The shopping agent uses LangGraph to implement a multi-step workflow:

```
User Query → Parse Query → Search Products → Generate Recommendations → Return Results
```

### LangGraph Workflow

1. **Parse Query Node**: Uses OpenAI to extract product type and budget from natural language
2. **Search Products Node**: Queries SerpAPI for relevant products across retailers
3. **Generate Recommendations Node**: Uses OpenAI to analyze results and create personalized recommendations

### Technologies Used

- **FastMCP**: MCP server framework
- **LangChain**: LLM application framework
- **LangGraph**: Agent workflow orchestration
- **OpenAI GPT-4**: Natural language understanding and generation
- **SerpAPI**: Real-time product search across retailers

## Usage Examples

### Example 1: Basic Product Search

```python
# Query
"I want to buy a scarf for 40 dollars"

# Response
{
  "recommendations": [
    {
      "name": "Winter Wool Scarf",
      "price": "$38.99",
      "description": "100% merino wool, various colors",
      "reason": "High quality, within budget, great reviews"
    },
    // ... 4 more recommendations
  ]
}
```

### Example 2: Specific Requirements

```python
# Query
"Find me wireless headphones under $100 with good noise cancellation"

# Response includes 5 curated recommendations with:
# - Product names
# - Prices
# - Detailed descriptions
# - Reasons for recommendation
# - Purchase links
```

## Testing

You can test the MCP server tools using curl:

```bash
# Test recommend_products tool
curl -X POST http://localhost:8000/mcp/tools/recommend_products \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{
    "query": "I want to buy a scarf for 40 dollars",
    "maxResults": 5
  }'
```

## Troubleshooting

### API Key Issues

If you see "API key not configured" errors:
1. Verify your API keys are set correctly
2. Check that environment variables are exported in the same shell session
3. Restart the server after setting environment variables

### No Results Returned

If searches return no results:
1. Try a more specific query with product name and budget
2. Check your SerpAPI quota at https://serpapi.com/dashboard
3. Review server logs for detailed error messages

### Import Errors

If you encounter import errors:
1. Ensure all dependencies are installed: `uv pip install -e .`
2. Check Python version is 3.10 or higher
3. Try reinstalling with `uv pip install --force-reinstall -e .`

## Development

### Project Structure

```
shopping_agent/
├── shopping_agent.py    # Main MCP server with LangGraph agent
├── pyproject.toml       # Dependencies and project metadata
├── README.md            # This file
├── Dockerfile           # Container configuration
└── __init__.py          # Package initialization
```

### Contributing

When contributing, ensure:
1. Code follows the existing style
2. All API keys are handled via environment variables
3. Error handling is comprehensive
4. Logging is informative but not excessive
5. Tests pass (if applicable)

## License

See the repository's LICENSE file for details.

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review server logs for detailed error messages
3. Ensure all API keys are valid and have sufficient quota
4. Open an issue in the repository with relevant logs

