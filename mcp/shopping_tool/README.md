# Shopping Agent MCP Tool

A Model Context Protocol (MCP) server that provides product search capabilities using SerpAPI. The server returns structured product data, and the calling agent provides intelligent reasoning and recommendations.

## Features

- **Product Search**: Leverages SerpAPI (Google Shopping) to search across multiple retailers
- **Structured Data**: Returns rich product info including titles, prices, descriptions, thumbnails, ratings, and reviews
- **Agent-Driven Analysis**: The MCP server returns raw data; your AI agent analyzes and provides reasoning
- **Budget Awareness**: Product data includes prices for the agent to consider constraints
- **Configurable Results**: Limit search results (default 10, max 20) based on your needs

## Tools

### 1. `recommend_products`

Searches for products based on a natural language query and returns structured product data. The calling agent should analyze this data and provide intelligent reasoning.

**Parameters:**
- `query` (string, required): Natural language product request (e.g., "I want to buy a scarf for 40 dollars")
- `max_results` (integer, optional): Maximum number of results (default: 10, max: 20)

**Returns:**
```json
{
  "query": "I want to buy a scarf for 40 dollars",
  "products": [
    {
      "name": "Cashmere Blend Scarf",
      "price": "$35.99",
      "description": "Soft and warm cashmere blend scarf in multiple colors...",
      "url": "https://example.com/product",
      "thumbnail": "https://example.com/image.jpg",
      "source": "Amazon",
      "rating": 4.5,
      "reviews": 120
    }
  ],
  "count": 5
}
```

**Features:**
- Returns rich structured product data including thumbnails and ratings
- The calling agent should analyze products and provide reasoning
- Products include names, prices, descriptions, and purchase URLs
- Only requires `SERPAPI_API_KEY` to function

### 2. `search_products`

Search for products across retailers (lower-level tool for raw search results).

**Parameters:**
- `query` (string, required): Product search query
- `max_results` (integer, optional): Maximum results to return (default: 10, max: 100)

**Returns:**
Raw search results from SerpAPI.

## Setup

### Prerequisites

- Python 3.10 or higher
- [uv](https://docs.astral.sh/uv/) package manager
- SerpAPI key (required for product search)

### Installation

1. **Get API Key:**
   - SerpAPI key: https://serpapi.com/manage-api-key

2. **Install Dependencies:**

```bash
cd mcp/shopping_tool
uv pip install -e .
```

### Configuration

Set the required environment variable:

```bash
export SERPAPI_API_KEY="your-serpapi-key"
```

Optional configuration:
```bash
export HOST="0.0.0.0"                     # Server host (default: 0.0.0.0)
export PORT="8000"                        # Server port (default: 8000)
export MCP_TRANSPORT="http"               # Transport type (default: http, Inspector-ready)
export MCP_JSON_RESPONSE="1"              # Force JSON responses (default: enabled)
export LOG_LEVEL="INFO"                   # Logging level (default: INFO)
```

## Running the Server

### Development Mode

```bash
cd mcp/shopping_tool
export SERPAPI_API_KEY="your-serpapi-key"
python shopping_agent.py
```

The server will start on `http://0.0.0.0:8000` by default and will return structured product data for your agent to analyze.

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
   cd mcp/shopping_tool
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
cd mcp/shopping_tool

# Build the image
docker build -t shopping-agent-mcp .

# Run the container
docker run -p 8000:8000 \
  -e SERPAPI_API_KEY="your-serpapi-key" \
  shopping-agent-mcp
```

## Architecture

The shopping agent MCP server provides product search data. The calling agent (your AI) provides reasoning and recommendations:

```
User Query → [MCP Tool] Search Products → Return Structured Data → [Your Agent] Analyze & Recommend
```

### Workflow

1. **MCP Server**: Receives query, searches SerpAPI, returns structured product data (no reasoning)
2. **Your Agent** (e.g., Claude, GPT-4): Analyzes the product data and provides intelligent reasoning for recommendations

### Suggested Agent System Prompt

When using this MCP server, configure your agent with a system prompt like this:

```
When users ask for product recommendations:
1. Use the recommend_products tool to search for products
2. Analyze each product considering:
   - How well it matches the user's requirements
   - Whether it fits within their budget
   - The value proposition (quality vs. price)
   - Any specific features mentioned in the query
3. For each product, provide:
   - A brief explanation of why it's a good match
   - A recommendation score (1-10)
   - Any caveats or considerations
4. Rank products by how well they match the user's needs
5. Present the top 3-5 recommendations with your reasoning
```

### Technologies Used

- **FastMCP**: MCP server framework
- **LangChain Community**: SerpAPI wrapper for product search
- **SerpAPI**: Real-time product search across retailers

## Usage Examples

### Example 1: Basic Product Search

```python
# Query
"I want to buy a scarf for 40 dollars"

# MCP Server Response (raw product data)
{
  "query": "I want to buy a scarf for 40 dollars",
  "products": [
    {
      "name": "Winter Wool Scarf",
      "price": "$38.99",
      "description": "100% merino wool...",
      "url": "https://example.com/product",
      "thumbnail": "https://...",
      "source": "Retailer A"
    },
    {
      "name": "Cashmere Blend Scarf",
      "price": "$35.99",
      "description": "Soft cashmere blend...",
      "url": "https://example.com/product2",
      "thumbnail": "https://...",
      "source": "Retailer B"
    }
  ],
  "count": 2
}

# Your Agent should then analyze these products and provide reasoning:
# "Here are my top recommendations:
# 
# 1. Winter Wool Scarf ($38.99) - Score: 9/10
#    This is an excellent choice within your $40 budget. The 100% merino wool 
#    provides superior warmth and quality, making it great value at this price point.
# 
# 2. Cashmere Blend Scarf ($35.99) - Score: 8/10
#    A more affordable option that still offers luxury with its cashmere blend..."
```

### Example 2: Specific Requirements

```python
# Query
"Find me wireless headphones under $100 with good noise cancellation"

# MCP Server Response (raw product data)
{
  "query": "wireless headphones under $100 with good noise cancellation",
  "products": [
    {
      "name": "Sony WH-CH710N Wireless Headphones",
      "price": "$89.99",
      "description": "Active noise canceling...",
      "url": "https://example.com/sony-headphones",
      "rating": 4.4,
      "reviews": 8500
    }
  ],
  "count": 1
}

# Your Agent analyzes and recommends:
# "I found the Sony WH-CH710N Wireless Headphones at $89.99 - this is an excellent 
# match for your requirements:
# - ✅ Wireless Bluetooth connectivity
# - ✅ Active noise cancellation feature
# - ✅ Well under your $100 budget ($89.99)
# - ✅ Bonus: 35-hour battery life
# 
# Recommendation Score: 9/10
# This product checks all your boxes and comes from Sony, a trusted brand in audio..."
```


### Using curl

You can also test the MCP server tools using curl:

```bash
# Test recommend_products tool (returns structured product data)
curl -X POST http://localhost:8000/mcp/tools/recommend_products \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{
    "query": "I want to buy a scarf for 40 dollars",
    "max_results": 5
  }'
```

## Troubleshooting

### API Key Issues

**SerpAPI Key Issues:**
If you see "SERPAPI_API_KEY not configured" errors:
1. Get your key from https://serpapi.com/manage-api-key
2. Export it: `export SERPAPI_API_KEY="your-key"`
3. Restart the server

**General API Key Tips:**
1. Verify your API key is set correctly
2. Check that the environment variable is exported in the same shell session
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
shopping_tool/
├── shopping_agent.py       # Main MCP server with SerpAPI integration
├── simple_test.py          # Test script for product search
├── pyproject.toml          # Dependencies and project metadata
├── README.md               # This file
├── Dockerfile              # Container configuration
└── __init__.py             # Package initialization
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

