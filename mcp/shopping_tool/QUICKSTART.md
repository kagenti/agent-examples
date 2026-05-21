# Shopping Agent - Quick Start Guide

This guide will help you get the Shopping Agent MCP server up and running quickly.

## What You'll Need

1. **SerpAPI Key** - Get it from [SerpAPI Dashboard](https://serpapi.com/manage-api-key)
2. **Python 3.10+** - Check with `python --version`
3. **uv package manager** - Install from [Astral UV](https://docs.astral.sh/uv/)

## Installation Steps

### Step 1: Set Up API Key

```bash
# Export your SerpAPI key
export SERPAPI_API_KEY="your-serpapi-key-here"
```

**Tip**: Add this to your `~/.bashrc` or `~/.zshrc` to persist it:
```bash
echo 'export SERPAPI_API_KEY="your-key"' >> ~/.zshrc
source ~/.zshrc
```

### Step 2: Install Dependencies

```bash
cd mcp/shopping_tool
uv pip install -e .
```

### Step 3: Start the Server

```bash
python shopping_agent.py
```

You should see:
```
INFO: Starting Shopping Agent MCP Server with SerpAPI
INFO: Note: This server provides search results. The calling agent provides reasoning.
INFO: Server running on http://0.0.0.0:8000
```

### Step 4: Test the Server

In a new terminal:

```bash
# Test with the provided test script
python simple_test.py

# Or test manually with curl
curl -X POST http://localhost:8000/mcp/tools/recommend_products \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{
    "query": "I want to buy a scarf for 40 dollars",
    "max_results": 5
  }'
```

## MCP Inspector Demo (HTTP Transport)

Use the MCP Inspector UI to explore the server interactively:

1. Start the shopping agent with explicit port/transport:
   ```bash
   cd mcp/shopping_tool
   export SERPAPI_API_KEY="your-key"
   MCP_TRANSPORT=http PORT=8001 python shopping_agent.py
   ```
2. In a separate terminal (Node.js ≥18 required) launch the inspector:
   ```bash
   npx @modelcontextprotocol/inspector
   ```
3. When the browser opens, choose **Add server** and fill in:
   - Name: `Shopping Agent`
   - Transport: `HTTP` (use `Streamable HTTP` if that is the option offered)
   - URL: `http://localhost:8001`
4. Connect and explore the `recommend_products` and `search_products` tools from the **Tools** tab. The response JSON renders in the inspector panel.

To compare behavior with the movie MCP server, repeat the steps with `PORT=8002 MCP_TRANSPORT=http python ../movie_tool/movie_tool.py` and add it as a second server in the inspector.

## Usage Examples

### Example 1: Shopping for Scarves

```bash
curl -X POST http://localhost:8000/mcp/tools/recommend_products \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{
    "query": "I want to buy a scarf for 40 dollars",
    "max_results": 5
  }'
```

**Expected Response** (MCP server returns structured product data):
```json
{
  "query": "I want to buy a scarf for 40 dollars",
  "products": [
    {
      "name": "Winter Wool Scarf",
      "price": "$38.99",
      "description": "Soft merino wool scarf in multiple colors...",
      "url": "https://example.com/product1",
      "thumbnail": "https://example.com/image.jpg",
      "source": "Amazon",
      "rating": 4.5,
      "reviews": 120
    },
    {
      "name": "Cashmere Blend Scarf",
      "price": "$35.99",
      "description": "Luxury cashmere blend, various patterns...",
      "url": "https://example.com/product2",
      "thumbnail": "https://example.com/image2.jpg",
      "source": "Etsy"
    }
  ],
  "count": 2
}
```

**Your AI agent should then analyze these products and add reasoning.**

### Example 2: Finding Headphones

```bash
curl -X POST http://localhost:8000/mcp/tools/recommend_products \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  -d '{
    "query": "wireless headphones under $100 with noise cancellation",
    "max_results": 5
  }'
```

### Example 3: Using Python Client

```python
import requests
import json

response = requests.post(
    "http://localhost:8000/mcp/tools/recommend_products",
    headers={
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream"
    },
    json={
        "query": "best laptop under $800 for programming",
        "max_results": 5
    }
)

products = response.json()
print(json.dumps(products, indent=2))

# Your AI agent would then analyze these products and provide recommendations
```

## Architecture Overview

The Shopping Agent uses a simple, efficient architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                      User Query                              │
│         "I want to buy a scarf for $40"                     │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│                 AI Agent (Your Client)                       │
│         Calls recommend_products MCP tool                   │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│           Shopping Agent MCP Server                          │
│          Search Products (SerpAPI)                          │
│          Parse Results                                      │
│          Return Structured Data                             │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│                 AI Agent (Your Client)                       │
│         Analyzes products and provides reasoning            │
│         • Evaluates match to requirements                   │
│         • Considers budget constraints                      │
│         • Ranks by suitability                              │
│         • Explains reasoning                                │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│              Return to User                                  │
│  Recommendations with names, prices, links, and reasoning   │
└─────────────────────────────────────────────────────────────┘
```

## Technologies Used

| Technology | Purpose |
|------------|---------|
| **FastMCP** | MCP server framework for tool exposure |
| **LangChain Community** | SerpAPI wrapper utilities |
| **SerpAPI** | Real-time product search across retailers |

## Configuring Your AI Agent

When using this MCP server, configure your AI agent with a system prompt like:

```
When users ask for product recommendations:

1. Use the recommend_products tool to search for products
2. Analyze each product considering:
   - How well it matches the user's requirements
   - Whether it fits within their budget
   - The value proposition (quality vs. price)
   - Any specific features mentioned

3. For each product, provide:
   - A brief explanation of why it's a good match
   - A recommendation score (1-10)
   - Any caveats or considerations

4. Rank products by how well they match the user's needs
5. Present the top 3-5 recommendations with your reasoning
```

## Example Agent Conversation

**User**: "I want to buy a scarf for 40 dollars"

**AI Agent**: *[Calls recommend_products MCP tool]*

**MCP Server**: *[Returns structured product data]*

**AI Agent Response to User**:
```
I found 2 great scarves within your $40 budget:

1. Winter Wool Scarf - $38.99 ⭐ Score: 9/10
   This is an excellent choice within your budget. The 100% merino wool 
   provides superior warmth and quality, making it great value at this 
   price point. Available in multiple colors.

2. Cashmere Blend Scarf - $35.99 ⭐ Score: 8/10
   A more affordable luxury option. The cashmere blend provides softness 
   and warmth at a lower price. The various patterns give you style 
   flexibility.

My top recommendation is the Winter Wool Scarf for its superior quality 
and durability.
```

## Troubleshooting

### Server Won't Start

**Problem**: Server fails to start with API key errors

**Solution**: 
```bash
# Verify your key is set
echo $SERPAPI_API_KEY

# If empty, export it again
export SERPAPI_API_KEY="your-key"
```

### Import Errors

**Problem**: `ModuleNotFoundError` when starting

**Solution**:
```bash
# Reinstall dependencies
uv pip install --force-reinstall -e .
```

### No Results Returned

**Problem**: Server runs but returns no products

**Solution**:
1. Check your SerpAPI quota at https://serpapi.com/dashboard
2. Verify the query is specific enough
3. Check server logs: `LOG_LEVEL=DEBUG python shopping_agent.py`

### Connection Refused

**Problem**: `Connection refused` when testing

**Solution**:
```bash
# Check if server is running
ps aux | grep shopping_agent

# If not, start the server
python shopping_agent.py
```

## Docker Deployment

### Build and Run

```bash
# Build the Docker image
docker build -t shopping-agent-mcp .

# Run with API key
docker run -p 8000:8000 \
  -e SERPAPI_API_KEY="your-serpapi-key" \
  shopping-agent-mcp
```

### Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'
services:
  shopping-agent:
    build: .
    ports:
      - "8000:8000"
    environment:
      - SERPAPI_API_KEY=${SERPAPI_API_KEY}
      - LOG_LEVEL=INFO
```

Run with:
```bash
docker-compose up
```

## API Reference

### Tool: `recommend_products`

**Description**: Search for products and return structured data for your AI agent to analyze

**Request**:
```json
{
  "query": "string (required) - Natural language product request",
  "max_results": "integer (optional) - Max results (default: 10, max: 20)"
}
```

**Response**:
```json
{
  "query": "string - Original query",
  "products": [
    {
      "name": "string - Product name",
      "price": "string - Price",
      "description": "string - Product description",
      "url": "string - Purchase link"
    }
  ],
  "count": "integer - Number of products",
  "note": "string - Reminder that agent should provide reasoning"
}
```

### Tool: `search_products`

**Description**: Raw product search (lower-level tool)

**Request**:
```json
{
  "query": "string (required) - Search query",
  "max_results": "integer (optional) - Max results (default: 10, max: 100)"
}
```

**Response**:
```json
{
  "query": "string - Search query",
  "results": "string - Raw search results",
  "note": "string - Usage note"
}
```

## Next Steps

1. **Integrate with Your Agent**: Connect your AI agent (Claude, GPT-4, etc.) to use this MCP server
2. **Configure System Prompt**: Use the suggested prompt above to guide your agent's analysis
3. **Test**: Try various product queries and refine your agent's reasoning
4. **Monitor**: Add logging and monitoring for production use
5. **Scale**: Deploy with Docker and load balancing for high traffic

## Support

- Check logs with `LOG_LEVEL=DEBUG python shopping_agent.py`
- Review the [README.md](README.md) for detailed documentation
- Review the [ARCHITECTURE.md](ARCHITECTURE.md) for architecture details
- Verify API key has sufficient quota at https://serpapi.com/dashboard
- Test with the provided `simple_test.py` script

## Summary

✅ You've created a lightweight Shopping Agent MCP server  
✅ It uses SerpAPI for real-time product search  
✅ It returns structured data for your AI agent to analyze  
✅ It follows MCP best practices with clear separation of concerns  
✅ It's ready for production deployment with Docker  

Your AI agent does the smart analysis and recommendations! 🛍️
