# Shopping Agent - Quick Start Guide

This guide will help you get the Shopping Agent MCP server up and running quickly.

## What You'll Need

1. **OpenAI API Key** - Get it from [OpenAI Platform](https://platform.openai.com/api-keys)
2. **SerpAPI Key** - Get it from [SerpAPI Dashboard](https://serpapi.com/manage-api-key)
3. **Python 3.10+** - Check with `python --version`
4. **uv package manager** - Install from [Astral UV](https://docs.astral.sh/uv/)

## Installation Steps

### Step 1: Set Up API Keys

```bash
# Export your API keys
export OPENAI_API_KEY="sk-your-openai-key-here"
export SERPAPI_API_KEY="your-serpapi-key-here"
```

**Tip**: Add these to your `~/.bashrc` or `~/.zshrc` to persist them:
```bash
echo 'export OPENAI_API_KEY="sk-your-key"' >> ~/.zshrc
echo 'export SERPAPI_API_KEY="your-key"' >> ~/.zshrc
source ~/.zshrc
```

### Step 2: Install Dependencies

```bash
cd mcp/shopping_agent
uv pip install -e .
```

### Step 3: Start the Server

```bash
python shopping_agent.py
```

You should see:
```
INFO: Starting Shopping Agent MCP Server with LangChain and LangGraph
INFO: Server running on http://0.0.0.0:8000
```

### Step 4: Test the Server

In a new terminal:

```bash
# Test with the provided test client
python test_client.py

# Or test manually with curl
curl -X POST http://localhost:8000/tools/recommend_products \
  -H "Content-Type: application/json" \
  -d '{
    "query": "I want to buy a scarf for 40 dollars",
    "maxResults": 5
  }'
```

## MCP Inspector Demo (HTTP Transport)

Use the MCP Inspector UI to explore the server without writing client code:

1. Start the shopping agent with explicit port/transport so it is easy to find:
   ```bash
   cd mcp/shopping_agent
   export OPENAI_API_KEY="your-key"
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

To compare behaviour with the movie MCP server, repeat the steps with `PORT=8002 MCP_TRANSPORT=http python ../movie_tool/movie_tool.py` and add it as a second server in the inspector.

## Usage Examples

### Example 1: Shopping for Scarves

```bash
curl -X POST http://localhost:8000/tools/recommend_products \
  -H "Content-Type: application/json" \
  -d '{
    "query": "I want to buy a scarf for 40 dollars. Recommend me some options.",
    "maxResults": 5
  }'
```

**Expected Response:**
```json
{
  "query": "I want to buy a scarf for 40 dollars. Recommend me some options.",
  "recommendations": [
    {
      "name": "Winter Wool Scarf",
      "price": "$38.99",
      "description": "Soft merino wool scarf in multiple colors",
      "url": "https://...",
      "reason": "High quality within budget with excellent reviews"
    },
    // ... 4 more recommendations
  ],
  "count": 5
}
```

### Example 2: Finding Headphones

```bash
curl -X POST http://localhost:8000/tools/recommend_products \
  -H "Content-Type: application/json" \
  -d '{
    "query": "wireless headphones under $100 with noise cancellation",
    "maxResults": 5
  }'
```

### Example 3: Using Python Client

```python
import requests
import json

response = requests.post(
    "http://localhost:8000/tools/recommend_products",
    json={
        "query": "best laptop under $800 for programming",
        "maxResults": 5
    }
)

recommendations = response.json()
print(json.dumps(recommendations, indent=2))
```

## Architecture Overview

The Shopping Agent uses a sophisticated LangGraph workflow:

```
┌─────────────────────────────────────────────────────────────┐
│                    User Query                                │
│         "I want to buy a scarf for $40"                     │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│              Parse Query Node (OpenAI)                       │
│  Extracts: product="scarf", budget="40"                     │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│          Search Products Node (SerpAPI)                      │
│  Searches: "scarf $40 buy online shop"                      │
│  Returns: Raw search results from retailers                 │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│      Generate Recommendations Node (OpenAI)                  │
│  Analyzes results and creates 5 personalized                │
│  recommendations with reasoning                             │
└─────────────────┬───────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────┐
│              Return Recommendations                          │
│  JSON with names, prices, descriptions, links, reasons      │
└─────────────────────────────────────────────────────────────┘
```

## Technologies Used

| Technology | Purpose |
|------------|---------|
| **FastMCP** | MCP server framework for tool exposure |
| **LangChain** | LLM application framework and utilities |
| **LangGraph** | Agent workflow orchestration with state management |
| **OpenAI GPT-4o-mini** | Natural language understanding and generation |
| **SerpAPI** | Real-time product search across retailers |

## Key Features

### ✅ MCP Server Implementation
- Uses FastMCP framework following the pattern of existing MCP tools
- Exposes two tools: `recommend_products` and `search_products`
- Follows MCP best practices with proper annotations

### ✅ LangChain Integration
- Uses `ChatOpenAI` for LLM operations
- Implements `SerpAPIWrapper` for search
- Uses LangChain message types (HumanMessage, AIMessage, SystemMessage)

### ✅ LangGraph Agent
- Implements a multi-node state graph
- Three nodes: parse_query, search_products, generate_recommendations
- Proper state management with `AgentState` TypedDict
- Sequential workflow with clear edge definitions

### ✅ OpenAI API Usage
- Query parsing to extract product and budget
- Recommendation generation with reasoning
- Uses GPT-4o-mini for cost-effective performance

### ✅ SerpAPI Integration
- Real-time product search across multiple retailers
- Optimized search queries for shopping results
- Error handling for API failures

## Troubleshooting

### Server Won't Start

**Problem**: Server fails to start with API key errors

**Solution**: 
```bash
# Verify your keys are set
echo $OPENAI_API_KEY
echo $SERPAPI_API_KEY

# If empty, export them again
export OPENAI_API_KEY="your-key"
export SERPAPI_API_KEY="your-key"
```

### Import Errors

**Problem**: `ModuleNotFoundError` when starting

**Solution**:
```bash
# Reinstall dependencies
uv pip install --force-reinstall -e .

# Or install individually
uv pip install fastmcp langchain langchain-openai langchain-community langgraph openai google-search-results
```

### No Recommendations Returned

**Problem**: Server runs but returns empty recommendations

**Solution**:
1. Check your SerpAPI quota at https://serpapi.com/dashboard
2. Verify the query is specific enough
3. Check server logs for detailed errors: `LOG_LEVEL=DEBUG python shopping_agent.py`

### Connection Refused

**Problem**: `Connection refused` when testing

**Solution**:
```bash
# Check if server is running
curl http://localhost:8000/health

# If not, start the server
python shopping_agent.py
```

## Docker Deployment

### Build and Run

```bash
# Build the Docker image
docker build -t shopping-agent-mcp .

# Run with API keys
docker run -p 8000:8000 \
  -e OPENAI_API_KEY="your-openai-key" \
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
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - SERPAPI_API_KEY=${SERPAPI_API_KEY}
      - LOG_LEVEL=INFO
```

Run with:
```bash
docker-compose up
```

## API Reference

### Tool: `recommend_products`

**Description**: Get AI-powered product recommendations based on natural language query

**Request**:
```json
{
  "query": "string (required) - Natural language product request",
  "maxResults": "integer (optional) - Max recommendations (default: 5, max: 20)"
}
```

**Response**:
```json
{
  "query": "string - Original query",
  "recommendations": [
    {
      "name": "string - Product name",
      "price": "string - Price",
      "description": "string - Product description",
      "url": "string - Purchase link",
      "reason": "string - Why recommended"
    }
  ],
  "count": "integer - Number of recommendations"
}
```

### Tool: `search_products`

**Description**: Raw product search (lower-level tool)

**Request**:
```json
{
  "query": "string (required) - Search query",
  "maxResults": "integer (optional) - Max results (default: 10, max: 100)"
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

1. **Customize**: Modify `shopping_agent.py` to adjust the agent's behavior
2. **Integrate**: Connect the MCP server to your AI application
3. **Monitor**: Add logging and monitoring for production use
4. **Scale**: Deploy with Docker and load balancing for high traffic
5. **Enhance**: Add more tools like price tracking, review analysis, etc.

## Support

- Check logs with `LOG_LEVEL=DEBUG python shopping_agent.py`
- Review the [README.md](README.md) for detailed documentation
- Verify API keys have sufficient quota
- Test with the provided `test_client.py` script

## Summary

✅ You've created a fully functional Shopping Agent MCP server  
✅ It uses LangChain and LangGraph for intelligent agent workflows  
✅ It integrates OpenAI and SerpAPI for smart recommendations  
✅ It follows MCP best practices and patterns  
✅ It's ready for production deployment with Docker  

Happy shopping! 🛍️

