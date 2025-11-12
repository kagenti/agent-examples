# MCP Tools

This directory contains Model Context Protocol (MCP) tools that can be used by AI assistants and agents.

## Available Tools

### 1. Weather Tool (`weather_tool/`)

Get weather information for any city.

**Features**:
- Current weather data
- Temperature, wind speed, conditions
- Uses Open-Meteo API (no API key required)

**Tools**:
- `get_weather(city: str)` - Get weather info for a city

### 2. Movie Tool (`movie_tool/`)

Get movie information and reviews from OMDb.

**Features**:
- Movie details (plot, ratings, actors, awards)
- Full plot summaries
- Uses OMDb API

**Tools**:
- `get_full_plot(movie_title: str)` - Get full plot summary
- `get_movie_details(movie_title: str)` - Get full movie details

**Requirements**:
- OMDB_API_KEY environment variable

### 3. Slack Tool (`slack_tool/`)

Interact with Slack workspaces.

**Features**:
- List channels
- Get channel history
- Optional fine-grained authorization

**Tools**:
- `get_channels()` - Lists all public and private channels
- `get_channel_history(channel_id: str, limit: int)` - Fetches recent messages

**Requirements**:
- SLACK_BOT_TOKEN environment variable
- Optional: ADMIN_SLACK_BOT_TOKEN for fine-grained auth

### 4. GitHub Tool (`github_tool/`)

Interact with GitHub repositories (written in Go).

**Features**:
- Repository management
- Issue tracking
- Pull request operations

**Requirements**:
- GitHub authentication token

### 5. Shopping Agent (`shopping_agent/`) ⭐ NEW

AI-powered shopping recommendations using LangChain, LangGraph, OpenAI, and SerpAPI.

**Features**:
- Natural language query understanding
- Real-time product search across retailers
- AI-curated recommendations with reasoning
- Budget-aware suggestions
- Multi-step LangGraph workflow

**Tools**:
- `recommend_products(query: str, maxResults: int)` - Get AI-powered product recommendations
- `search_products(query: str, maxResults: int)` - Raw product search

**Requirements**:
- OPENAI_API_KEY environment variable
- SERPAPI_API_KEY environment variable

**Example Usage**:
```bash
curl -X POST http://localhost:8000/tools/recommend_products \
  -H "Content-Type: application/json" \
  -d '{
    "query": "I want to buy a scarf for 40 dollars",
    "maxResults": 5
  }'
```

**Technologies**:
- FastMCP - MCP server framework
- LangChain - LLM application framework
- LangGraph - Agent workflow orchestration
- OpenAI GPT-4o-mini - Natural language understanding/generation
- SerpAPI - Product search across retailers

**Documentation**:
- [README.md](shopping_agent/README.md) - Full documentation
- [QUICKSTART.md](shopping_agent/QUICKSTART.md) - Quick start guide
- [ARCHITECTURE.md](shopping_agent/ARCHITECTURE.md) - Architecture details
- [VERIFICATION.md](shopping_agent/VERIFICATION.md) - Requirements verification

## Getting Started

### General Setup

All MCP tools follow a similar pattern:

1. **Install dependencies**:
```bash
cd <tool_directory>
uv pip install -e .
```

2. **Configure environment variables**:
```bash
export API_KEY="your-api-key-here"
```

3. **Run the server**:
```bash
python <tool_name>.py
```

4. **Test the server**:
```bash
curl http://localhost:8000/health
```

### Docker Deployment

Each tool includes a Dockerfile for containerized deployment:

```bash
cd <tool_directory>
docker build -t <tool-name>-mcp .
docker run -p 8000:8000 -e API_KEY="your-key" <tool-name>-mcp
```

## MCP Protocol

All tools expose functionality via the Model Context Protocol (MCP), which allows AI assistants to discover and use these tools programmatically.

### Key Features

- **Tool Discovery**: Tools are self-describing with metadata
- **Type Safety**: Strong typing for parameters and returns
- **Documentation**: Built-in documentation for each tool
- **Error Handling**: Standardized error responses
- **Transport**: HTTP transport support (streamable HTTP optional)

### Tool Annotations

Tools use annotations to describe their behavior:
- `readOnlyHint`: Indicates if the tool only reads data
- `destructiveHint`: Warns if the tool modifies or deletes data
- `idempotentHint`: Indicates if repeated calls have the same effect

## Framework Comparison

| Tool | Language | Framework | Key Library |
|------|----------|-----------|-------------|
| Weather | Python | FastMCP | requests |
| Movie | Python | FastMCP | requests + OMDb |
| Slack | Python | FastMCP | slack_sdk |
| GitHub | Go | Custom | GitHub API |
| Shopping Agent | Python | FastMCP | LangChain + LangGraph |

## Advanced Example: Shopping Agent Architecture

The Shopping Agent demonstrates an advanced MCP tool with:

```
User Query
    ↓
Parse Query Node (OpenAI)
    ↓
Search Products Node (SerpAPI)
    ↓
Generate Recommendations Node (OpenAI)
    ↓
Structured Response
```

**Key Technologies**:
- **LangGraph**: Multi-node agent workflow with state management
- **LangChain**: LLM framework for tool integration
- **OpenAI**: Natural language understanding and generation
- **SerpAPI**: Real-time search across retailers

See [shopping_agent/ARCHITECTURE.md](shopping_agent/ARCHITECTURE.md) for detailed architecture.

## Creating Your Own MCP Tool

### Basic Template

```python
import os
from fastmcp import FastMCP

mcp = FastMCP("My Tool")

@mcp.tool(annotations={"readOnlyHint": True})
def my_function(param: str) -> str:
    """Tool description"""
    # Your logic here
    return result

def run_server():
    transport = os.getenv("MCP_TRANSPORT", "http")
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    mcp.run(transport=transport, host=host, port=port)

if __name__ == "__main__":
    run_server()
```

### Best Practices

1. **Environment Variables**: Use env vars for API keys and configuration
2. **Error Handling**: Return structured error responses
3. **Logging**: Use appropriate log levels
4. **Documentation**: Include docstrings for all tools
5. **Type Hints**: Use type hints for parameters and returns
6. **Testing**: Provide test clients or scripts
7. **Docker**: Include Dockerfile for deployment
8. **README**: Comprehensive documentation

## Common Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| HOST | Server host address | 0.0.0.0 |
| PORT | Server port | 8000 |
| MCP_TRANSPORT | Transport protocol | http |
| LOG_LEVEL | Logging level | INFO |

## Troubleshooting

### Server Won't Start

1. Check API keys are set
2. Verify port 8000 is available
3. Check Python version (3.10+)
4. Review logs for errors

### Tool Returns Errors

1. Verify API keys are valid
2. Check API quota limits
3. Review request parameters
4. Check network connectivity

### Import Errors

1. Install dependencies: `uv pip install -e .`
2. Verify Python version
3. Check for conflicting packages

## Contributing

When adding new MCP tools:

1. Follow the existing structure
2. Use FastMCP framework (or document why not)
3. Include comprehensive README
4. Add Dockerfile
5. Provide test client
6. Document all tools and parameters
7. Use environment variables for secrets
8. Add logging and error handling

## Resources

- [FastMCP Documentation](https://github.com/jlowin/fastmcp)
- [Model Context Protocol Spec](https://modelcontextprotocol.io/)
- [LangChain Documentation](https://python.langchain.com/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)

## License

See the repository's LICENSE file for details.

