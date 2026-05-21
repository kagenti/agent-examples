# Restaurant Reservation Agent

An AI-powered restaurant reservation assistant built with LangGraph and MCP (Model Context Protocol) tools. This agent provides a natural language interface for searching restaurants, checking availability, and managing reservations.

## Features

- **Natural Language Interface** - Converse naturally about restaurant reservations
- **Restaurant Search** - Find restaurants by city, cuisine, and price tier
- **Availability Checking** - See available time slots at restaurants
- **Reservation Management** - Make, list, and cancel reservations
- **MCP Integration** - Connects to reservation MCP tools via streamable HTTP

## Quick Start

### Prerequisites

- Python 3.11+
- `uv` package manager
- Access to an Ollama instance (or OpenAI-compatible LLM API)
- Reservation MCP server running (see [reservation_tool](../../mcp/reservation_tool/))

### Running Locally

1. **Generate lock file:**
   ```bash
   cd agent-examples/a2a/reservation_service
   uv lock
   ```

2. **Set environment variables:**
   ```bash
   export MCP_URL="http://localhost:8000/mcp"
   export LLM_API_BASE="http://localhost:11434/v1"
   export LLM_MODEL="llama3.2:3b-instruct-fp16"
   export LLM_API_KEY="dummy"
   ```

3. **Run the agent:**
   ```bash
   uv run server
   ```

The agent will start on `http://0.0.0.0:8000`.

## Configuration

Configure via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_URL` | `http://reservation-tool:8000/mcp` | URL of the reservation MCP server |
| `MCP_TRANSPORT` | `streamable_http` | MCP transport protocol |
| `LLM_API_BASE` | `http://host.docker.internal:11434/v1` | LLM API endpoint |
| `LLM_MODEL` | `llama3.2:3b-instruct-fp16` | LLM model to use |
| `LLM_API_KEY` | `dummy` | API key for LLM (use "dummy" for Ollama) |

## Usage Examples

Once deployed, you can interact with the agent using natural language:

```
Find Italian restaurants in Boston
```

```
Check availability at Trattoria di Mare for 4 people on December 25th at 7 PM
```

```
Make a reservation at Trattoria di Mare for December 25th at 7:00 PM, party of 4.
Name: Jane Smith, Phone: +1-555-987-6543, Email: jane@example.com
```

```
List all my reservations using email jane@example.com
```

```
Cancel reservation res_12345 because plans changed
```

## Deployment in Kagenti

### Via Kagenti UI

1. Navigate to **Import New Agent** in the Kagenti UI
2. Fill in the deployment details:
   - **Namespace**: `team1` (or your namespace)
   - **Environment Variable Sets**: Select `ollama` and `mcp-reservations`
   - **Agent Source Repository URL**: `https://github.com/kagenti/agent-examples`
   - **Git Branch or Tag**: `main`
   - **Protocol**: `a2a`
   - **Source Subfolder**: `a2a/reservation_service`
3. Click **Build & Deploy New Agent**
4. Wait for the build to complete (2-5 minutes)
5. Test via the Agent Catalog

### Environment Variable Sets Required

Ensure these are configured in `sample-environments.yaml`:

```yaml
ollama: |
  [
    {"name": "LLM_API_BASE", "value": "http://host.docker.internal:11434/v1"},
    {"name": "LLM_API_KEY", "value": "dummy"},
    {"name": "LLM_MODEL", "value": "llama3.2:3b-instruct-fp16"}
  ]

mcp-reservations: |
  [
    {"name": "MCP_URL", "value": "http://reservation-tool:8000/mcp"}
  ]
```

## Architecture

The agent uses:

- **LangGraph** - For orchestrating multi-step agent workflows
- **LangChain MCP Adapters** - To connect to MCP tools
- **A2A SDK** - For agent-to-agent communication protocol
- **OpenAI-compatible LLM** - For natural language understanding (Ollama or OpenAI)

### Workflow

1. User sends natural language request
2. Agent analyzes request and determines which MCP tools to use
3. Agent calls appropriate reservation tools (search, check availability, etc.)
4. Agent synthesizes results into natural language response
5. Multi-turn conversations supported for complex reservations

## Development

### Running Tests

```bash
uv run pytest
```

### Docker Build

```bash
docker build -t reservation-service:latest .
docker run -p 8000:8000 \
  -e MCP_URL=http://reservation-tool:8000/mcp \
  -e LLM_API_BASE=http://host.docker.internal:11434/v1 \
  reservation-service:latest
```

## Troubleshooting

### Agent can't connect to MCP server

**Symptom**: Error messages about MCP connection failures

**Solution**:
- Verify reservation MCP server is running: `kubectl get pods -n team1`
- Check MCP_URL environment variable is correct
- Test MCP server directly: `curl http://reservation-tool:8000/mcp`

### LLM connection errors

**Symptom**: Timeouts or connection errors to LLM

**Solution**:
- Verify Ollama is running (if using local Ollama)
- Check LLM_API_BASE environment variable
- Ensure model is downloaded: `ollama list`

### Agent responds but doesn't use tools

**Symptom**: Agent gives generic responses without calling MCP tools

**Solution**:
- Check MCP server has tools registered
- Verify system prompt in `graph.py`
- Check LLM model supports tool calling (llama3.2+ recommended)

## Related

- [Reservation MCP Tool](../../mcp/reservation_tool/) - The MCP server this agent connects to
- [Weather Service Agent](../weather_service/) - Similar agent pattern for weather data

## License

Apache License 2.0
