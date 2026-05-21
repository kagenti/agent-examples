# Restaurant Reservation MCP Server

A Model Context Protocol (MCP) server that provides tools for searching restaurants, checking availability, and managing reservations. This example demonstrates provider abstraction patterns for future integration with real reservation systems (OpenTable, SevenRooms, Resy, etc.).

## Features

- **5 MCP Tools** with proper MCP annotations (readOnly, destructive, idempotent hints)
- **Provider Abstraction Layer** for easy integration with different reservation backends
- **MockProvider** with deterministic data for demonstration and testing
- **Type-Safe** using Pydantic models for all data structures
- **Production-Ready** with logging, error handling, and optional JWT authentication
- **Fully Tested** with unit and integration tests

## Tools

| Tool | Description | Read-Only | Destructive | Idempotent |
|------|-------------|-----------|-------------|------------|
| `search_restaurants` | Find restaurants by city, cuisine, price tier, etc. | ✅ | ❌ | ✅ |
| `check_availability` | Get available time slots for a restaurant | ✅ | ❌ | ✅ |
| `place_reservation` | Book a table at a restaurant | ❌ | ❌ | ✅ |
| `cancel_reservation` | Cancel an existing reservation | ❌ | ✅ | ✅ |
| `list_reservations` | List all reservations for a user | ✅ | ❌ | ✅ |

## Quick Start

### Prerequisites

- Python 3.10+
- `uv` package manager ([install instructions](https://github.com/astral-sh/uv))

### Running Locally

1. **Install dependencies:**
   ```bash
   cd agent-examples/mcp/reservation_tool
   uv sync
   ```

2. **Start the server:**
   ```bash
   uv run reservation_tool.py
   ```

3. **Test with the demo script:**
   ```bash
   chmod +x demo.sh
   ./demo.sh
   ```

The server will start on `http://0.0.0.0:8000` by default.

## Configuration

Configure the server using environment variables:

| Variable | Required? | Default | Description |
|----------|-----------|---------|-------------|
| `LOG_LEVEL` | No | `INFO` | Application log level (DEBUG, INFO, WARNING, ERROR) |
| `HOST` | No | `0.0.0.0` | Server host address |
| `PORT` | No | `8000` | Server port |
| `MCP_TRANSPORT` | No | `streamable-http` | MCP transport protocol |
| `JWKS_URI` | No | - | JWKS endpoint for JWT validation (optional) |
| `ISSUER` | No | - | Expected JWT issuer (optional, requires `JWKS_URI`) |
| `CLIENT_ID` | No | `reservation-tool` | OAuth client ID for JWT audience validation |

### Example with Authentication

```bash
JWKS_URI=https://keycloak.example.com/realms/master/protocol/openid-connect/certs \
ISSUER=https://keycloak.example.com/realms/master \
uv run reservation_tool.py
```

## Usage Examples

### MCP Protocol Initialization

```bash
MCP_URL="http://localhost:8000/mcp"

# Initialize session
curl -X POST "$MCP_URL" \
  -H "Content-Type: application/json" \
  -H "Accept: application/json, text/event-stream" \
  --data '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
      "protocolVersion": "2025-03-26",
      "capabilities": {},
      "clientInfo": {"name": "TestClient", "version": "1.0.0"}
    }
  }'

# Extract SESSION_ID from response headers (mcp-session-id)
SESSION_ID="<your-session-id>"

# Send initialized notification
curl -X POST "$MCP_URL" \
  -H "mcp-session-id: $SESSION_ID" \
  -H "Content-Type: application/json" \
  --data '{
    "jsonrpc": "2.0",
    "method": "notifications/initialized"
  }'
```

### Search Restaurants

```bash
curl -X POST "$MCP_URL" \
  -H "mcp-session-id: $SESSION_ID" \
  -H "Content-Type: application/json" \
  --data '{
    "jsonrpc": "2.0",
    "id": 2,
    "method": "tools/call",
    "params": {
      "name": "search_restaurants",
      "arguments": {
        "city": "Boston",
        "cuisine": "Italian",
        "price_tier": 3
      }
    }
  }' | jq
```

### Check Availability

```bash
curl -X POST "$MCP_URL" \
  -H "mcp-session-id: $SESSION_ID" \
  -H "Content-Type: application/json" \
  --data '{
    "jsonrpc": "2.0",
    "id": 3,
    "method": "tools/call",
    "params": {
      "name": "check_availability",
      "arguments": {
        "restaurant_id": "rest_001",
        "date_time": "2025-12-25T18:00:00",
        "party_size": 4
      }
    }
  }' | jq
```

### Place Reservation

```bash
curl -X POST "$MCP_URL" \
  -H "mcp-session-id: $SESSION_ID" \
  -H "Content-Type: application/json" \
  --data '{
    "jsonrpc": "2.0",
    "id": 4,
    "method": "tools/call",
    "params": {
      "name": "place_reservation",
      "arguments": {
        "restaurant_id": "rest_001",
        "date_time": "2025-12-25T19:00:00",
        "party_size": 4,
        "name": "John Doe",
        "phone": "+1-555-123-4567",
        "email": "john@example.com",
        "notes": "Window seat preferred"
      }
    }
  }' | jq
```

## Testing with a Generic Agent

If you're using this MCP server with a Kagenti-deployed agent, you can test with prompts like:

```
Find me Italian restaurants in Boston with a price tier of 3
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
Cancel reservation <reservation_id> because plans changed
```

## Development

### Running Tests

```bash
# Install dev dependencies
uv sync --extra dev

# Run tests with pytest
uv run pytest tests/ -v

# Run with coverage
uv run pytest tests/ --cov=. --cov-report=html
```

### Code Quality

```bash
# Lint with flake8
uv run flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
uv run flake8 . --count --max-complexity=10 --max-line-length=127 --statistics
```

### Docker Build

```bash
# Build the image
docker build -t reservation-tool:latest .

# Run the container
docker run -p 8000:8000 \
  -e LOG_LEVEL=DEBUG \
  reservation-tool:latest
```

## Architecture

### Provider Pattern

The server uses a provider abstraction layer to decouple MCP tool handlers from the actual reservation backend:

```
ReservationProvider (ABC)
├── MockProvider (included)
└── Future providers:
    ├── OpenTableProvider
    ├── SevenRoomsProvider
    └── ResyProvider
```

This design allows you to:
- Add new reservation systems without changing tool handlers
- Test with deterministic mock data
- Swap providers via configuration
- Implement provider-specific optimizations

### Directory Structure

```
reservation_tool/
├── __init__.py
├── reservation_tool.py     # MCP server & tool handlers
├── schemas.py               # Pydantic data models
├── providers/
│   ├── __init__.py
│   ├── base.py             # Abstract provider interface
│   └── mock.py             # Mock implementation
├── tests/
│   ├── __init__.py
│   └── test_reservation_tool.py
├── pyproject.toml
├── Dockerfile
├── .dockerignore
├── README.md
├── PLAN.md                  # Architecture & design decisions
└── demo.sh                  # End-to-end demo script
```

## Mock Data

The MockProvider includes 10 sample restaurants across 4 cities:

- **Boston**: Italian, American, Japanese, French, Mexican (5 restaurants)
- **New York**: Chinese, Italian (2 restaurants)
- **San Francisco**: Indian, Vegetarian (2 restaurants)
- **Austin**: BBQ (1 restaurant)

Availability is generated deterministically based on:
- Restaurant ID
- Date/time
- Party size

This ensures consistent results for testing while simulating realistic availability patterns.

## Deployment in Kagenti

### 1. Deploy via Kagenti UI

1. Navigate to **Import New Tool** in the Kagenti UI
2. Select your namespace (e.g., `team1`)
3. Set **Target Port** to `8000`
4. Set **Source Subfolder** to `mcp/reservation_tool`
5. Configure environment variables (optional):
   - `LOG_LEVEL=INFO`
   - `JWKS_URI=<your-keycloak-jwks-uri>` (if using auth)
   - `ISSUER=<your-keycloak-issuer>`
6. Click **Build & Deploy New Tool**

### 2. Add to sample-environments.yaml

```yaml
mcp-reservations: |
  [
    {"name": "MCP_URL", "value": "http://reservation-tool:8000/mcp"}
  ]
```

### 3. Test via MCP Gateway

The MCP Gateway will automatically discover and route to this tool. Agents configured with the `mcp-reservations` environment will be able to use all 5 reservation tools.

## Future Extensions

### Real Provider Implementations

```python
from providers.base import ReservationProvider

class OpenTableProvider(ReservationProvider):
    def __init__(self, api_key: str):
        self.client = OpenTableClient(api_key)

    def search_restaurants(self, city, **kwargs):
        # Call OpenTable API
        ...
```

Then update `reservation_tool.py`:

```python
provider_type = os.getenv("PROVIDER_TYPE", "mock")
if provider_type == "opentable":
    provider = OpenTableProvider(api_key=os.getenv("OPENTABLE_API_KEY"))
elif provider_type == "mock":
    provider = MockProvider()
```

### Enhanced Features

- Restaurant reviews/ratings integration
- Dietary restriction filtering
- Waitlist support
- Reservation modification (not just cancel)
- Multi-restaurant search with geolocation
- Integration with calendaring systems