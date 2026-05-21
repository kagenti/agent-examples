# Restaurant Reservation MCP Server - Implementation Plan

## Overview

This document outlines the implementation plan for the Restaurant Reservation MCP server, a new example for the agent-examples repository that demonstrates how to build a multi-tool MCP server with provider abstraction for future extensibility.

## Alignment with Kagenti/agent-examples Conventions

This implementation follows the established patterns in the agent-examples repository:

### Language & Framework
- **Python 3.10+** with `uv` package manager (matching weather_tool, movie_tool, slack_tool)
- **FastMCP framework** for MCP server implementation
- **Type hints** throughout for clarity and IDE support

### Directory Structure
- Located in `mcp/reservation_tool/` following the convention
- Modular structure with separate files for:
  - `reservation_tool.py` - Main MCP server and tool handlers
  - `schemas.py` - Pydantic models for data validation
  - `providers/` - Provider abstraction layer (base + mock)
  - `tests/` - Unit and integration tests

### MCP Tool Annotations
All tools include Kagenti-specific MCP annotations:
- `readOnlyHint`: True for search/query operations, False for mutations
- `destructiveHint`: True only for cancel_reservation, False otherwise
- `idempotentHint`: True for safe retry operations

### Transport & Configuration
- Default transport: `streamable-http` on `0.0.0.0:8000`
- Environment variables for configuration (LOG_LEVEL, HOST, PORT, MCP_TRANSPORT)
- Optional JWT authentication support (JWKS_URI, ISSUER)

### Docker & Deployment
- Dockerfile based on `ghcr.io/astral-sh/uv:python3.12-bookworm-slim`
- Runs as non-root user (1001:1001)
- Standard .dockerignore to optimize builds

### CI/CD Integration
- Follows existing flake8 linting patterns
- pytest for testing
- Added to build.yaml matrix for Docker multi-arch builds

## Architecture

### Provider Pattern

We implement a **provider abstraction layer** to enable future integration with real reservation systems:

```
ReservationProvider (ABC)
├── MockProvider (included)
└── Future providers:
    ├── OpenTableProvider
    ├── SevenRoomsProvider
    ├── ResyProvider
    └── Custom providers
```

**Why this approach:**
- **Clean separation** between MCP tool handlers and data source logic
- **Easy testing** with deterministic mock data
- **Future extensibility** for real API integrations
- **Follows SOLID principles** (Dependency Inversion, Open/Closed)

### Data Models (Pydantic)

Using Pydantic for:
- Runtime type validation
- Clear API contracts
- Automatic JSON serialization
- FastMCP compatibility

**Schemas:**
- `Location` - Geographic coordinates and address
- `Restaurant` - Restaurant details (id, name, cuisine, price tier, location)
- `AvailabilitySlot` - Time slots with party size capacity
- `Reservation` - Booking confirmation details
- `CancellationReceipt` - Cancellation confirmation

### MCP Tools

#### 1. `search_restaurants` (Read-Only)
**Purpose:** Find restaurants matching criteria

**Parameters:**
- `city` (str): City name
- `cuisine` (str, optional): Cuisine type filter
- `date_time` (str, optional): ISO 8601 datetime
- `party_size` (int, optional): Number of guests
- `price_tier` (int, optional): 1-4 ($-$$$$)
- `distance_km` (float, optional): Max distance from city center

**Returns:** List[Restaurant]

**Annotations:** `readOnlyHint=True, destructiveHint=False, idempotentHint=True`

#### 2. `check_availability` (Read-Only)
**Purpose:** Get available time slots for a restaurant

**Parameters:**
- `restaurant_id` (str): Restaurant identifier
- `date_time` (str): ISO 8601 datetime (day to check)
- `party_size` (int): Number of guests

**Returns:** List[AvailabilitySlot]

**Annotations:** `readOnlyHint=True, destructiveHint=False, idempotentHint=True`

#### 3. `place_reservation` (Mutating, Idempotent)
**Purpose:** Book a table (mock confirmation)

**Parameters:**
- `restaurant_id` (str): Restaurant identifier
- `date_time` (str): ISO 8601 datetime
- `party_size` (int): Number of guests
- `name` (str): Guest name
- `phone` (str): Contact phone
- `email` (str): Contact email
- `notes` (str, optional): Special requests

**Returns:** Reservation

**Annotations:** `readOnlyHint=False, destructiveHint=False, idempotentHint=True`

**Note:** Idempotent because duplicate requests return same confirmation

#### 4. `cancel_reservation` (Destructive, Idempotent)
**Purpose:** Cancel an existing reservation

**Parameters:**
- `reservation_id` (str): Reservation identifier
- `reason` (str, optional): Cancellation reason

**Returns:** CancellationReceipt

**Annotations:** `readOnlyHint=False, destructiveHint=True, idempotentHint=True`

#### 5. `list_reservations` (Read-Only)
**Purpose:** List reservations for a user (stub for demo)

**Parameters:**
- `user_id` (str): User identifier (email or phone)

**Returns:** List[Reservation]

**Annotations:** `readOnlyHint=True, destructiveHint=False, idempotentHint=True`

## MockProvider Implementation

### Deterministic Data
- **Fixed restaurant dataset** (10-15 restaurants across multiple cities)
- **Predictable availability** based on time/date heuristics
- **In-memory reservation store** for demo persistence within session
- **No external dependencies** or API calls

### Demo Scenarios
The mock provider supports realistic workflows:
1. Search for Italian restaurants in Boston
2. Check availability at a specific restaurant
3. Place a reservation
4. List user's reservations
5. Cancel a reservation

## Testing Strategy

### Unit Tests
- Schema validation (Pydantic models)
- Provider methods (MockProvider logic)
- Tool handlers (input/output contracts)

### Integration Tests
- Full MCP server initialization
- Tool invocation via FastMCP test client
- End-to-end reservation workflow

### Demo Script (`demo.sh`)
- Boots the server locally
- Exercises all 5 tools in sequence
- Prints formatted JSON outputs
- Validates the happy path workflow

## Documentation

### README.md
- Quick start guide
- Environment variables table
- Local run instructions (`uv run reservation_tool.py`)
- Example curl commands for MCP initialization and tool calls
- Sample Generic Agent prompts for testing with Kagenti

### PLAN.md (this file)
- Architecture decisions
- Alignment with repo conventions
- Future extensibility notes

## Deviations from Simpler Examples

Unlike `weather_tool.py` or `movie_tool.py`, this implementation uses a more structured approach:

**Why:**
- **Educational value**: Demonstrates provider pattern for contributors
- **Complexity**: Multiple related tools warrant better organization
- **Extensibility**: Clear path for real provider implementations
- **Reusability**: Provider interface can be copied for other tools

**Trade-offs:**
- Slightly more files/complexity vs. single-file simplicity
- Better suited for real-world use cases
- Still follows all core conventions (FastMCP, uv, Docker, etc.)

## Future Work (Out of Scope for MVP)

These would be follow-up issues/PRs:

1. **Real Provider Implementations**
   - OpenTable API integration
   - SevenRooms SDK integration
   - Authentication/API key management

2. **Enhanced Features**
   - Restaurant rating/review integration
   - Dietary restriction filtering
   - Reservation modification (not just cancel)
   - Waitlist support

3. **Advanced Auth**
   - Fine-grained authz (user can only see their own reservations)
   - OAuth integration with restaurant systems

4. **Observability**
   - Structured logging with trace IDs
   - Metrics for tool invocations
   - Integration with Kagenti observability stack

5. **Testing**
   - Contract tests for future providers
   - Performance benchmarks
   - Chaos testing for failure scenarios

## Commit Strategy

Following Conventional Commits style seen in the repo:

1. `feat(reservation-mcp): scaffold server with schemas and provider interface`
2. `feat(reservation-mcp): implement search and availability tools`
3. `feat(reservation-mcp): add place/cancel/list reservation tools`
4. `test(reservation-mcp): add unit and integration tests`
5. `docs(reservation-mcp): add README and demo script`
6. `ci(reservation-mcp): add to build workflow matrix`

## Success Criteria

✅ Server boots and registers all 5 tools
✅ `tools/list` returns properly annotated tool schemas
✅ All tools callable via MCP HTTPStreamable protocol
✅ Mock provider returns deterministic, realistic data
✅ Tests pass with pytest
✅ Linting passes with flake8
✅ Docker build succeeds
✅ Demo script executes full workflow
✅ Documentation clear and complete
✅ Follows all agent-examples conventions
