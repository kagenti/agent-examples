# Flight MCP tool

This MCP server exposes tools for searching flights via a fast-flights style API, modeled after `mcp/weather_tool`.

## Tools
- `search_flights(from_airport, to_airport, departure_date, return_date?, cabin?, adults?, children?, infants_in_seat?, infants_on_lap?, currency?, airlines?, max_stops?)`
- `list_prices(result_json)`
- `list_durations_and_stops(result_json)`
- `best_flight(result_json, prefer?)` where `prefer` in `price|duration|stops` or composite by default

### search_flights Parameters:
- **Required**: `from_airport`, `to_airport`, `departure_date`
- **Optional**: `return_date`, `cabin` (defaults to economy), `adults` (defaults to 1), `children` (defaults to 0), `infants_in_seat` (defaults to 0), `infants_on_lap` (defaults to 0), `currency` (defaults to USD), `airlines`, `max_stops`

## Environment
- `FAST_FLIGHTS_CURRENCY` (optional, e.g., `USD`)
- `FAST_FLIGHTS_MOCK` (optional, set to `true` to use mock data for testing)
- `HOST` (default: `0.0.0.0`)
- `PORT` (default: `8000`)
- `MCP_TRANSPORT` (default: `streamable-http`)
- `LOG_LEVEL` (default: `INFO`)

## Run locally
```bash
# Normal mode (requires fast_flights service)
uv run --no-sync flight_tool.py

# Mock mode for testing (no external service needed)
FAST_FLIGHTS_MOCK=true uv run --no-sync flight_tool.py
```

## Docker
```bash
docker build -t flight-tool .
docker run -p 8000:8000 -e HOST=0.0.0.0 -e PORT=8000 flight-tool
```

## Notes
The fast-flights API here is illustrative. Adjust `FAST_FLIGHTS_BASE_URL` and authentication to your provider.

