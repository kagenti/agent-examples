# Flight MCP tool

This MCP server exposes tools for searching flights via fast-flights, a Python library for accessing Google Flights data

## Tools
- `search_flights(from_airport, to_airport, departure_date, return_date?, cabin?, adults?, children?, infants_in_seat?, infants_on_lap?, currency?, airlines?, max_stops?)` - wrapper around fast-flights flight search API. Returns flights that fit the given parameter.
- `search_airports(query, limit=10)` — wrapper around the fast-flights airport search API. This tool returns the raw API results (serialized) so callers can inspect enum members or IATA codes directly.

### search_flights Parameters:
- **Required**: `from_airport`, `to_airport`, `departure_date`
- **Optional**: `return_date`, `cabin` (defaults to economy), `adults` (defaults to 1), `children` (defaults to 0), `infants_in_seat` (defaults to 0), `infants_on_lap` (defaults to 0), `currency` (defaults to USD), `airlines`, `max_stops`

### Example agent flow

1. Call `search_airports("nyc")` to get candidate airports (JFK, LGA, EWR).
2. Choose one (ask the user or pick a default). Example chosen code: `JFK`.
3. Call `search_flights(from_airport="JFK", to_airport="LAX", departure_date="2026-01-10")` — ensure the date is in the future.
4. Present or post-process the returned `summary` array.

## Environment
- `FAST_FLIGHTS_CURRENCY` (optional, e.g., `USD`)
- `HOST` (default: `0.0.0.0`)
- `PORT` (default: `8000`)
- `MCP_TRANSPORT` (default: `streamable-http`)
- `LOG_LEVEL` (default: `INFO`)

## Run locally
```bash
uv run --no-sync flight_tool.py
```
