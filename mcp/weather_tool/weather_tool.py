"Weather MCP tool example"

import json
import logging
import os
import sys
import time

import requests
from fastmcp import FastMCP
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

mcp = FastMCP("Weather")
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    stream=sys.stdout,
    format="%(levelname)s: %(message)s",
)

# Configurable timeouts and retries for external API resilience
_REQUEST_TIMEOUT = int(os.getenv("WEATHER_REQUEST_TIMEOUT", "30"))
_MAX_RETRIES = int(os.getenv("WEATHER_MAX_RETRIES", "3"))
_BACKOFF_FACTOR = float(os.getenv("WEATHER_BACKOFF_FACTOR", "1.0"))


def _resilient_session() -> requests.Session:
    """Create an HTTP session with automatic retry and exponential backoff."""
    session = requests.Session()
    retry_strategy = Retry(
        total=_MAX_RETRIES,
        backoff_factor=_BACKOFF_FACTOR,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


@mcp.tool(annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True})
def get_weather(city: str) -> str:
    """Get weather info for a city"""
    logger.debug(f"Getting weather info for city '{city}'.")
    session = _resilient_session()

    # Geocoding: resolve city name to coordinates
    base_url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {"name": city, "count": 1}
    try:
        response = session.get(base_url, params=params, timeout=_REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError) as exc:
        logger.warning("Geocoding API error for '%s': %s", city, exc)
        return f"Weather service temporarily unavailable for {city} (geocoding error)"

    if not data or "results" not in data:
        return f"City {city} not found"
    latitude = data["results"][0]["latitude"]
    longitude = data["results"][0]["longitude"]

    # Forecast: get current weather at coordinates
    weather_url = "https://api.open-meteo.com/v1/forecast"
    weather_params = {
        "latitude": latitude,
        "longitude": longitude,
        "temperature_unit": "fahrenheit",
        "current_weather": True,
    }
    try:
        weather_response = session.get(
            weather_url, params=weather_params, timeout=_REQUEST_TIMEOUT
        )
        weather_response.raise_for_status()
        weather_data = weather_response.json()
    except (requests.RequestException, ValueError) as exc:
        logger.warning("Forecast API error for '%s': %s", city, exc)
        return f"Weather service temporarily unavailable for {city} (forecast error)"

    return json.dumps(weather_data["current_weather"])


# host can be specified with HOST env variable
# transport can be specified with MCP_TRANSPORT env variable (defaults to streamable-http)
def run_server():
    "Run the MCP server"
    transport = os.getenv("MCP_TRANSPORT", "streamable-http")
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    mcp.run(transport=transport, host=host, port=port)


if __name__ == "__main__":
    run_server()
