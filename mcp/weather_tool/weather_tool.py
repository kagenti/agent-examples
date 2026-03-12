"Weather MCP tool example"

import json
import logging
import os
import sys

import requests
from fastmcp import FastMCP

mcp = FastMCP("Weather")
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    stream=sys.stdout,
    format="%(levelname)s: %(message)s",
)


@mcp.tool(annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True})
def get_weather(city: str) -> str:
    """Get weather info for a city"""
    logger.debug(f"Getting weather info for city '{city}'.")
    base_url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {"name": city, "count": 1}
    response = requests.get(base_url, params=params, timeout=10)
    data = response.json()
    if not data or "results" not in data:
        return f"City {city} not found"
    latitude = data["results"][0]["latitude"]
    longitude = data["results"][0]["longitude"]

    weather_url = "https://api.open-meteo.com/v1/forecast"
    weather_params = {
        "latitude": latitude,
        "longitude": longitude,
        "temperature_unit": "fahrenheit",
        "current_weather": True,
    }
    weather_response = requests.get(weather_url, params=weather_params, timeout=10)
    weather_data = weather_response.json()

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
