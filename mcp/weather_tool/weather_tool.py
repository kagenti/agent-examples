import os
import requests
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Weather", port=8000)

@mcp.tool()
def get_weather(city: str) -> str:
    """Get weather info for a city"""
    base_url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {"name": city, "count": 1}
    response = requests.get(base_url, params=params)
    data = response.json()
    if not data["results"]:
        return "City not found"
    latitude = data["results"][0]["latitude"]
    longitude = data["results"][0]["longitude"]
    
    weather_url = "https://api.open-meteo.com/v1/forecast"
    weather_params = {
        "latitude": latitude,
        "longitude": longitude,
        "current_weather": True
    }
    weather_response = requests.get(weather_url, params=weather_params)
    weather_data = weather_response.json()

    return weather_data["current_weather"]

# host can be specified with HOST env variable
# transport can be specified with MCP_TRANSPORT env variable (defaults to sse)
def run_server():
    transport = os.getenv("MCP_TRANSPORT", "streamable-http")
    mcp.run(transport=transport) 

if __name__ == "__main__":
    run_server()
