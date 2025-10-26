"""
Weather MCP Server - Provides weather data using OpenWeatherMap API
"""

import os
import httpx
from typing import Any
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastMCP server
mcp = FastMCP(name="weather")

# Constants
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")
OPENWEATHER_BASE_URL = "https://api.openweathermap.org/data/2.5"


async def make_weather_request(url: str) -> dict[str, Any] | None:
    """Make a request to OpenWeatherMap API with error handling."""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, timeout=30.0)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            print(f"Weather API error: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error: {e}")
            return None


@mcp.tool()
async def get_current_weather(location: str, units: str = "metric") -> str:
    """Get current weather conditions for any location.

    Args:
        location: City name (e.g., "London", "New York", "Tokyo")
        units: Temperature units - "metric" (Celsius), "imperial" (Fahrenheit), or "standard" (Kelvin)

    Returns:
        Current weather information including temperature, humidity, and conditions
    """
    if not OPENWEATHER_API_KEY:
        return "Error: OPENWEATHER_API_KEY not configured"

    url = f"{OPENWEATHER_BASE_URL}/weather?q={location}&appid={OPENWEATHER_API_KEY}&units={units}"
    data = await make_weather_request(url)

    if not data:
        return f"Unable to fetch weather data for {location}. Please check the city name."

    # Format response
    temp_unit = "°C" if units == "metric" else "°F" if units == "imperial" else "K"

    weather_info = f"""
Weather in {data['name']}, {data['sys']['country']}:
Temperature: {data['main']['temp']}{temp_unit}
Feels like: {data['main']['feels_like']}{temp_unit}
Conditions: {data['weather'][0]['description'].capitalize()}
Humidity: {data['main']['humidity']}%
Wind Speed: {data['wind']['speed']} {"m/s" if units == "metric" else "mph"}
"""
    return weather_info.strip()


@mcp.tool()
async def get_forecast(location: str, days: int = 5) -> str:
    """Get weather forecast for upcoming days.

    Args:
        location: City name (e.g., "London", "Paris")
        days: Number of days to forecast (1-5)

    Returns:
        Weather forecast for the specified number of days
    """
    if not OPENWEATHER_API_KEY:
        return "Error: OPENWEATHER_API_KEY not configured"

    # Limit days to 1-5
    days = max(1, min(5, days))

    url = f"{OPENWEATHER_BASE_URL}/forecast?q={location}&appid={OPENWEATHER_API_KEY}&units=metric&cnt={days * 8}"
    data = await make_weather_request(url)

    if not data:
        return f"Unable to fetch forecast for {location}. Please check the city name."

    # Parse forecast data (API returns 3-hour intervals)
    forecasts = []
    for i in range(0, min(len(data['list']), days * 8), 8):  # One forecast per day
        forecast_data = data['list'][i]
        date = forecast_data['dt_txt'].split()[0]
        temp = forecast_data['main']['temp']
        description = forecast_data['weather'][0]['description']

        forecasts.append(f"{date}: {temp}°C, {description.capitalize()}")

    forecast_text = f"Weather forecast for {data['city']['name']}, {data['city']['country']}:\n"
    forecast_text += "\n".join(forecasts)

    return forecast_text


# Run the server
if __name__ == "__main__":
    print("Starting Weather MCP Server...")
    print(f"API Key configured: {'Yes' if OPENWEATHER_API_KEY else 'No'}")
    mcp.run()
