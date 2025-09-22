#!/usr/bin/env -S uv --quiet run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "requests",
#   "rich",
# ]
# ///

"""
Weather CLI tool using Open-Meteo APIs.

This script fetches the current temperature for a given city using the Open-Meteo geocoding
and weather APIs. It is intended to be run from the command line.

Examples
--------
Get the current temperature for Berlin:

    $ python weather.py Berlin

If the city is not found or the API is unreachable, an error message will be printed to stderr.
"""

import sys

import requests
from rich.console import Console

console = Console()


def get_city_coords(city_name: str) -> tuple[float, float, str]:
    """
    Get latitude, longitude, and timezone for a city using the Open-Meteo geocoding API.

    Parameters
    ----------
    city_name : str
        The name of the city to search for.

    Returns
    -------
    tuple of (float, float, str)
        The latitude, longitude, and timezone of the city.

    Raises
    ------
    SystemExit
        If the city is not found or the API request fails.
    """
    geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city_name}&count=1"
    response = requests.get(geo_url)
    response.raise_for_status()  # Raise an exception for bad status codes
    geo_data = response.json()
    if not geo_data.get("results"):
        console.print(f"[red]Error:[/red] Could not find city '{city_name}'", file=sys.stderr)
        sys.exit(1)
    location = geo_data["results"][0]
    # Return latitude, longitude, and timezone (default to "auto" if not present)
    return location["latitude"], location["longitude"], location.get("timezone", "auto")


def get_weather(latitude: float, longitude: float, timezone: str) -> dict:
    """
    Fetch the current weather for a given latitude, longitude, and timezone.

    Parameters
    ----------
    latitude : float
        The latitude of the location.
    longitude : float
        The longitude of the location.
    timezone : str
        The timezone string (e.g., "Europe/Berlin" or "auto").

    Returns
    -------
    dict
        The current weather data as returned by the API.

    Raises
    ------
    requests.exceptions.RequestException
        If the API request fails.
    """
    weather_url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={latitude}&longitude={longitude}"
        f"&current=temperature_2m,weather_code"
        f"&timezone={timezone}"
    )
    response = requests.get(weather_url)
    response.raise_for_status()
    return response.json()["current"]


if __name__ == "__main__":
    # Check for correct number of command-line arguments
    if len(sys.argv) != 2:
        console.print("[yellow]Usage:[/yellow] weather <city_name>", file=sys.stderr)
        sys.exit(1)

    city: str = sys.argv[1]
    try:
        lat, lon, tz = get_city_coords(city)
        current_weather = get_weather(lat, lon, tz)
        temp = current_weather["temperature_2m"]
        console.print(
            f"The current temperature in [bold]{city.title()}[/bold] is [cyan]{temp}°C[/cyan]."
        )
    except requests.exceptions.RequestException as e:
        console.print(
            f"[red]Error:[/red] Failed to connect to the weather service. {e}", file=sys.stderr
        )
        sys.exit(1)
