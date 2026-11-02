import logging

import requests
from django.conf import settings

logger = logging.getLogger(__name__)

WEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"


class CityNotFound(Exception):
    """The weather service does not know this city."""


class WeatherError(Exception):
    """The weather service failed or is not configured."""


def fetch_weather(city: str) -> dict:
    """Current weather for a city, as a small dictionary."""
    if not settings.OPENWEATHER_API_KEY:
        logger.error("OPENWEATHER_API_KEY is not configured")
        raise WeatherError("Weather service is not configured")

    params = {"q": city, "appid": settings.OPENWEATHER_API_KEY, "units": "metric"}
    try:
        resp = requests.get(WEATHER_URL, params=params, timeout=settings.WEATHER_TIMEOUT)
    except requests.exceptions.RequestException as e:
        logger.error("Weather API network error: %s", type(e).__name__)
        raise WeatherError("Weather service unreachable")

    if resp.status_code == 404:
        raise CityNotFound(city)
    if resp.status_code != 200:
        logger.error("Weather API returned %s", resp.status_code)
        raise WeatherError("Weather service error")

    try:
        data = resp.json()
        return {
            "city": data["name"],
            "temperature_c": data["main"]["temp"],
            "condition": data["weather"][0]["description"],
            "humidity": data["main"]["humidity"],
        }
    except (ValueError, KeyError, IndexError):
        raise WeatherError("Unexpected weather response")