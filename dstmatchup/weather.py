"""Roof and forecast weather signal for one game.

`roof_signal` gives the base value: dome or closed roofs are 0.0 (dome offense
is favored, so no defensive bonus), outdoor or open roofs are 0.5. Outdoor
games add 0.5 when the forecast wind is above 15 mph or the temperature is
below 25 F. Domes ignore the forecast. Missing data stays missing (None).

`fetch_forecast` calls Open-Meteo and returns None on any failure.
"""

from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import requests

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
WIND_THRESHOLD_MPH = 15.0
COLD_THRESHOLD_F = 25.0
TIMEOUT_SECONDS = 5

# Approximate (lat, lon) of each team's home stadium, keyed by nflverse code.
STADIUM_COORDS = {
    "ARI": (33.5276, -112.2626),
    "ATL": (33.7554, -84.4010),
    "BAL": (39.2780, -76.6227),
    "BUF": (42.7738, -78.7870),
    "CAR": (35.2258, -80.8528),
    "CHI": (41.8623, -87.6167),
    "CIN": (39.0954, -84.5160),
    "CLE": (41.5061, -81.6995),
    "DAL": (32.7473, -97.0945),
    "DEN": (39.7439, -105.0201),
    "DET": (42.3400, -83.0456),
    "GB": (44.5013, -88.0622),
    "HOU": (29.6847, -95.4107),
    "IND": (39.7601, -86.1639),
    "JAX": (30.3239, -81.6373),
    "KC": (39.0489, -94.4839),
    "LA": (33.9535, -118.3387),
    "LAC": (33.9535, -118.3387),
    "LV": (36.0909, -115.1833),
    "MIA": (25.9580, -80.2389),
    "MIN": (44.9735, -93.2575),
    "NE": (42.0909, -71.2643),
    "NO": (29.9511, -90.0812),
    "NYG": (40.8135, -74.0745),
    "NYJ": (40.8135, -74.0745),
    "PHI": (39.9008, -75.1675),
    "PIT": (40.4468, -80.0158),
    "SF": (37.4033, -121.9694),
    "SEA": (47.5952, -122.3316),
    "TB": (27.9759, -82.5033),
    "TEN": (36.1665, -86.7713),
    "WAS": (38.9076, -76.8645),
}


def roof_signal(roof) -> float | None:
    if roof is None or (isinstance(roof, float) and np.isnan(roof)):
        return None
    text = str(roof).strip().lower()
    if text in ("dome", "closed"):
        return 0.0
    if text in ("outdoors", "open"):
        return 0.5
    return None


def weather_signal(roof, forecast: dict | None) -> float | None:
    base = roof_signal(roof)
    if base is None:
        return None
    if base == 0.0 or forecast is None:
        return base
    windy = forecast["wind"] > WIND_THRESHOLD_MPH
    cold = forecast["temp"] < COLD_THRESHOLD_F
    return base + 0.5 if (windy or cold) else base


def _utc_hour_key(kickoff: str) -> str:
    """Kickoff in Eastern time (nflverse convention) to Open-Meteo's UTC hour key."""
    local = datetime.fromisoformat(kickoff.replace(" ", "T"))
    if local.tzinfo is None:
        local = local.replace(tzinfo=ZoneInfo("America/New_York"))
    utc = local.astimezone(ZoneInfo("UTC"))
    return utc.strftime("%Y-%m-%dT%H:00")


def fetch_forecast(lat: float, lon: float, kickoff: str) -> dict | None:
    """Forecast {"temp": F, "wind": mph} for the kickoff hour, or None on any error."""
    try:
        hour = _utc_hour_key(kickoff)
        resp = requests.get(
            OPEN_METEO_URL,
            params={
                "latitude": lat,
                "longitude": lon,
                "hourly": "temperature_2m,wind_speed_10m",
                "temperature_unit": "fahrenheit",
                "wind_speed_unit": "mph",
                "timezone": "UTC",
                "forecast_days": 16,
            },
            timeout=TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        hourly = resp.json()["hourly"]
        idx = hourly["time"].index(hour)
        return {
            "temp": float(hourly["temperature_2m"][idx]),
            "wind": float(hourly["wind_speed_10m"][idx]),
        }
    except Exception:
        return None
