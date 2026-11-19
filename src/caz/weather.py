"""Hourly ERA5 reanalysis weather for a location, from the Open-Meteo archive API.

ERA5 (ECMWF) is on a 0.25 degree grid (~25 km); Open-Meteo serves it for any
coordinates, free and without an account. Data licence: ERA5 under the
Copernicus licence; Open-Meteo CC BY 4.0 (see docs/DATA_SOURCES.md).

Timing: openair concentrations are hour-beginning (00:00 = mean over 00:00-01:00).
Open-Meteo gives most variables at the instant of the hour, but radiation and
precipitation as the total or mean over the *preceding* hour, so those two are
moved back one hour to cover the same interval as the concentration.
"""

import datetime as dt
import json
from pathlib import Path

import pandas as pd
import requests

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
TIMEOUT_S = 120

# Open-Meteo variable -> our column (names as in Liu et al. 2023, Fig. 6, where they exist)
VARIABLES = {
    "wind_speed_10m": "ws",  # m/s
    "wind_direction_10m": "wd",  # degrees
    "temperature_2m": "temp",  # degrees C
    "relative_humidity_2m": "rh",  # %
    "surface_pressure": "sp",  # hPa
    "boundary_layer_height": "blh",  # m
    "shortwave_radiation": "ssr",  # W/m2, preceding-hour mean
    "cloud_cover": "tcc",  # %
    "precipitation": "tp",  # mm, preceding-hour sum
}
PRECEDING_HOUR = ["ssr", "tp"]  # shifted back one hour (see module docstring)


def fetch_era5(
    lat: float, lon: float, start: dt.date, end: dt.date, cache: Path, refresh: bool = False
) -> dict:
    """Raw Open-Meteo response for one location and period, cached as JSON."""
    if cache.exists() and not refresh:
        return json.loads(cache.read_text())
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start.isoformat(),
        # one extra day, so the last hours can take their shifted values
        "end_date": (end + dt.timedelta(days=1)).isoformat(),
        "hourly": ",".join(VARIABLES),
        "models": "era5",
        "timezone": "GMT",
        "wind_speed_unit": "ms",
    }
    response = requests.get(ARCHIVE_URL, params=params, timeout=TIMEOUT_S)
    response.raise_for_status()
    data = response.json()
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(data))
    return data


def era5_table(data: dict, start: dt.date, end: dt.date) -> pd.DataFrame:
    """Hourly weather from fetch_era5 as a table: measured_at (UTC, hour-beginning) + VARIABLES."""
    hourly = data["hourly"]
    df = pd.DataFrame({ours: hourly[theirs] for theirs, ours in VARIABLES.items()})
    df.insert(0, "measured_at", pd.to_datetime(hourly["time"]).tz_localize("UTC"))
    for column in PRECEDING_HOUR:
        df[column] = df[column].shift(-1)  # value labelled 11:00 covers 10:00-11:00
    first = pd.Timestamp(start, tz="UTC")
    last = pd.Timestamp(end, tz="UTC") + pd.Timedelta(hours=23)
    df = df[df["measured_at"].between(first, last)].reset_index(drop=True)
    if df["measured_at"].duplicated().any() or len(df) != (end - start).days * 24 + 24:
        raise ValueError("ERA5 table does not have exactly one row per hour")
    return df
