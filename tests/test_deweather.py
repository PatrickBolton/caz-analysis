"""Weather normalisation and ERA5 weather (no network or database needed)."""

import datetime as dt

import numpy as np
import pandas as pd
import pytest

from caz.deweather import normalise, time_variables, weekly_means
from caz.weather import VARIABLES, era5_table


def synthetic_site(n_days: int = 200, step: float = 0.0, seed: int = 1) -> pd.DataFrame:
    """Hourly NO2 = 40 - 3 * wind speed + daily cycle + noise, minus `step` from day 100."""
    rng = np.random.default_rng(seed)
    t = pd.Series(pd.date_range("2021-01-04", periods=24 * n_days, freq="h", tz="UTC"))
    ws = rng.gamma(2, 2, len(t))
    after = (t >= t[0] + pd.Timedelta(days=100)).to_numpy()
    value = 40 - 3 * ws + 5 * np.sin(2 * np.pi * t.dt.hour / 24) - step * after
    value = value + rng.normal(0, 2, len(t))
    return pd.DataFrame({"measured_at": t, "value": value, "ws": ws,
                         "temp": rng.normal(8, 5, len(t))})  # fmt: skip


def test_normalisation_removes_the_weather_effect() -> None:
    site = synthetic_site()
    result = normalise(site, ["ws", "temp"], n_samples=50, n_trees=100)
    h = result.hourly
    assert abs(np.corrcoef(h["observed"], site["ws"])[0, 1]) > 0.8
    assert abs(np.corrcoef(h["normalised"], site["ws"])[0, 1]) < 0.1
    assert result.r2_oob > 0.8


def test_a_step_change_survives_normalisation() -> None:
    # The policy effect is a change in level over time: it must not be averaged away
    site = synthetic_site(step=6.0)
    h = normalise(site, ["ws", "temp"], n_samples=50, n_trees=100).hourly
    cut = site["measured_at"][0] + pd.Timedelta(days=100)
    month = pd.Timedelta(days=30)
    before = h.loc[h["measured_at"].between(cut - month, cut, inclusive="left"), "normalised"]
    after = h.loc[h["measured_at"].between(cut, cut + month, inclusive="left"), "normalised"]
    assert before.mean() - after.mean() == pytest.approx(6.0, abs=1.0)


def test_hours_with_missing_values_are_skipped() -> None:
    site = synthetic_site(n_days=30)
    site.loc[:9, "value"] = np.nan
    site.loc[10:14, "ws"] = np.nan
    h = normalise(site, ["ws", "temp"], n_samples=5, n_trees=20).hourly
    assert len(h) == len(site) - 15


def test_time_variables() -> None:
    t = pd.Series(pd.to_datetime(["2021-05-24 13:00"]).tz_localize("UTC"))  # a Monday
    row = time_variables(t).iloc[0]
    assert (row["weekday"], row["hour"], row["day_julian"]) == (0, 13, 144)
    assert row["date_unix"] == 1621861200


def test_weekly_means_start_on_monday() -> None:
    t = pd.Series(pd.date_range("2021-05-23", periods=48, freq="h", tz="UTC"))  # Sun + Mon
    weekly = weekly_means(pd.DataFrame({"measured_at": t, "x": [1.0] * 24 + [3.0] * 24}), ["x"])
    assert weekly["week_start"].astype(str).tolist() == ["2021-05-17", "2021-05-24"]
    assert weekly["x"].tolist() == [1.0, 3.0]


def test_era5_table_aligns_preceding_hour_variables() -> None:
    start = dt.date(2021, 1, 1)
    times = pd.date_range("2021-01-01", periods=48, freq="h").strftime("%Y-%m-%dT%H:%M")
    hourly = {name: list(range(48)) for name in VARIABLES} | {"time": list(times)}
    table = era5_table({"hourly": hourly}, start, start)
    assert len(table) == 24
    first = table.iloc[0]
    assert first["temp"] == 0  # instantaneous: value at 00:00
    assert first["ssr"] == 1 and first["tp"] == 1  # 00:00-01:00 is labelled 01:00
