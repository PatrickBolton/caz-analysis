"""Shared steps of the Liu et al. (2023) replication and its robustness checks.

Used by scripts/replicate_liu2023.py and scripts/robustness_liu2023.py.
A `Window` fixes the study period and the (real or placebo) treatment date;
weather-normalised series are cached per window and random seed, because the
normalisation takes ~20 s per site.
"""

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sqlalchemy import Engine

from caz import synth
from caz.deweather import normalise, weekly_means
from caz.reference import REPO_ROOT
from caz.weather import VARIABLES, era5_table, fetch_era5

ERA5_CACHE = REPO_ROOT / "data" / "raw" / "era5"
OUT = REPO_ROOT / "data" / "processed" / "replication_liu2023"
MIN_WEEK_HOURS = 0.75 * 168  # a week counts if >= 75% of its hours have a value
MAX_DONOR_GAP_WEEKS = 4  # donor gaps up to this long are interpolated; longer: dropped


@dataclass(frozen=True)
class Window:
    """Study period, treatment week, and the post-treatment windows to report."""

    name: str
    start: dt.date
    end: dt.date
    treatment_week: pd.Timestamp  # Monday of the first treated week
    effect_windows: dict  # name -> last week start included (None: to the end)

    @property
    def first_full_week(self) -> pd.Timestamp:
        """First Monday on or after the start."""
        start = pd.Timestamp(self.start)
        return start + pd.Timedelta(days=(7 - start.weekday()) % 7)

    @property
    def last_full_week(self) -> pd.Timestamp:
        """Monday of the last complete week (Monday-Sunday) before the end."""
        end = pd.Timestamp(self.end)
        last_sunday = end - pd.Timedelta(days=(end.weekday() + 1) % 7)
        return last_sunday - pd.Timedelta(days=6)


# Liu et al. (2023): data 1 Jan 2021 - 31 May 2022 (p. 207); effects from the week
# of 24 May 2021 (p. 210); Table 1 reports 7 months (to Dec 2021) and 1 year.
PAPER_WINDOW = Window(
    name="paper",
    start=dt.date(2021, 1, 1),
    end=dt.date(2022, 5, 31),
    treatment_week=pd.Timestamp("2021-05-24"),
    effect_windows={"7 months": pd.Timestamp("2021-12-27"), "1 year": None},
)
# In-time placebo (as the paper's Fig. 13): the same design two years earlier,
# a fake zone from the week of 27 May 2019, 7 months after (COVID starts in 2020).
PLACEBO_2019 = Window(
    name="placebo2019",
    start=dt.date(2019, 1, 1),
    end=dt.date(2019, 12, 31),
    treatment_week=pd.Timestamp("2019-05-27"),
    effect_windows={"7 months": None},
)

HOURLY = """
    SELECT measured_at, no2 AS value FROM obs.hourly
    WHERE network = 'AURN' AND site_id = %(site_id)s
      AND measured_at >= %(start)s AND measured_at < %(end)s
    ORDER BY measured_at
"""
SITES = """
    SELECT r.site_id, r.role, r.site_class, r.paper_name,
           ST_Y(s.geom) AS lat, ST_X(s.geom) AS lon
    FROM ref.replication_sites r
    JOIN aq.sites s USING (network, site_id)
    WHERE r.study = 'liu2023'
    ORDER BY r.role DESC, r.site_class, r.site_id
"""


def replication_sites(engine: Engine) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(SITES, conn)


def deweathered(
    engine: Engine, site, window: Window, seed: int = 0, refresh: bool = False
) -> pd.DataFrame | None:
    """Hourly observed and weather-normalised NO2 for one site and window (cached).

    None if the site has no NO2 in the window.
    """
    cache = OUT / "deweathered" / f"{window.start}_{window.end}_seed{seed}" / f"{site.site_id}.csv"
    if cache.exists() and not refresh:
        out = pd.read_csv(cache, parse_dates=["measured_at"])
        return out if len(out) else None
    params = {"site_id": site.site_id, "start": window.start,
              "end": window.end + dt.timedelta(days=1)}  # fmt: skip
    with engine.connect() as conn:
        no2 = pd.read_sql(HOURLY, conn, params=params)
    cache.parent.mkdir(parents=True, exist_ok=True)
    if no2["value"].notna().sum() < 24 * 30:  # under a month of data: nothing to normalise
        pd.DataFrame(columns=["measured_at", "observed", "normalised", "r2_oob"]).to_csv(
            cache, index=False
        )
        return None
    no2["measured_at"] = no2["measured_at"].dt.tz_convert("UTC")
    era5_file = ERA5_CACHE / f"AURN_{site.site_id}_{window.start}_{window.end}.json"
    raw = fetch_era5(site.lat, site.lon, window.start, window.end, era5_file)
    weather = era5_table(raw, window.start, window.end)
    hourly = no2.merge(weather, on="measured_at", how="left")
    result = normalise(hourly, list(VARIABLES.values()), seed=seed)
    out = result.hourly.assign(r2_oob=result.r2_oob)
    out.to_csv(cache, index=False)
    return out


def weekly_panel(hourly: dict[str, pd.DataFrame], window: Window) -> pd.DataFrame:
    """Weeks x sites of weather-normalised NO2; weeks with < 75% of hours are missing."""
    columns = {}
    for site_id, h in hourly.items():
        weekly = weekly_means(h, ["normalised"]).set_index("week_start")
        week_of_hour = (
            h["measured_at"].dt.tz_convert("UTC").dt.tz_localize(None)
            .dt.to_period("W-SUN").dt.start_time
        )  # fmt: skip
        hours = week_of_hour.value_counts()  # hours with a normalised value, per week
        columns[site_id] = weekly["normalised"].where(hours.reindex(weekly.index) >= MIN_WEEK_HOURS)
    panel = pd.DataFrame(columns).sort_index()
    return panel.loc[window.first_full_week : window.last_full_week]


def estimate(panel: pd.DataFrame, treated: str, donors: list[str], window: Window) -> dict:
    """Ridge ASCM for one treated series; average effects over the window's periods.

    Donor gaps up to MAX_DONOR_GAP_WEEKS are interpolated, donors with longer gaps
    dropped; the treated series is never filled (only its own weeks are used).
    """
    filled, dropped = synth.fill_donor_gaps(panel[donors], max_gap=MAX_DONOR_GAP_WEEKS)
    usable = list(filled.columns)
    weeks = panel.index[panel[treated].notna()]
    y1 = panel.loc[weeks, treated].to_numpy()
    y0 = filled.loc[weeks, usable].to_numpy().T
    t0 = int((weeks < window.treatment_week).sum())
    result = synth.fit(y1, y0, t0)
    post = weeks >= window.treatment_week
    effects = {}
    for name, last in window.effect_windows.items():
        selected = post if last is None else post & (weeks <= last)
        effects[name] = synth.average_effect(result, y1, selected)
    return {
        "result": result,
        "weeks": weeks,
        "donors_used": usable,
        "donors_dropped": sorted(dropped),
        "pre_rmse": float(np.sqrt(np.mean(result.gap[:t0] ** 2))),
        "effects": effects,
    }
