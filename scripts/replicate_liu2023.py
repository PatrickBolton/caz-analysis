"""Replicate Liu et al. (2023) at the two Birmingham sites in our data: BIRR and BMLD.

Liu, B. et al. (2023), Assessing the Impacts of Birmingham's Clean Air Zone on Air
Quality, Environmental and Resource Economics 86, 203-231 (see docs/LITERATURE.md).

Run from the project root (database running, hourly data loaded):
    uv run --env-file .env python scripts/replicate_liu2023.py
    uv run --env-file .env python scripts/replicate_liu2023.py --refresh   # redo normalisation

Steps, as in the paper:
1. Hourly NO2, 1 Jan 2021 - 31 May 2022, for the treated sites and the donors of
   their type (ref.replication_sites: Tables 4 and 5).
2. Weather normalisation per site (random forest, ERA5 weather; caz.deweather).
3. Weekly means (Monday start). A week counts if >= 75% of its hours have data.
   Donor gaps of up to 4 weeks are interpolated (a missing first or last week
   takes its neighbour's value); donors with longer gaps are
   dropped (the paper dropped sites with much missing data). The treated site's
   series is never filled: only its own weeks are used.
4. Ridge-augmented synthetic control per treated site (caz.synth): pre-period
   = weeks before 24 May 2021; effects over 7 months (to 31 Dec 2021) and 1 year
   (to 31 May 2022), with 95% jackknife+ intervals.
5. Comparison with the paper's Table 1 (p. 213).
"""

import argparse
import datetime as dt
import time
from pathlib import Path

import numpy as np
import pandas as pd

from caz import synth
from caz.db import get_engine
from caz.deweather import normalise, weekly_means
from caz.weather import VARIABLES, era5_table, fetch_era5

ROOT = Path(__file__).resolve().parents[1]
ERA5_CACHE = ROOT / "data" / "raw" / "era5"
OUT = ROOT / "data" / "processed" / "replication_liu2023"
START, END = dt.date(2021, 1, 1), dt.date(2022, 5, 31)  # p. 207
TREATMENT_WEEK = pd.Timestamp("2021-05-24")  # p. 210: effects began the week before 1 June
SEVEN_MONTHS_END = pd.Timestamp("2021-12-31")
MIN_WEEK_HOURS = 0.75 * 168
MAX_DONOR_GAP_WEEKS = 4  # donor gaps up to this long are interpolated; longer: donor dropped

# Liu et al. (2023), Table 1, p. 213: NO2, absolute change ug/m3 (95% interval), percent change
PAPER = {
    ("BIRR", "7 months"): (-0.84, -1.66, 0.10, -2.52),
    ("BIRR", "1 year"): (-0.53, -1.34, 0.40, -1.58),
    ("BMLD", "7 months"): (-0.94, -2.08, 0.03, -5.60),
    ("BMLD", "1 year"): (-0.94, -2.16, 0.06, -5.45),
}
# Donors whose own city started a charging zone inside the post-period
TREATED_DONORS = {"POAR": "Portsmouth CAZ charging from 29 Nov 2021"}

SITES = """
    SELECT r.site_id, r.role, r.site_class, r.paper_name, ST_Y(s.geom) AS lat, ST_X(s.geom) AS lon
    FROM ref.replication_sites r
    JOIN aq.sites s USING (network, site_id)
    WHERE r.study = 'liu2023'
    ORDER BY r.role DESC, r.site_class, r.site_id
"""
HOURLY = """
    SELECT measured_at, no2 AS value FROM obs.hourly
    WHERE network = 'AURN' AND site_id = %(site_id)s
      AND measured_at >= %(start)s AND measured_at < %(end)s
    ORDER BY measured_at
"""


def deweathered(engine, site: pd.Series, refresh: bool) -> pd.DataFrame:
    """Hourly observed and weather-normalised NO2 for one site (cached as CSV)."""
    cache = OUT / "deweathered" / f"{site.site_id}.csv"
    if cache.exists() and not refresh:
        return pd.read_csv(cache, parse_dates=["measured_at"])
    params = {"site_id": site.site_id, "start": START, "end": END + dt.timedelta(days=1)}
    with engine.connect() as conn:
        no2 = pd.read_sql(HOURLY, conn, params=params)
    no2["measured_at"] = no2["measured_at"].dt.tz_convert("UTC")
    era5_file = ERA5_CACHE / f"AURN_{site.site_id}_{START}_{END}.json"
    weather = era5_table(fetch_era5(site.lat, site.lon, START, END, era5_file), START, END)
    result = normalise(no2.merge(weather, on="measured_at", how="left"), list(VARIABLES.values()))
    out = result.hourly.assign(r2_oob=result.r2_oob)
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache, index=False)
    return out


def weekly_panel(hourly: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Weeks x sites of weather-normalised NO2; weeks with < 75% of hours are missing."""
    columns = {}
    for site_id, h in hourly.items():
        weekly = weekly_means(h, ["normalised"]).set_index("week_start")
        counts = (
            h["measured_at"]
            .dt.tz_convert("UTC")
            .dt.tz_localize(None)
            .dt.to_period("W-SUN")
            .dt.start_time.value_counts()
        )  # fmt: skip  (hours with a normalised value, per week)
        columns[site_id] = weekly["normalised"].where(
            counts.reindex(weekly.index) >= MIN_WEEK_HOURS
        )
    panel = pd.DataFrame(columns).sort_index()
    # complete weeks only: 1 Jan 2021 is a Friday, 31 May 2022 a Tuesday
    first_full_week, last_full_week = pd.Timestamp("2021-01-04"), pd.Timestamp("2022-05-23")
    return panel.loc[first_full_week:last_full_week]


def estimate(panel: pd.DataFrame, treated: str, donors: list[str]) -> dict:
    """Ridge ASCM for one treated site; effects over the paper's two post windows."""
    # Donors: fill short gaps by linear interpolation (weekly weather-normalised
    # series are smooth); drop donors with a longer gap or missing first/last weeks.
    filled, _ = synth.fill_donor_gaps(panel[donors], max_gap=MAX_DONOR_GAP_WEEKS)
    usable = list(filled.columns)
    # Treated site: never filled; only the weeks it has
    weeks = panel.index[panel[treated].notna()]
    y1 = panel.loc[weeks, treated].to_numpy()
    y0 = filled.loc[weeks, usable].to_numpy().T
    t0 = int((weeks < TREATMENT_WEEK).sum())
    result = synth.fit(y1, y0, t0)
    post = weeks >= TREATMENT_WEEK
    windows = {"7 months": post & (weeks <= SEVEN_MONTHS_END), "1 year": post}
    return {
        "result": result,
        "weeks": weeks,
        "donors_used": usable,
        "donors_dropped": sorted(set(donors) - set(usable)),
        "effects": {name: synth.average_effect(result, y1, w) for name, w in windows.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--refresh", action="store_true", help="redo the weather normalisation")
    args = parser.parse_args()

    engine = get_engine()
    with engine.connect() as conn:
        sites = pd.read_sql(SITES, conn)
    hourly = {}
    for i, site in enumerate(sites.itertuples(index=False), start=1):
        started = time.monotonic()
        hourly[site.site_id] = deweathered(engine, site, args.refresh)
        r2 = hourly[site.site_id]["r2_oob"].iloc[0]
        print(f"  {i:2d}/{len(sites)} {site.site_id:5s} {site.paper_name:32s} OOB R2 {r2:.2f}"
              f"  ({time.monotonic() - started:.0f} s)")  # fmt: skip
    panel = weekly_panel(hourly)
    panel.to_csv(OUT / "weekly_normalised.csv")

    runs = {
        "BIRR": ("BIRR", "roadside", []),
        "BMLD": ("BMLD", "background", []),
        "BIRR without POAR": ("BIRR", "roadside", list(TREATED_DONORS)),
    }
    rows = []
    for name, (treated, site_class, exclude) in runs.items():
        is_donor = (sites["role"] == "donor") & (sites["site_class"] == site_class)
        donors = sites.loc[is_donor, "site_id"]
        run = estimate(panel, treated, [d for d in donors if d not in exclude])
        r = run["result"]
        pre_rmse = float(np.sqrt(np.mean(r.gap[: r.t0] ** 2)))
        print(f"\n{name}: {len(run['weeks'])} weeks ({r.t0} before treatment), "
              f"{len(run['donors_used'])} donors, lambda {r.lam:.3g}, "
              f"pre-period RMSE {pre_rmse:.2f} ug/m3")  # fmt: skip
        if run["donors_dropped"]:
            print(f"  donors dropped (missing weeks): {', '.join(run['donors_dropped'])}")
        top = np.argsort(-np.abs(r.weights))[:5]
        print("  largest weights: " + ", ".join(
            f"{run['donors_used'][j]} {r.weights[j]:+.2f}" for j in top))  # fmt: skip
        for window, e in run["effects"].items():
            paper = PAPER.get((name, window))
            rows.append({
                "run": name, "window": window,
                "ours": f"{e.absolute:+.2f} ({e.lower:+.2f}, {e.upper:+.2f})",
                "ours_pct": f"{e.percent:+.1f}%",
                "paper": f"{paper[0]:+.2f} ({paper[1]:+.2f}, {paper[2]:+.2f})" if paper else "",
                "paper_pct": f"{paper[3]:+.1f}%" if paper else "",
            })  # fmt: skip
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "comparison.csv", index=False)
    print("\nNO2 effect, ug/m3 (95% interval), ours vs Liu et al. (2023) Table 1:")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
