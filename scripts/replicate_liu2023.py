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
import time

import numpy as np
import pandas as pd

from caz.db import get_engine
from caz.replication import (
    OUT,
    PAPER_WINDOW,
    deweathered,
    estimate,
    replication_sites,
    weekly_panel,
)

# Liu et al. (2023), Table 1, p. 213: NO2, absolute change ug/m3 (95% interval), percent change
PAPER = {
    ("BIRR", "7 months"): (-0.84, -1.66, 0.10, -2.52),
    ("BIRR", "1 year"): (-0.53, -1.34, 0.40, -1.58),
    ("BMLD", "7 months"): (-0.94, -2.08, 0.03, -5.60),
    ("BMLD", "1 year"): (-0.94, -2.16, 0.06, -5.45),
}
# Donors whose own city started a charging zone inside the post-period
TREATED_DONORS = {"POAR": "Portsmouth CAZ charging from 29 Nov 2021"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--refresh", action="store_true", help="redo the weather normalisation")
    args = parser.parse_args()

    engine = get_engine()
    sites = replication_sites(engine)
    hourly = {}
    for i, site in enumerate(sites.itertuples(index=False), start=1):
        started = time.monotonic()
        hourly[site.site_id] = deweathered(engine, site, PAPER_WINDOW, refresh=args.refresh)
        r2 = hourly[site.site_id]["r2_oob"].iloc[0]
        print(f"  {i:2d}/{len(sites)} {site.site_id:5s} {site.paper_name:32s} OOB R2 {r2:.2f}"
              f"  ({time.monotonic() - started:.0f} s)")  # fmt: skip
    panel = weekly_panel(hourly, PAPER_WINDOW)
    panel.to_csv(OUT / "weekly_normalised.csv")

    runs = {
        "BIRR": ("BIRR", "roadside", []),
        "BMLD": ("BMLD", "background", []),
        "BIRR without POAR": ("BIRR", "roadside", list(TREATED_DONORS)),
    }
    rows = []
    for name, (treated, site_class, exclude) in runs.items():
        is_donor = (sites["role"] == "donor") & (sites["site_class"] == site_class)
        donors = [d for d in sites.loc[is_donor, "site_id"] if d not in exclude]
        run = estimate(panel, treated, donors, PAPER_WINDOW)
        r = run["result"]
        print(f"\n{name}: {len(run['weeks'])} weeks ({r.t0} before treatment), "
              f"{len(run['donors_used'])} donors, lambda {r.lam:.3g}, "
              f"pre-period RMSE {run['pre_rmse']:.2f} ug/m3")  # fmt: skip
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
