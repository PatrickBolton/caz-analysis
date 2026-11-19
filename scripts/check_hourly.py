"""Check the loaded hourly NO2 against openair's own annual statistics.

Run from the project root, after scripts/load_hourly.py:
    uv run --env-file .env python scripts/check_hourly.py

openair publishes, per network and year, each site's annual mean NO2 and data
capture (data/raw/openair/<network>/annual_<year>.rds, cached by
scripts/feasibility.py). Annual means computed from our hourly rows should match
them: a mistake in timestamps, units or parsing would show up here.
"""

from pathlib import Path

import pandas as pd

from caz.db import get_engine
from caz.openair import read_annual_summary

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "openair"
TOLERANCE = 0.5  # ug/m3: summaries are rounded, and late data revisions shift means slightly

OURS = """
    SELECT network, site_id, extract(year FROM measured_at AT TIME ZONE 'UTC')::int AS year,
           avg(no2) AS mean_ours, count(no2) AS hours_ours
    FROM obs.hourly
    WHERE no2 IS NOT NULL
    GROUP BY 1, 2, 3
"""


def main() -> None:
    with get_engine().connect() as conn:
        ours = pd.read_sql(OURS, conn)
    theirs = pd.concat(
        [
            read_annual_summary(path, path.parent.name)
            for path in sorted(RAW.glob("*/annual_*.rds"))
        ],
        ignore_index=True,
    ).astype({"year": int})
    both = ours.merge(theirs, on=["network", "site_id", "year"])
    both["diff"] = both["mean_ours"] - both["mean"]
    # capture is relative to all hours in the year
    hours_in_year = both["year"].map(lambda y: 8784 if y % 4 == 0 else 8760)
    both["capture_ours"] = both["hours_ours"] / hours_in_year
    off = both[both["diff"].abs() > TOLERANCE]

    print(f"Site-years with NO2 in both: {len(both)} (ours: {len(ours)})")
    diff = both["diff"]
    print(f"Annual mean difference (ours - openair), ug/m3: median {diff.median():+.3f}, "
          f"max |diff| {diff.abs().max():.3f}")  # fmt: skip
    gap = (both["capture_ours"] - both["capture"]).abs()
    print(f"Data capture difference: median {gap.median():.4f}, max {gap.max():.4f}")
    print(f"Site-years differing by more than {TOLERANCE} ug/m3: {len(off)}")
    if len(off):
        columns = ["network", "site_id", "year", "mean_ours", "mean", "capture_ours", "capture"]
        print(off.sort_values("diff", key=abs, ascending=False)[columns].head(20).round(3))


if __name__ == "__main__":
    main()
