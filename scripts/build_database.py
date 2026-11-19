"""Build the project database: reference tables, monitoring sites, and views.

Run from the project root, with the database container running:
    docker compose up -d --wait
    uv run --env-file .env python scripts/build_database.py            # cached downloads
    uv run --env-file .env python scripts/build_database.py --refresh  # re-download

Everything is rebuilt from data/reference/ and the cached openair site metadata
(data/raw/openair/<network>/metadata.RData), in one transaction.
"""

import argparse
from pathlib import Path

import pandas as pd

from caz.database import build_database, panel_summary
from caz.db import get_engine
from caz.openair import NETWORKS, DownloadError, download, metadata_url, read_metadata

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "openair"
FEASIBILITY_SITES = ROOT / "data" / "processed" / "feasibility_sites.csv"

LOCATION_COUNTS = """
    SELECT l.zone_id, z.include, l.zone_location, count(*) AS n
    FROM aq.site_zone_location l
    JOIN aq.sites s USING (network, site_id)
    JOIN ref.zones z USING (zone_id)
    WHERE EXISTS (SELECT 1 FROM aq.site_parameters p
                  WHERE (p.network, p.site_id) = (s.network, s.site_id) AND p.parameter = 'NO2')
    GROUP BY 1, 2, 3
"""


def read_all_metadata(refresh: bool) -> pd.DataFrame:
    """Site metadata of every network that can be read; unavailable ones are reported."""
    parts = []
    for name, network in NETWORKS.items():
        try:
            path = download(metadata_url(network), RAW / name / "metadata.RData", refresh)
        except DownloadError as e:
            print(f"!! {name} unavailable, left out: {e}")
            continue
        parts.append(read_metadata(path, network))
    return pd.concat(parts, ignore_index=True)


def compare_with_feasibility(engine) -> None:
    """Check PostGIS labels against the Python labels in the feasibility output."""
    if not FEASIBILITY_SITES.exists():
        print("(no feasibility output to compare with; run scripts/feasibility.py)")
        return
    python = pd.read_csv(FEASIBILITY_SITES)
    python = python[python["area_type"] == "zone"].rename(columns={"area_id": "zone_id"})
    with engine.connect() as conn:
        postgis = pd.read_sql("SELECT * FROM aq.site_zone_location", conn)
    both = python.merge(postgis, on=["network", "site_id", "zone_id"], suffixes=("_py", "_db"))
    differ = both[both["zone_location_py"] != both["zone_location_db"]]
    gap = (both["distance_to_zone_m_py"] - both["distance_to_zone_m_db"]).abs()
    print(
        f"\nPostGIS vs Python (feasibility output): {len(both)} site-zone pairs compared, "
        f"{len(differ)} labels differ, largest distance difference {gap.max():.0f} m"
    )
    if len(differ):
        print(differ[["zone_id", "site_id", "zone_location_py", "zone_location_db"]])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--refresh", action="store_true", help="re-download site metadata")
    args = parser.parse_args()

    engine = get_engine()
    counts = build_database(engine, read_all_metadata(args.refresh))
    print("Built:", ", ".join(f"{table} {n}" for table, n in counts.items()))

    with engine.connect() as conn:
        counts = pd.read_sql(LOCATION_COUNTS, conn)
    table = counts.pivot_table(
        index=["zone_id", "include"], columns="zone_location", values="n", fill_value=0
    ).astype(int)
    print("\nNO2 instruments within 5 km of each zone boundary (all years, usable or not):")
    print(table[["inside", "edge", "outside"]].to_string())
    compare_with_feasibility(engine)

    print("\nPanels (cleaning rules in sql/views/04_panels.sql):")
    for name, value in panel_summary(engine).items():
        print(f"  {name:24s} {value:>10,}")


if __name__ == "__main__":
    main()
