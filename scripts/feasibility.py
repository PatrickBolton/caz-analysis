"""Phase 0.4 feasibility check: usable NO2 monitors near each zone and control city.

Run from the project root:
    uv run python scripts/feasibility.py            # uses cached downloads
    uv run python scripts/feasibility.py --refresh  # re-download everything

Outputs:
    data/processed/feasibility_sites.csv    one row per (site, nearby area)
    data/processed/feasibility_summary.csv  one row per zone
"""

import argparse
import datetime as dt
from pathlib import Path

import pandas as pd

from caz import feasibility as f
from caz.boundaries import EDGE_M, read_boundaries
from caz.openair import (
    NETWORKS,
    DownloadError,
    annual_summary_url,
    download,
    metadata_url,
    read_annual_summary,
    read_metadata,
)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "openair"
PROCESSED = ROOT / "data" / "processed"
REFERENCE = ROOT / "data" / "reference"
BOUNDARIES = REFERENCE / "zone_boundaries.geojson"  # built by scripts/build_zone_boundaries.py


def load_areas() -> pd.DataFrame:
    """Zones and included control cities as one table: area_id, area_type, lat, lon, dates."""
    zones = pd.read_csv(REFERENCE / "zones.csv", parse_dates=["charging_start", "decision_date"])
    controls = pd.read_csv(REFERENCE / "control_cities.csv")
    controls = controls[controls["include"]]
    return pd.concat(
        [
            pd.DataFrame(
                {
                    "area_id": zones["zone_id"],
                    "area_type": "zone",
                    "lat": zones["approx_centre_lat"],
                    "lon": zones["approx_centre_lon"],
                    "decision_date": zones["decision_date"].dt.date,
                    "charging_start": zones["charging_start"].dt.date,
                }
            ),
            pd.DataFrame(
                {
                    "area_id": controls["city_id"],
                    "area_type": "control",
                    "lat": controls["approx_centre_lat"],
                    "lon": controls["approx_centre_lon"],
                }
            ),
        ],
        ignore_index=True,
    )


def fetch_network(name: str, years: range, refresh: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download (or read from cache) one network's metadata and annual NO2 summaries."""
    network = NETWORKS[name]
    folder = RAW / name
    meta_path = download(metadata_url(network), folder / "metadata.RData", refresh)
    metadata = read_metadata(meta_path, network)
    annual_parts = []
    for year in years:
        try:
            path = download(
                annual_summary_url(network, year), folder / f"annual_{year}.rds", refresh
            )
        except DownloadError as e:
            print(f"    {name} {year}: no annual summary ({e})")
            continue
        annual_parts.append(read_annual_summary(path, name))
    return metadata, pd.concat(annual_parts, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--refresh", action="store_true", help="re-download cached files")
    parser.add_argument("--first-year", type=int, default=2010)
    parser.add_argument("--last-year", type=int, default=dt.date.today().year - 1)
    args = parser.parse_args()
    years = range(args.first_year, args.last_year + 1)

    areas = load_areas()
    metadata_parts, annual_parts, unavailable = [], [], []
    for name in NETWORKS:
        print(f"Fetching {name} ...")
        try:
            metadata, annual = fetch_network(name, years, args.refresh)
        except DownloadError as e:
            print(f"    !! {name} UNAVAILABLE: {e}")
            unavailable.append(name)
            continue
        metadata_parts.append(metadata)
        annual_parts.append(annual)

    all_sites = f.no2_sites(pd.concat(metadata_parts))
    sites, mapping = f.merge_duplicate_sites(all_sites, list(NETWORKS))
    annual = f.pool_merged_annual(pd.concat(annual_parts, ignore_index=True), mapping)
    usable = f.usable_years(annual)
    relocations = f.find_relocations(sites)
    pairs = f.describe_pairs(f.sites_near(sites, areas), usable, areas)
    pairs = f.flag_relocations(pairs, relocations, mapping)
    pairs = f.locate_sites(pairs, read_boundaries(BOUNDARIES))
    summary = f.summarise_zones(pairs, areas, relocations)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    out = pairs.assign(usable_years=pairs["usable_years"].map(lambda y: ";".join(map(str, y))))
    out = out.sort_values(["area_type", "area_id", "distance_km"], ascending=[False, True, True])
    out.to_csv(PROCESSED / "feasibility_sites.csv", index=False)
    summary.to_csv(PROCESSED / "feasibility_summary.csv", index=False)

    print(f"\nYears {years.start}-{years.stop - 1}; capture >= {f.MIN_CAPTURE:.0%}; "
          f"radius {f.SEARCH_RADIUS_KM:g} km; usable = >= {f.MIN_YEARS_BEFORE} years before "
          f"and >= {f.MIN_YEARS_AFTER} after.\n")  # fmt: skip
    short = {
        "zone_id": "zone",
        "sites_within_radius": "sites",
        "usable_roadside": "road",
        "usable_background": "bkgd",
        "usable_other": "other",
        "usable_vs_charging": "usable",
        "usable_inside": "in",
        "usable_edge": "edge",
        "usable_outside": "out",
        "verdict_inside": "verdict_in",
        "verdict_vs_charging": "verdict",
        "usable_vs_charging_if_relocations_joined": "if_joined",
        "usable_vs_decision": "usable_dec",
        "verdict_vs_decision": "verdict_dec",
        "donor_cities_vs_charging": "donors",
        "donor_cities_vs_decision": "donors_dec",
    }
    print("usable/verdict: before-period ends at charging start; *_dec: ends at council decision")
    print(f"in/edge/out: usable sites inside the zone boundary, outside but within "
          f"{EDGE_M:g} m of it, or further; verdict_in counts only sites inside.")  # fmt: skip
    print(summary[list(short)].rename(columns=short).to_string(index=False))
    control_pairs = pairs[pairs["area_type"] == "control"]
    print("\nControl cities: sites within radius / sites with >= 8 usable years")
    for city, g in control_pairs.groupby("area_id"):
        print(f"  {city:14s} {len(g):3d} / {(g['n_usable_years'] >= 8).sum():3d}")
    missing = set(areas.loc[areas["area_type"] == "control", "area_id"]) - set(
        control_pairs["area_id"]
    )
    for city in sorted(missing):
        print(f"  {city:14s}   0 /   0")
    if unavailable:
        print(f"\n!! WARNING: networks unavailable: {', '.join(unavailable)}. "
              "Counts are LOWER BOUNDS; re-run with --refresh when they are back.")  # fmt: skip


if __name__ == "__main__":
    main()
