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
from caz.geo import haversine_km
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


def describe_site_area_pairs(pairs: pd.DataFrame, areas: pd.DataFrame) -> pd.DataFrame:
    """Add usable-year counts relative to each zone's decision and charging dates."""
    dates = areas.set_index("area_id")[["decision_date", "charging_start"]]
    out = pairs.copy()
    out["n_usable_years"] = out["usable_years"].map(len)
    out["first_usable_year"] = out["usable_years"].map(lambda y: min(y) if y else pd.NA)
    out["last_usable_year"] = out["usable_years"].map(lambda y: max(y) if y else pd.NA)
    is_zone = out["area_type"] == "zone"
    for col in [
        "years_before_decision",
        "years_before_charging",
        "years_after_charging",
        "usable_vs_decision",
        "usable_vs_charging",
    ]:
        out[col] = pd.NA
    for i, row in out[is_zone].iterrows():
        decision, charging = dates.loc[row["area_id"]]
        years = row["usable_years"]
        out.at[i, "years_before_decision"] = f.years_before(years, decision)
        out.at[i, "years_before_charging"] = f.years_before(years, charging)
        out.at[i, "years_after_charging"] = f.years_after(years, charging)
        out.at[i, "usable_vs_decision"] = f.is_usable(years, decision, charging)
        out.at[i, "usable_vs_charging"] = f.is_usable(years, charging, charging)
    return out


def flag_relocations(
    pairs: pd.DataFrame, relocations: pd.DataFrame, mapping: pd.DataFrame
) -> pd.DataFrame:
    """Add columns naming merged copies and possible predecessor/successor sites."""
    out = pairs.copy()
    copies = mapping[
        (mapping["network"] != mapping["canon_network"])
        | (mapping["site_id"] != mapping["canon_site_id"])
    ]
    merged = copies.groupby(["canon_network", "canon_site_id"]).apply(
        lambda g: ";".join(g["network"] + ":" + g["site_id"]), include_groups=False
    )
    key = list(zip(out["network"], out["site_id"], strict=True))
    out["merged_copies"] = [merged.get(k, "") for k in key]
    succ = {
        (r.old_network, r.old_site_id): f"{r.new_network}:{r.new_site_id}"
        for r in relocations.itertuples()
    }
    pred = {
        (r.new_network, r.new_site_id): f"{r.old_network}:{r.old_site_id}"
        for r in relocations.itertuples()
    }
    out["replaced_by"] = [succ.get(k, "") for k in key]
    out["replaces"] = [pred.get(k, "") for k in key]
    return out


def summarise_zones(
    sites: pd.DataFrame, areas: pd.DataFrame, relocations: pd.DataFrame
) -> pd.DataFrame:
    """One row per zone: usable site counts, verdicts, and number of usable donor cities."""
    controls = areas[areas["area_type"] == "control"]
    control_sites = sites[sites["area_type"] == "control"]
    rows = []
    for _, zone in areas[areas["area_type"] == "zone"].iterrows():
        z = sites[sites["area_id"] == zone["area_id"]]
        ok = z[z["usable_vs_charging"] == True]  # noqa: E712 (column holds True/False/NA)
        ok_dec = z[z["usable_vs_decision"] == True]  # noqa: E712
        # What-if: join relocated sites into one series each (an analysis decision, Phase 1).
        years_by_key = {(r.network, r.site_id): r.usable_years for r in z.itertuples(index=False)}
        series = f.chain_relocations(list(years_by_key), relocations)
        chained_years = [sorted({y for k in s for y in years_by_key[k]}) for s in series]
        chained_ok = sum(
            f.is_usable(y, zone["charging_start"], zone["charging_start"]) for y in chained_years
        )
        donors, donors_dec = 0, 0
        for _, c in controls.iterrows():
            if haversine_km(zone["lat"], zone["lon"], c["lat"], c["lon"]) < f.MIN_DONOR_DISTANCE_KM:
                continue  # too close: spillover risk (e.g. Leeds for Bradford)
            years = control_sites.loc[control_sites["area_id"] == c["area_id"], "usable_years"]
            charging, decision = zone["charging_start"], zone["decision_date"]
            donors += any(f.is_usable(y, charging, charging) for y in years)
            donors_dec += any(f.is_usable(y, decision, charging) for y in years)
        rows.append(
            {
                "zone_id": zone["area_id"],
                "sites_within_radius": len(z),
                "usable_roadside": (ok["site_class"] == "roadside").sum(),
                "usable_background": (ok["site_class"] == "background").sum(),
                "usable_other": (~ok["site_class"].isin(["roadside", "background"])).sum(),
                "usable_vs_charging": len(ok),
                "usable_vs_decision": len(ok_dec),
                "verdict_vs_charging": f.verdict(len(ok)),
                "verdict_vs_decision": f.verdict(len(ok_dec)),
                "usable_vs_charging_if_relocations_joined": chained_ok,
                "donor_cities_vs_charging": donors,
                "donor_cities_vs_decision": donors_dec,
            }
        )
    return pd.DataFrame(rows)


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
    # Pool annual data from every copy of a merged site under its kept record.
    annual = pd.concat(annual_parts, ignore_index=True).merge(mapping, on=["network", "site_id"])
    annual = annual.drop(columns=["network", "site_id"]).rename(
        columns={"canon_network": "network", "canon_site_id": "site_id"}
    )
    usable = f.usable_years(annual)
    relocations = f.find_relocations(sites)
    pairs = f.sites_near(sites, areas).merge(usable, on=["network", "site_id"], how="left")
    pairs["usable_years"] = pairs["usable_years"].map(lambda y: y if isinstance(y, list) else [])
    pairs = describe_site_area_pairs(pairs, areas)
    pairs = flag_relocations(pairs, relocations, mapping)
    summary = summarise_zones(pairs, areas, relocations)

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
        "verdict_vs_charging": "verdict",
        "usable_vs_charging_if_relocations_joined": "if_joined",
        "usable_vs_decision": "usable_dec",
        "verdict_vs_decision": "verdict_dec",
        "donor_cities_vs_charging": "donors",
        "donor_cities_vs_decision": "donors_dec",
    }
    print("usable/verdict: before-period ends at charging start; *_dec: ends at council decision")
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
