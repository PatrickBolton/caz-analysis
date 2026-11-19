"""Feasibility checks: are there enough NO2 monitors near each zone and control city?

All functions here are pure (tables in, tables out) so they can be tested offline.
Downloading lives in `caz.openair`; orchestration in `scripts/feasibility.py`.

Year conventions: a calendar year counts as *before* a date if it ends before the
date's year starts (year < date.year), and *after* if it starts after the date's
year ends (year > date.year). The year containing the date is excluded from both,
because it is partly treated.
"""

import datetime as dt

import pandas as pd
from shapely.geometry.base import BaseGeometry

from caz.boundaries import locate
from caz.geo import haversine_km

MIN_CAPTURE = 0.75  # UK convention for a valid annual mean
MIN_YEARS_BEFORE = 3
MIN_YEARS_AFTER = 1
SEARCH_RADIUS_KM = 5.0
DUPLICATE_SITE_KM = 0.01  # sites in different networks this close = same instrument
RELOCATION_KM = 0.3  # a site replaced by another this close may be a relocation
RELOCATION_GAP_DAYS = 366  # ... if the new one starts within this time of the old one ending
MIN_DONOR_DISTANCE_KM = 15.0  # control city closer than this to a zone is not a donor

# openair "location_type" -> our coarser site class
SITE_CLASSES = {
    "Urban Traffic": "roadside",
    "Urban Background": "background",
    "Suburban Background": "background",
    "Urban Industrial": "industrial",
    "Suburban Industrial": "industrial",
    "Rural Background": "rural",
}


def site_class(location_type: str) -> str:
    """Map an openair location type to roadside / background / industrial / rural / other."""
    return SITE_CLASSES.get(location_type, "other")


def no2_sites(metadata: pd.DataFrame) -> pd.DataFrame:
    """One row per (network, site) that has ever measured NO2.

    Input: metadata with one row per site x pollutant (possibly several rows per
    pollutant if the instrument changed). Output columns: network, site_id,
    site_name, location_type, site_class, latitude, longitude, no2_start, no2_end
    (NaT end = still running).
    """
    no2 = metadata[metadata["parameter"] == "NO2"]
    grouped = no2.groupby(["network", "site_id"], as_index=False, sort=False)
    sites = grouped.agg(
        site_name=("site_name", "first"),
        location_type=("location_type", "first"),
        latitude=("latitude", "first"),
        longitude=("longitude", "first"),
        no2_start=("start_date", "min"),
        # any open-ended record means the site is still running
        no2_end=("end_date", lambda s: pd.NaT if s.isna().any() else s.max()),
    )
    sites["site_class"] = sites["location_type"].map(site_class)
    return sites


def merge_duplicate_sites(
    sites: pd.DataFrame, network_priority: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge sites that are the same instrument listed in several networks.

    Two sites in *different* networks within DUPLICATE_SITE_KM are treated as one
    instrument, even if their codes differ (e.g. Aberdeen Union Street is ABD7 in
    AURN and ABD3 in SAQN). The highest-priority network's record is kept; if
    several kept records are within range, the copy joins the nearest one.

    Returns (kept_sites, mapping) where mapping has columns network, site_id,
    canon_network, canon_site_id and covers every input site, so annual data from
    all copies can be pooled under the kept record.
    """
    rank = {name: i for i, name in enumerate(network_priority)}
    ordered = sites.assign(_rank=sites["network"].map(rank)).sort_values("_rank", kind="stable")
    kept: list[pd.Series] = []
    mapping = []
    for _, row in ordered.iterrows():
        # (distance, index) of kept sites in other networks close enough to be this instrument
        matches = [
            (d, i)
            for i, k in enumerate(kept)
            if k["network"] != row["network"]
            and (
                d := haversine_km(row["latitude"], row["longitude"], k["latitude"], k["longitude"])
            )
            <= DUPLICATE_SITE_KM
        ]
        if matches:
            canon = kept[min(matches)[1]]  # nearest
        else:
            kept.append(row)
            canon = row
        mapping.append(
            {
                "network": row["network"],
                "site_id": row["site_id"],
                "canon_network": canon["network"],
                "canon_site_id": canon["site_id"],
            }
        )
    mapping_df = pd.DataFrame(mapping)
    kept_sites = pd.DataFrame(kept).drop(columns="_rank").reset_index(drop=True)
    # Measurement period of a merged site = union over all its copies.
    periods = (
        sites.merge(mapping_df, on=["network", "site_id"])
        .groupby(["canon_network", "canon_site_id"])
        .agg(
            no2_start=("no2_start", "min"),
            no2_end=("no2_end", lambda s: pd.NaT if s.isna().any() else s.max()),
        )
    )
    keys = list(zip(kept_sites["network"], kept_sites["site_id"], strict=True))
    kept_sites["no2_start"] = [periods.loc[k, "no2_start"] for k in keys]
    kept_sites["no2_end"] = [periods.loc[k, "no2_end"] for k in keys]
    return kept_sites, mapping_df


def find_relocations(sites: pd.DataFrame) -> pd.DataFrame:
    """Pairs (old, new) where a closed site was replaced by a nearby new one.

    Flags candidates only: whether two series can be joined is an analysis decision.
    Output columns: old_network, old_site_id, new_network, new_site_id, distance_m, gap_days.
    """
    closed = sites[sites["no2_end"].notna()]
    pairs = []
    for _, old in closed.iterrows():
        # cheap pre-filter: 0.01 degrees is ~0.7-1.1 km, comfortably above RELOCATION_KM
        nearby = sites[
            ((sites["latitude"] - old["latitude"]).abs() < 0.01)
            & ((sites["longitude"] - old["longitude"]).abs() < 0.01)
        ]
        for _, new in nearby.iterrows():
            if (new["network"], new["site_id"]) == (old["network"], old["site_id"]):
                continue
            gap = (new["no2_start"] - old["no2_end"]).days
            if not -30 <= gap <= RELOCATION_GAP_DAYS:  # allow a short overlap
                continue
            d = haversine_km(old["latitude"], old["longitude"], new["latitude"], new["longitude"])
            if d <= RELOCATION_KM:
                pairs.append(
                    {
                        "old_network": old["network"],
                        "old_site_id": old["site_id"],
                        "new_network": new["network"],
                        "new_site_id": new["site_id"],
                        "distance_m": round(d * 1000),
                        "gap_days": gap,
                    }
                )
    cols = ["old_network", "old_site_id", "new_network", "new_site_id", "distance_m", "gap_days"]
    return pd.DataFrame(pairs, columns=cols)


def chain_relocations(site_keys: list[tuple[str, str]], relocations: pd.DataFrame) -> list[set]:
    """Group sites into series, joining relocation pairs (union-find over site keys)."""
    parent = {k: k for k in site_keys}

    def root(k: tuple[str, str]) -> tuple[str, str]:
        while parent[k] != k:
            k = parent[k]
        return k

    for _, r in relocations.iterrows():
        a, b = (r["old_network"], r["old_site_id"]), (r["new_network"], r["new_site_id"])
        if a in parent and b in parent:
            parent[root(a)] = root(b)
    groups: dict[tuple[str, str], set] = {}
    for k in site_keys:
        groups.setdefault(root(k), set()).add(k)
    return list(groups.values())


def sites_near(
    sites: pd.DataFrame, areas: pd.DataFrame, radius_km: float = SEARCH_RADIUS_KM
) -> pd.DataFrame:
    """Pair every site with every area (zone or control city) whose centre is within radius.

    `areas` needs columns area_id, area_type, lat, lon. A site near two areas
    appears twice. Adds a distance_km column.
    """
    rows = []
    for _, area in areas.iterrows():
        dist = sites.apply(
            lambda s, a=area: haversine_km(s["latitude"], s["longitude"], a["lat"], a["lon"]),
            axis=1,
        )
        near = sites[dist <= radius_km].assign(
            area_id=area["area_id"], area_type=area["area_type"], distance_km=dist.round(2)
        )
        rows.append(near)
    return pd.concat(rows, ignore_index=True)


def usable_years(annual: pd.DataFrame, min_capture: float = MIN_CAPTURE) -> pd.DataFrame:
    """Table of (network, site_id) -> sorted list of years with capture >= min_capture."""
    ok = annual[annual["capture"] >= min_capture]
    return (
        ok.groupby(["network", "site_id"])["year"]
        .apply(lambda y: sorted({int(v) for v in y}))  # set: merged copies share years
        .rename("usable_years")
        .reset_index()
    )


def years_before(years: list[int], date: dt.date) -> int:
    """Number of years strictly before the calendar year of `date`."""
    return sum(y < date.year for y in years)


def years_after(years: list[int], date: dt.date) -> int:
    """Number of years strictly after the calendar year of `date`."""
    return sum(y > date.year for y in years)


def is_usable(years: list[int], pre_end: dt.date, post_start: dt.date) -> bool:
    """Enough clean years before `pre_end` and after `post_start` for a before/after comparison."""
    return (
        years_before(years, pre_end) >= MIN_YEARS_BEFORE
        and years_after(years, post_start) >= MIN_YEARS_AFTER
    )


def verdict(n_usable_sites: int) -> str:
    """GO with >= 2 usable sites, THIN with 1, NONE with 0."""
    if n_usable_sites >= 2:
        return "GO"
    return "THIN" if n_usable_sites == 1 else "NONE"


# --- combining the pieces (used by scripts/feasibility.py) ------------------


def pool_merged_annual(annual: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    """Relabel annual rows from every copy of a merged site with the kept site's key.

    `mapping` comes from merge_duplicate_sites. Rows for sites not in the mapping
    (no NO2 metadata) are dropped, since they have no location.
    """
    pooled = annual.merge(mapping, on=["network", "site_id"], how="inner")
    return pooled.drop(columns=["network", "site_id"]).rename(
        columns={"canon_network": "network", "canon_site_id": "site_id"}
    )


def describe_pairs(pairs: pd.DataFrame, usable: pd.DataFrame, areas: pd.DataFrame) -> pd.DataFrame:
    """Add usable years, and for zones the before/after counts and usable flags.

    `pairs` comes from sites_near, `usable` from usable_years, and `areas` needs
    decision_date and charging_start (as dates) for zones. Control-city rows get
    NA in the zone-only columns. Flags are pandas "boolean", counts "Int64".
    """
    out = pairs.merge(usable, on=["network", "site_id"], how="left")
    out["usable_years"] = out["usable_years"].map(lambda y: y if isinstance(y, list) else [])
    out["n_usable_years"] = out["usable_years"].map(len)
    out["first_usable_year"] = out["usable_years"].map(lambda y: min(y) if y else pd.NA)
    out["last_usable_year"] = out["usable_years"].map(lambda y: max(y) if y else pd.NA)

    dates = areas.set_index("area_id")[["decision_date", "charging_start"]]
    decision = out["area_id"].map(dates["decision_date"])
    charging = out["area_id"].map(dates["charging_start"])
    is_zone = out["area_type"] == "zone"

    def per_zone_row(fn, dtype: str) -> pd.Series:
        values = [
            fn(y, d, c) if z else pd.NA
            for y, d, c, z in zip(out["usable_years"], decision, charging, is_zone, strict=True)
        ]
        return pd.Series(values, index=out.index, dtype=dtype)

    out["years_before_decision"] = per_zone_row(lambda y, d, c: years_before(y, d), "Int64")
    out["years_before_charging"] = per_zone_row(lambda y, d, c: years_before(y, c), "Int64")
    out["years_after_charging"] = per_zone_row(lambda y, d, c: years_after(y, c), "Int64")
    out["usable_vs_decision"] = per_zone_row(lambda y, d, c: is_usable(y, d, c), "boolean")
    out["usable_vs_charging"] = per_zone_row(lambda y, d, c: is_usable(y, c, c), "boolean")
    return out


def flag_relocations(
    pairs: pd.DataFrame, relocations: pd.DataFrame, mapping: pd.DataFrame
) -> pd.DataFrame:
    """Add columns naming merged copies and possible predecessor/successor sites.

    Values look like "SAQN:ABD3"; several are joined with ";". Empty string = none.
    """
    out = pairs.copy()
    copies = mapping[
        (mapping["network"] != mapping["canon_network"])
        | (mapping["site_id"] != mapping["canon_site_id"])
    ]
    merged: dict[tuple[str, str], list[str]] = {}
    for r in copies.itertuples(index=False):
        merged.setdefault((r.canon_network, r.canon_site_id), []).append(f"{r.network}:{r.site_id}")
    succ: dict[tuple[str, str], list[str]] = {}
    pred: dict[tuple[str, str], list[str]] = {}
    for r in relocations.itertuples(index=False):
        succ.setdefault((r.old_network, r.old_site_id), []).append(
            f"{r.new_network}:{r.new_site_id}"
        )
        pred.setdefault((r.new_network, r.new_site_id), []).append(
            f"{r.old_network}:{r.old_site_id}"
        )
    keys = list(zip(out["network"], out["site_id"], strict=True))
    out["merged_copies"] = [";".join(merged.get(k, [])) for k in keys]
    out["replaced_by"] = [";".join(succ.get(k, [])) for k in keys]
    out["replaces"] = [";".join(pred.get(k, [])) for k in keys]
    return out


def locate_sites(pairs: pd.DataFrame, boundaries: dict[str, BaseGeometry]) -> pd.DataFrame:
    """Add zone_location (inside / edge / outside / unknown) and distance_to_zone_m.

    Only for zone rows; control-city rows get NA. "unknown" = no boundary for that zone.
    """
    out = pairs.copy()
    located = [
        locate(lat, lon, boundaries.get(area)) if kind == "zone" else (pd.NA, pd.NA)
        for lat, lon, area, kind in zip(
            out["latitude"], out["longitude"], out["area_id"], out["area_type"], strict=True
        )
    ]
    out["zone_location"] = pd.Series([loc for loc, _ in located], index=out.index, dtype="string")
    out["distance_to_zone_m"] = pd.Series(
        [d if d is None or d is pd.NA else round(d) for _, d in located], index=out.index
    ).astype("Int64")
    return out


def summarise_zones(
    pairs: pd.DataFrame, areas: pd.DataFrame, relocations: pd.DataFrame
) -> pd.DataFrame:
    """One row per zone: usable site counts, verdicts, and number of usable donor cities.

    `pairs` comes from describe_pairs then locate_sites. usable_inside/edge/outside
    split the usable sites by position relative to the zone boundary (NA if the
    zone has no boundary); verdict_inside uses only the sites inside. A donor city
    is an included control city at least MIN_DONOR_DISTANCE_KM from the zone centre
    with at least one site usable for that zone's dates.
    """
    zones = areas[areas["area_type"] == "zone"]
    controls = areas[areas["area_type"] == "control"]
    control_years = pairs[pairs["area_type"] == "control"].groupby("area_id")["usable_years"]
    rows = []
    for zone in zones.itertuples(index=False):
        z = pairs[pairs["area_id"] == zone.area_id]
        ok = z[z["usable_vs_charging"].fillna(False).astype(bool)]
        ok_dec = z[z["usable_vs_decision"].fillna(False).astype(bool)]
        charging, decision = zone.charging_start, zone.decision_date

        # What-if: join relocated sites into one series each (an analysis decision, Phase 1).
        years_by_key = {(r.network, r.site_id): r.usable_years for r in z.itertuples(index=False)}
        series = chain_relocations(list(years_by_key), relocations)
        chained_ok = sum(
            is_usable(sorted({y for k in s for y in years_by_key[k]}), charging, charging)
            for s in series
        )

        donors, donors_dec = 0, 0
        for c in controls.itertuples(index=False):
            if haversine_km(zone.lat, zone.lon, c.lat, c.lon) < MIN_DONOR_DISTANCE_KM:
                continue  # too close: spillover risk (e.g. Leeds for Bradford)
            years = control_years.get_group(c.area_id) if c.area_id in control_years.groups else []
            donors += any(is_usable(y, charging, charging) for y in years)
            donors_dec += any(is_usable(y, decision, charging) for y in years)

        rows.append(
            {
                "zone_id": zone.area_id,
                "sites_within_radius": len(z),
                "usable_roadside": int((ok["site_class"] == "roadside").sum()),
                "usable_background": int((ok["site_class"] == "background").sum()),
                "usable_other": int((~ok["site_class"].isin(["roadside", "background"])).sum()),
                **split_by_location(ok["zone_location"]),
                "usable_vs_charging": len(ok),
                "usable_vs_decision": len(ok_dec),
                "verdict_vs_charging": verdict(len(ok)),
                "verdict_vs_decision": verdict(len(ok_dec)),
                "usable_vs_charging_if_relocations_joined": chained_ok,
                "donor_cities_vs_charging": donors,
                "donor_cities_vs_decision": donors_dec,
            }
        )
    return pd.DataFrame(rows)


def split_by_location(locations: pd.Series) -> dict:
    """Counts of usable sites inside / at the edge of / outside a zone, and the inside verdict."""
    if (locations == "unknown").any():
        return {
            "usable_inside": pd.NA,
            "usable_edge": pd.NA,
            "usable_outside": pd.NA,
            "verdict_inside": "NO BOUNDARY",
        }
    counts = {loc: int((locations == loc).sum()) for loc in ("inside", "edge", "outside")}
    return {
        "usable_inside": counts["inside"],
        "usable_edge": counts["edge"],
        "usable_outside": counts["outside"],
        "verdict_inside": verdict(counts["inside"]),
    }
