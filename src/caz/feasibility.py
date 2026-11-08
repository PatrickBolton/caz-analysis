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
    AURN and ABD3 in SAQN). The highest-priority network's record is kept.

    Returns (kept_sites, mapping) where mapping has columns network, site_id,
    canon_network, canon_site_id and covers every input site, so annual data from
    all copies can be pooled under the kept record.
    """
    rank = {name: i for i, name in enumerate(network_priority)}
    ordered = sites.assign(_rank=sites["network"].map(rank)).sort_values("_rank", kind="stable")
    kept: list[pd.Series] = []
    mapping = []
    for _, row in ordered.iterrows():
        canon = next(
            (
                k
                for k in kept
                if k["network"] != row["network"]
                and haversine_km(row["latitude"], row["longitude"], k["latitude"], k["longitude"])
                <= DUPLICATE_SITE_KM
            ),
            None,
        )
        if canon is None:
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
