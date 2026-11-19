"""Monitoring sites from the openair metadata: tidy, validate, and insert into Postgres.

Input is the metadata of one or more networks as returned by
caz.openair.read_metadata (one row per site x pollutant x measurement period).
Output is two tables:
- sites: one row per (network, site_id), with its site class, location and the
  instrument it belongs to (canon_*: copies in a second network point to the
  kept record; see caz.feasibility.duplicate_mapping);
- parameters: one row per (network, site_id, parameter, start_date).
"""

import pandas as pd
import pandera.pandas as pa
from sqlalchemy import Connection, text

from caz.db import records
from caz.feasibility import SITE_CLASSES, duplicate_mapping, site_class
from caz.openair import NETWORKS

UNKNOWN_LOCATION_TYPE = "unknown unknown"  # openair's value when the type is not recorded
SITE_COLUMNS = ["site_name", "location_type", "latitude", "longitude", "provider_code"]
# British Isles: the AURN includes Mace Head, on the west coast of Ireland
LAT_RANGE = (49.5, 61.0)
LON_RANGE = (-11.0, 2.0)

SITES_SCHEMA = pa.DataFrameSchema(
    {
        "network": pa.Column("string", pa.Check.isin(list(NETWORKS))),
        "site_id": pa.Column("string", pa.Check.str_length(min_value=1)),
        "site_name": pa.Column("string"),
        "location_type": pa.Column("string"),
        "site_class": pa.Column("string", pa.Check.isin([*set(SITE_CLASSES.values()), "other"])),
        "latitude": pa.Column(float, pa.Check.in_range(*LAT_RANGE)),
        "longitude": pa.Column(float, pa.Check.in_range(*LON_RANGE)),
        "provider_code": pa.Column("string", nullable=True),
        "canon_network": pa.Column("string"),
        "canon_site_id": pa.Column("string"),
    },
    checks=pa.Check(
        lambda s: (s["network"] == "LMAM") == s["provider_code"].notna(),
        error="LMAM sites, and only they, have a provider code",
    ),
    unique=["network", "site_id"],
    strict=True,
)

PARAMETERS_SCHEMA = pa.DataFrameSchema(
    {
        "network": pa.Column("string"),
        "site_id": pa.Column("string"),
        "parameter": pa.Column("string", pa.Check.str_length(min_value=1)),
        "start_date": pa.Column("datetime64[ns]"),
        "end_date": pa.Column("datetime64[ns]", nullable=True),  # NaT = still running
    },
    checks=pa.Check(
        lambda p: p["end_date"].isna() | (p["end_date"] >= p["start_date"]),
        error="a measurement period cannot end before it starts",
    ),
    unique=["network", "site_id", "parameter", "start_date"],
    strict=True,
)


def drop_conflicting_copies(metadata: pd.DataFrame) -> pd.DataFrame:
    """Resolve sites listed with conflicting names, types or coordinates.

    Some sites appear twice in a network's file with slightly different details
    (e.g. LMAM EL3, once with location type "unknown unknown"). Where a site has
    more than one version, versions with an unknown location type are dropped.
    Raises ValueError if a conflict remains, rather than picking one silently.
    """
    keys = ["network", "site_id"]
    versions = metadata.drop_duplicates(keys + SITE_COLUMNS)
    n_versions = versions.groupby(keys)["site_name"].transform("size")
    drop = versions[(n_versions > 1) & (versions["location_type"] == UNKNOWN_LOCATION_TYPE)]
    out = metadata.merge(drop[keys + SITE_COLUMNS], how="left", indicator=True)
    out = out[out["_merge"] == "left_only"].drop(columns="_merge")
    still = out.drop_duplicates(keys + SITE_COLUMNS).groupby(keys).size()
    if (still > 1).any():
        raise ValueError(f"sites with conflicting details: {list(still[still > 1].index)}")
    return out.reset_index(drop=True)


def site_tables(metadata: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(sites, parameters) tables from openair metadata, validated."""
    metadata = drop_conflicting_copies(metadata)
    keys = ["network", "site_id"]
    sites = metadata.drop_duplicates(keys)[keys + SITE_COLUMNS].reset_index(drop=True)
    sites["site_class"] = sites["location_type"].map(site_class)
    sites = sites.merge(duplicate_mapping(sites, list(NETWORKS)), on=keys)
    text_columns = [c for c in sites.columns if c not in ("latitude", "longitude")]
    sites = sites.astype(dict.fromkeys(text_columns, "string"))

    parameters = metadata[[*keys, "parameter", "start_date", "end_date"]].drop_duplicates()
    parameters = parameters.astype(dict.fromkeys([*keys, "parameter"], "string"))
    for column in ["start_date", "end_date"]:
        parameters[column] = parameters[column].astype("datetime64[ns]")
    return SITES_SCHEMA.validate(sites), PARAMETERS_SCHEMA.validate(parameters)


INSERT_SITES = text("""
    INSERT INTO aq.sites (
        network, site_id, site_name, location_type, site_class, geom, provider_code,
        canon_network, canon_site_id)
    VALUES (
        :network, :site_id, :site_name, :location_type, :site_class,
        ST_SetSRID(ST_MakePoint(:longitude, :latitude), 4326), :provider_code,
        :canon_network, :canon_site_id)
""")

INSERT_PARAMETERS = text("""
    INSERT INTO aq.site_parameters (network, site_id, parameter, start_date, end_date)
    VALUES (:network, :site_id, :parameter, :start_date, :end_date)
""")


def insert_sites(conn: Connection, sites: pd.DataFrame, parameters: pd.DataFrame) -> None:
    """Insert validated site tables (from site_tables) into aq.sites / aq.site_parameters."""
    conn.execute(INSERT_SITES, records(sites))
    conn.execute(INSERT_PARAMETERS, records(parameters))
