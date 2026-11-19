"""Reference data (zones, control cities, zone boundaries): read, validate, load into Postgres.

The files in data/reference/ are hand-made or built by scripts, so each is checked
against a pandera schema before it is written to the database. The database
tables (sql/schema/01_reference.sql) repeat the key rules as constraints, so a bad
row is stopped even if it bypasses this module. The database is built by
caz.database.build_database.
"""

import json
from pathlib import Path

import pandas as pd
import pandera.pandas as pa
import shapely
from shapely.geometry import MultiPolygon, shape
from shapely.geometry.base import BaseGeometry
from sqlalchemy import Connection, text

from caz.db import records

REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_DIR = REPO_ROOT / "data" / "reference"

GB_LAT = (49.8, 60.9)  # rough bounding box for Great Britain
GB_LON = (-8.7, 1.8)
DATE_FORMAT = "%Y-%m-%d"


def _text(nullable: bool = False, **kwargs) -> pa.Column:
    """A text column (pandas 'string' dtype)."""
    return pa.Column("string", nullable=nullable, **kwargs)


def _url(nullable: bool = False) -> pa.Column:
    return _text(nullable, checks=pa.Check.str_startswith("https://"))


ZONES_SCHEMA = pa.DataFrameSchema(
    {
        "zone_id": _text(unique=True, checks=pa.Check.str_matches(r"^[a-z_]+$")),
        "city": _text(),
        "country": _text(checks=pa.Check.isin(["England", "Scotland"])),
        "scheme_type": _text(checks=pa.Check.isin(["charging_caz", "lez"])),
        "class": _text(checks=pa.Check.isin(["B", "C", "D", "LEZ"])),
        "charging_start": pa.Column("datetime64[ns]"),
        "decision_date": pa.Column("datetime64[ns]"),
        "start_date_announced": pa.Column("datetime64[ns]"),
        "includes_private_cars": pa.Column(bool),
        "approx_centre_lat": pa.Column(float, pa.Check.in_range(*GB_LAT)),
        "approx_centre_lon": pa.Column(float, pa.Check.in_range(*GB_LON)),
        "source_url": _url(),
        "decision_source_url": _url(),
        "include": pa.Column(bool),
        "exclusion_reason": _text(nullable=True),
        "notes": _text(nullable=True),
    },
    checks=[
        pa.Check(
            lambda z: (
                (z["decision_date"] <= z["start_date_announced"])
                & (z["start_date_announced"] < z["charging_start"])
            ),
            error="dates must run decision <= announced < charging start",
        ),
        pa.Check(
            lambda z: z["include"] == z["exclusion_reason"].isna(),
            error="a zone has an exclusion_reason exactly when include is false",
        ),
        pa.Check(
            lambda z: z["includes_private_cars"] == z["class"].isin(["D", "LEZ"]),
            error="Class D and LEZ cover private cars; B and C do not",
        ),
    ],
    strict=True,  # no missing or unexpected columns
)

CONTROL_CITIES_SCHEMA = pa.DataFrameSchema(
    {
        "city_id": _text(unique=True, checks=pa.Check.str_matches(r"^[a-z_]+$")),
        "city": _text(),
        "country": _text(checks=pa.Check.isin(["England", "Wales", "Scotland"])),
        "include": pa.Column(bool),
        "role": _text(checks=pa.Check.isin(["control", "never_charged", "excluded"])),
        "approx_centre_lat": pa.Column(float, pa.Check.in_range(*GB_LAT)),
        "approx_centre_lon": pa.Column(float, pa.Check.in_range(*GB_LON)),
        "exclusion_reason": _text(nullable=True),
        "notes": _text(nullable=True),
        "source_url": _url(nullable=True),
    },
    checks=[
        pa.Check(
            lambda c: c["include"] == c["exclusion_reason"].isna(),
            error="a city has an exclusion_reason exactly when include is false",
        ),
        pa.Check(
            lambda c: c["include"] == (c["role"] != "excluded"),
            error="role is 'excluded' exactly when include is false",
        ),
    ],
    strict=True,
)

BOUNDARIES_SCHEMA = pa.DataFrameSchema(
    {
        "zone_id": _text(unique=True),
        "source": _text(checks=pa.Check.isin(["council", "openstreetmap"])),
        "source_url": _url(),
        "licence": _text(),
        "retrieved": pa.Column("datetime64[ns]"),
        "area_km2": pa.Column(float, pa.Check.gt(0)),
        "geometry": pa.Column(
            object,
            checks=pa.Check(
                lambda g: g.map(
                    lambda x: isinstance(x, MultiPolygon) and x.is_valid and not x.is_empty
                ),
                error="geometry must be a valid, non-empty (multi)polygon",
            ),
        ),
    },
    strict=True,
)


REPLICATION_SITES_SCHEMA = pa.DataFrameSchema(
    {
        "study": _text(checks=pa.Check.str_matches(r"^[a-z]+[0-9]{4}$")),
        "site_id": _text(),
        "network": _text(checks=pa.Check.isin(["AURN", "AQE", "SAQN", "LMAM"])),
        "role": _text(checks=pa.Check.isin(["treated", "donor"])),
        "site_class": _text(checks=pa.Check.isin(["roadside", "background"])),
        "paper_name": _text(),
        "paper_location": _text(),
        "source": _text(),
    },
    unique=["study", "network", "site_id"],
    strict=True,
)


def _read_csv(path: Path, date_columns: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    for column in date_columns:  # strict: a date in any other format is an error
        df[column] = pd.to_datetime(df[column], format=DATE_FORMAT).astype("datetime64[ns]")
    return df


def read_zones(path: Path = REFERENCE_DIR / "zones.csv") -> pd.DataFrame:
    """zones.csv, validated."""
    dates = ["charging_start", "decision_date", "start_date_announced"]
    return ZONES_SCHEMA.validate(_read_csv(path, dates))


def read_control_cities(path: Path = REFERENCE_DIR / "control_cities.csv") -> pd.DataFrame:
    """control_cities.csv, validated."""
    return CONTROL_CITIES_SCHEMA.validate(_read_csv(path, []))


def _as_multipolygon(geometry: BaseGeometry) -> BaseGeometry:
    return MultiPolygon([geometry]) if geometry.geom_type == "Polygon" else geometry


def read_zone_boundaries(
    path: Path = REFERENCE_DIR / "zone_boundaries.geojson",
    zone_ids: pd.Series | None = None,
) -> pd.DataFrame:
    """zone_boundaries.geojson as a table (one row per zone), validated.

    Polygons become one-part MultiPolygons, so every zone has the same geometry type.
    If `zone_ids` is given, every boundary must belong to one of those zones.
    """
    features = json.loads(path.read_text())["features"]
    df = pd.DataFrame(
        [{**f["properties"], "geometry": _as_multipolygon(shape(f["geometry"]))} for f in features]
    ).astype({"zone_id": "string", "source": "string", "source_url": "string"})
    df = df.astype({"licence": "string", "area_km2": float})
    df["retrieved"] = pd.to_datetime(df["retrieved"], format=DATE_FORMAT).astype("datetime64[ns]")
    df = BOUNDARIES_SCHEMA.validate(df)
    if zone_ids is not None:
        unknown = set(df["zone_id"]) - set(zone_ids)
        if unknown:
            raise ValueError(f"boundaries for unknown zones: {sorted(unknown)}")
    return df


INSERT_ZONES = text("""
    INSERT INTO ref.zones (
        zone_id, city, country, scheme_type, class, charging_start, decision_date,
        start_date_announced, includes_private_cars, include, exclusion_reason, centre,
        source_url, decision_source_url, notes)
    VALUES (
        :zone_id, :city, :country, :scheme_type, :class, :charging_start, :decision_date,
        :start_date_announced, :includes_private_cars, :include, :exclusion_reason,
        ST_SetSRID(ST_MakePoint(:approx_centre_lon, :approx_centre_lat), 4326),
        :source_url, :decision_source_url, :notes)
""")

INSERT_CONTROL_CITIES = text("""
    INSERT INTO ref.control_cities (
        city_id, city, country, include, role, exclusion_reason, centre, notes, source_url)
    VALUES (
        :city_id, :city, :country, :include, :role, :exclusion_reason,
        ST_SetSRID(ST_MakePoint(:approx_centre_lon, :approx_centre_lat), 4326),
        :notes, :source_url)
""")

INSERT_BOUNDARIES = text("""
    INSERT INTO ref.zone_boundaries (
        zone_id, geom, source, source_url, licence, retrieved, area_km2)
    VALUES (
        :zone_id, ST_GeomFromWKB(decode(:wkb, 'hex'), 4326), :source, :source_url, :licence,
        :retrieved, :area_km2)
""")


def read_replication_sites(reference_dir: Path = REFERENCE_DIR) -> pd.DataFrame:
    """All replication_<study>.csv files as one validated table, with a study column."""
    parts = [
        pd.read_csv(path).assign(study=path.stem.removeprefix("replication_"))
        for path in sorted(reference_dir.glob("replication_*.csv"))
    ]
    df = pd.concat(parts, ignore_index=True).astype("string")
    return REPLICATION_SITES_SCHEMA.validate(df[list(REPLICATION_SITES_SCHEMA.columns)])


INSERT_REPLICATION_SITES = text("""
    INSERT INTO ref.replication_sites (
        study, network, site_id, role, site_class, paper_name, paper_location, source)
    VALUES (
        :study, :network, :site_id, :role, :site_class, :paper_name, :paper_location, :source)
""")


def read_reference(reference_dir: Path = REFERENCE_DIR) -> dict[str, pd.DataFrame]:
    """The reference files, validated: zones, control cities, boundaries, replication sites."""
    zones = read_zones(reference_dir / "zones.csv")
    return {
        "replication_sites": read_replication_sites(reference_dir),
        "zones": zones,
        "control_cities": read_control_cities(reference_dir / "control_cities.csv"),
        "zone_boundaries": read_zone_boundaries(
            reference_dir / "zone_boundaries.geojson", zone_ids=zones["zone_id"]
        ),
    }


def insert_reference(conn: Connection, reference: dict[str, pd.DataFrame]) -> None:
    """Insert validated reference tables (from read_reference) into the ref.* tables."""
    boundaries = reference["zone_boundaries"].copy()
    # WKB (binary) keeps every coordinate exactly; hex so it travels as text
    wkb = shapely.to_wkb(boundaries.pop("geometry").to_numpy(), hex=True)
    conn.execute(INSERT_ZONES, records(reference["zones"]))
    conn.execute(INSERT_CONTROL_CITIES, records(reference["control_cities"]))
    conn.execute(INSERT_BOUNDARIES, records(boundaries.assign(wkb=wkb)))
    conn.execute(INSERT_REPLICATION_SITES, records(reference["replication_sites"]))
