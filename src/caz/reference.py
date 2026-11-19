"""Reference data (zones, control cities, zone boundaries): read, validate, load into Postgres.

The files in data/reference/ are hand-made or built by scripts, so each is checked
against a pandera schema before it is written to the database. The database
tables (sql/schema/reference.sql) repeat the key rules as constraints, so a bad
row is stopped even if it bypasses this module.
"""

import json
from pathlib import Path

import pandas as pd
import pandera.pandas as pa
import shapely
from shapely.geometry import MultiPolygon, shape
from shapely.geometry.base import BaseGeometry
from sqlalchemy import Engine, text

REPO_ROOT = Path(__file__).resolve().parents[2]
REFERENCE_DIR = REPO_ROOT / "data" / "reference"
SCHEMA_SQL = REPO_ROOT / "sql" / "schema" / "reference.sql"

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


def _records(df: pd.DataFrame) -> list[dict]:
    """Rows as dicts for SQL parameters: dates as date, missing values as None."""
    out = df.copy()
    for column in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[column]):
            out[column] = out[column].dt.date
    out = out.astype(object).where(out.notna(), None)
    return out.to_dict("records")


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


def load_reference(engine: Engine, reference_dir: Path = REFERENCE_DIR) -> dict[str, int]:
    """Validate the reference files and (re)build the ref.* tables from them.

    Everything runs in one transaction: if any step fails, the database is left
    as it was. Returns the number of rows loaded per table.
    """
    zones = read_zones(reference_dir / "zones.csv")
    controls = read_control_cities(reference_dir / "control_cities.csv")
    boundaries = read_zone_boundaries(
        reference_dir / "zone_boundaries.geojson", zone_ids=zones["zone_id"]
    )
    # WKB (binary) keeps every coordinate exactly; hex so it travels as text
    wkb = shapely.to_wkb(boundaries.pop("geometry").to_numpy(), hex=True)
    boundaries = boundaries.assign(wkb=wkb)

    with engine.begin() as conn:
        conn.exec_driver_sql(SCHEMA_SQL.read_text())
        conn.execute(INSERT_ZONES, _records(zones))
        conn.execute(INSERT_CONTROL_CITIES, _records(controls))
        conn.execute(INSERT_BOUNDARIES, _records(boundaries))
    return {
        "zones": len(zones),
        "control_cities": len(controls),
        "zone_boundaries": len(boundaries),
    }
