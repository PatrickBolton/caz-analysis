"""Hourly measurements: read one openair site-year file, validate it, load it into obs.hourly.

An openair site-year file (e.g. NEWC_2023.RData) holds several tables; the hourly
one has the file's name. Its `date` column is seconds since 1970-01-01 UTC,
hour-beginning. Each site-year is loaded in its own transaction, replacing any
rows already loaded for it, and the outcome is recorded in obs.hourly_loads.
"""

import datetime as dt
import warnings
from pathlib import Path

import pandas as pd
import pandera.pandas as pa
import rdata
from sqlalchemy import Connection, Engine, text

# openair column -> our column
COLUMNS = {"NO2": "no2", "PM2.5": "pm25", "ws": "ws", "wd": "wd", "temp": "air_temp"}
VALUE_COLUMNS = list(COLUMNS.values())


def _bounded(low: float, high: float) -> pa.Column:
    """A measurement column: missing allowed, otherwise within physical bounds."""
    return pa.Column(float, pa.Check.in_range(low, high), nullable=True)


# Bounds catch broken files (wrong units, sentinel values like -9999), not
# doubtful readings: values are stored as published. Instruments read slightly
# negative near zero, and a drifting zero can go further (LMAM SH2, March 2010:
# down to -21). Cleaning rules for single hours belong in a later, documented step.
HOURLY_SCHEMA = pa.DataFrameSchema(
    {
        "measured_at": pa.Column(
            "datetime64[ns, UTC]",
            pa.Check(lambda t: (t.dt.minute == 0) & (t.dt.second == 0), error="not on the hour"),
            unique=True,
        ),
        "no2": _bounded(-50, 1500),
        "pm25": _bounded(-50, 1500),
        "ws": _bounded(0, 75),
        "wd": _bounded(0, 360),
        "air_temp": _bounded(-40, 45),
    },
    strict=True,
)


class FileContentError(ValueError):
    """A site-year file does not hold the expected hourly table."""


def read_site_year(path: Path, site_id: str, year: int) -> pd.DataFrame:
    """Hourly table from an openair site-year file, validated, with our column names.

    Hours with no value for any of our columns are dropped. Columns a site does
    not measure (e.g. weather at LMAM sites) are all missing.
    """
    name = f"{site_id}_{year}"
    with warnings.catch_warnings():
        # rdata warns that it has no converter for R's date-time class; we
        # convert the underlying seconds ourselves below
        warnings.simplefilter("ignore", UserWarning)
        objects = rdata.read_rda(path)
    if name not in objects:
        raise FileContentError(f"{path.name}: no table {name} (found {list(objects)})")
    raw = pd.DataFrame(objects[name])
    if "date" not in raw:
        raise FileContentError(f"{path.name}: no date column")
    df = pd.DataFrame(
        {"measured_at": pd.to_datetime(raw["date"].astype(float), unit="s", utc=True)}
    )
    for source, column in COLUMNS.items():
        df[column] = pd.to_numeric(raw[source], errors="raise") if source in raw else float("nan")
    df["measured_at"] = df["measured_at"].astype("datetime64[ns, UTC]")
    df = df.dropna(subset=VALUE_COLUMNS, how="all").reset_index(drop=True)

    start = pd.Timestamp(dt.datetime(year, 1, 1, tzinfo=dt.UTC))
    outside = ~df["measured_at"].between(start, start + pd.DateOffset(years=1), inclusive="left")
    if outside.any():
        raise FileContentError(f"{path.name}: {int(outside.sum())} hours outside {year}")
    return HOURLY_SCHEMA.validate(df)


DELETE_YEAR = text("""
    DELETE FROM obs.hourly
    WHERE network = :network AND site_id = :site_id
      AND measured_at >= make_timestamptz(:year, 1, 1, 0, 0, 0, 'UTC')
      AND measured_at <  make_timestamptz(:year + 1, 1, 1, 0, 0, 0, 'UTC')
""")

RECORD_LOAD = text("""
    INSERT INTO obs.hourly_loads (
        network, site_id, year, status, n_hours, n_no2, detail, source_url)
    VALUES (:network, :site_id, :year, :status, :n_hours, :n_no2, :detail, :source_url)
    ON CONFLICT (network, site_id, year) DO UPDATE SET
        status = EXCLUDED.status, n_hours = EXCLUDED.n_hours, n_no2 = EXCLUDED.n_no2,
        detail = EXCLUDED.detail, source_url = EXCLUDED.source_url, loaded_at = now()
""")


def _copy_rows(conn: Connection, network: str, site_id: str, hourly: pd.DataFrame) -> None:
    """Bulk insert with COPY: far faster than one INSERT per row for ~9,000 rows."""
    columns = ["network", "site_id", "measured_at", *VALUE_COLUMNS]
    cursor = conn.connection.driver_connection.cursor()
    with cursor.copy(f"COPY obs.hourly ({', '.join(columns)}) FROM STDIN") as copy:
        for row in hourly.itertuples(index=False):
            values = [None if pd.isna(v) else float(v) for v in row[1:]]
            copy.write_row([network, site_id, row.measured_at.to_pydatetime(), *values])


def record_outcome(
    engine: Engine, network: str, site_id: str, year: int, status: str, url: str, detail: str
) -> None:
    """Record a site-year that could not be loaded (not_found / rejected)."""
    with engine.begin() as conn:
        conn.execute(DELETE_YEAR, {"network": network, "site_id": site_id, "year": year})
        conn.execute(
            RECORD_LOAD,
            {"network": network, "site_id": site_id, "year": year, "status": status,
             "n_hours": 0, "n_no2": 0, "detail": detail, "source_url": url},
        )  # fmt: skip


def load_site_year(
    engine: Engine, network: str, site_id: str, year: int, hourly: pd.DataFrame, url: str
) -> None:
    """Replace one site-year in obs.hourly with `hourly` (from read_site_year), atomically."""
    if (hourly["measured_at"].dt.year != year).any():
        raise ValueError(f"{network} {site_id}: rows outside {year}")
    with engine.begin() as conn:
        conn.execute(DELETE_YEAR, {"network": network, "site_id": site_id, "year": year})
        _copy_rows(conn, network, site_id, hourly)
        conn.execute(
            RECORD_LOAD,
            {"network": network, "site_id": site_id, "year": year, "status": "loaded",
             "n_hours": len(hourly), "n_no2": int(hourly["no2"].notna().sum()),
             "detail": None, "source_url": url},
        )  # fmt: skip
