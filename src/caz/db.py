"""Connecting to the project database (the Postgres container in compose.yaml).

Settings come from environment variables, normally loaded from .env by uv:
    uv run --env-file .env python scripts/build_database.py
"""

import os

import pandas as pd
from sqlalchemy import URL, Engine, create_engine

REQUIRED = ("POSTGRES_DB", "POSTGRES_USER", "POSTGRES_PASSWORD")


def database_url(dbname: str | None = None) -> URL:
    """SQLAlchemy URL for the database (or for `dbname` on the same server)."""
    missing = [name for name in REQUIRED if not os.environ.get(name)]
    if missing:
        raise RuntimeError(
            f"database settings missing: {', '.join(missing)}. "
            "Run with `uv run --env-file .env ...` (see .env.example)."
        )
    return URL.create(
        "postgresql+psycopg",  # SQLAlchemy dialect + the psycopg 3 driver
        username=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
        host=os.environ.get("POSTGRES_HOST", "localhost"),
        port=int(os.environ.get("POSTGRES_PORT", "5432")),
        database=dbname or os.environ["POSTGRES_DB"],
    )


def get_engine(dbname: str | None = None) -> Engine:
    """A connection pool to the database. Use `with engine.begin() as conn:` for a transaction."""
    return create_engine(database_url(dbname))


def records(df: pd.DataFrame) -> list[dict]:
    """Rows as dicts for SQL parameters: dates as date, missing values as None."""
    out = df.copy()
    for column in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[column]):
            out[column] = out[column].dt.date
    out = out.astype(object).where(out.notna(), None)
    return out.to_dict("records")
