"""Building the project database from the repository's files and cached downloads.

The build is a full rebuild, in one transaction:
    1. sql/schema/*.sql  - drop and recreate the schemas and tables, in file order
    2. insert reference data (zones, control cities, boundaries)
    3. insert monitoring sites (from openair metadata)
    4. sql/views/*.sql   - recreate the views, in file order
All inputs are validated before the database is touched, and a failure at any
step leaves the database as it was.
"""

from pathlib import Path

import pandas as pd
import psycopg.errors
import sqlalchemy.exc
from sqlalchemy import Connection, Engine

from caz.reference import REFERENCE_DIR, REPO_ROOT, insert_reference, read_reference
from caz.sites import insert_sites, site_tables

SQL_DIR = REPO_ROOT / "sql"
# Dropping a schema waits until no other session is reading its tables (e.g. a
# table left open in the VS Code PostgreSQL panel). Give up after this long
# rather than wait silently.
LOCK_TIMEOUT = "30s"


class DatabaseBusyError(RuntimeError):
    """Another session holds the tables the build or refresh needs."""


def _limit_lock_waits(conn: Connection) -> None:
    conn.exec_driver_sql(f"SET LOCAL lock_timeout = '{LOCK_TIMEOUT}'")


def _is_lock_timeout(error: Exception) -> bool:
    # psycopg's own error (SQL files run on the driver's cursor), or wrapped by SQLAlchemy
    return isinstance(error, psycopg.errors.LockNotAvailable) or isinstance(
        getattr(error, "orig", None), psycopg.errors.LockNotAvailable
    )


def _busy_message(error: Exception) -> DatabaseBusyError:
    return DatabaseBusyError(
        f"waited {LOCK_TIMEOUT} for another session to release the tables. Close open "
        "tables or queries (e.g. in VS Code's PostgreSQL panel, or psql) and run again."
    )


def run_sql_files(conn: Connection, folder: Path) -> None:
    """Run every .sql file in `folder`, in name order (01_..., 02_...).

    Uses the driver's cursor without parameters, so a '%' in the SQL (e.g. in a
    comment) is just a character, not a parameter placeholder. Same connection,
    so it is part of the caller's transaction.
    """
    cursor = conn.connection.cursor()
    for path in sorted(folder.glob("*.sql")):
        cursor.execute(path.read_text())


def build_database(
    engine: Engine, metadata: pd.DataFrame, reference_dir: Path = REFERENCE_DIR
) -> dict[str, int]:
    """Rebuild all tables and views. `metadata`: openair site metadata (all networks).

    Returns the number of rows inserted per table.
    """
    reference = read_reference(reference_dir)  # validate everything first
    sites, parameters = site_tables(metadata)
    try:
        with engine.begin() as conn:
            _limit_lock_waits(conn)
            run_sql_files(conn, SQL_DIR / "schema")
            insert_reference(conn, reference)
            insert_sites(conn, sites, parameters)
            run_sql_files(conn, SQL_DIR / "views")
    except (sqlalchemy.exc.OperationalError, psycopg.errors.LockNotAvailable) as e:
        if _is_lock_timeout(e):
            raise _busy_message(e) from e
        raise
    counts = {name: len(table) for name, table in reference.items()}
    return counts | {"sites": len(sites), "site_parameters": len(parameters)}


# Materialized panels, in dependency order (sql/views/04_panels.sql)
PANELS = ["panel.site_daily", "panel.site_weekly", "panel.area_weekly"]

PANEL_SUMMARY = """
    SELECT
        (SELECT count(*) FROM obs.hourly WHERE no2 IS NOT NULL) AS no2_hours_loaded,
        (SELECT count(*) FILTER (WHERE no2 IS NOT NULL) FROM panel.hourly_clean)
            AS no2_hours_clean,
        (SELECT count(*) FILTER (WHERE no2_removed) FROM panel.hourly_clean)
            AS removed_below_minus_5,
        (SELECT count(*) FILTER (WHERE only_in_copy AND no2 IS NOT NULL)
         FROM panel.hourly_clean) AS no2_hours_from_copies,
        (SELECT count(*) FROM panel.site_daily) AS days,
        (SELECT count(*) FROM panel.site_daily WHERE no2 IS NULL) AS days_under_18_hours,
        (SELECT count(*) FROM panel.site_weekly) AS weeks,
        (SELECT count(*) FROM panel.site_weekly WHERE no2 IS NULL) AS weeks_under_5_days
"""


def refresh_panels(engine: Engine) -> None:
    """Recompute the stored panels from obs.hourly (after loading new hourly data)."""
    try:
        with engine.begin() as conn:
            _limit_lock_waits(conn)
            for name in PANELS:
                conn.exec_driver_sql(f"REFRESH MATERIALIZED VIEW {name}")
    except (sqlalchemy.exc.OperationalError, psycopg.errors.LockNotAvailable) as e:
        if _is_lock_timeout(e):
            raise _busy_message(e) from e
        raise


def panel_summary(engine: Engine) -> dict[str, int]:
    """How many values each cleaning rule removed, and what came from copies."""
    with engine.connect() as conn:
        return dict(conn.exec_driver_sql(PANEL_SUMMARY).mappings().one())
