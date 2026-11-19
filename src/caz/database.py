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
from sqlalchemy import Connection, Engine

from caz.reference import REFERENCE_DIR, REPO_ROOT, insert_reference, read_reference
from caz.sites import insert_sites, site_tables

SQL_DIR = REPO_ROOT / "sql"


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
    with engine.begin() as conn:
        run_sql_files(conn, SQL_DIR / "schema")
        insert_reference(conn, reference)
        insert_sites(conn, sites, parameters)
        run_sql_files(conn, SQL_DIR / "views")
    counts = {name: len(table) for name, table in reference.items()}
    return counts | {"sites": len(sites), "site_parameters": len(parameters)}
