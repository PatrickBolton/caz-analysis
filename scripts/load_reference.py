"""Load the reference files (zones, control cities, zone boundaries) into Postgres.

Run from the project root, with the database container running:
    docker compose up -d --wait
    uv run --env-file .env python scripts/load_reference.py

The ref.* tables are rebuilt from data/reference/ each time (one transaction).
"""

import pandas as pd

from caz.db import get_engine
from caz.reference import load_reference

# Area computed by PostGIS on the WGS84 ellipsoid, next to the area stored in the
# boundary file (computed in Python with a flat local projection): an independent
# check of both.
CHECK_AREAS = """
    SELECT z.zone_id, z.class, z.include, b.source, b.area_km2 AS area_file_km2,
           round((ST_Area(b.geom::geography) / 1e6)::numeric, 2) AS area_postgis_km2
    FROM ref.zones z
    LEFT JOIN ref.zone_boundaries b USING (zone_id)
    ORDER BY z.charging_start
"""


def main() -> None:
    engine = get_engine()
    counts = load_reference(engine)
    print("Loaded:", ", ".join(f"{table} {n}" for table, n in counts.items()))
    with engine.connect() as conn:
        print(pd.read_sql(CHECK_AREAS, conn).to_string(index=False))


if __name__ == "__main__":
    main()
