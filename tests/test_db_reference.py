"""Loading the reference data into Postgres/PostGIS (needs the database container).

Run with:  uv run --env-file .env pytest -m db

Uses a separate, throwaway database (caz_test) on the same server, so the
development database is never touched.
"""

from collections.abc import Iterator

import numpy as np
import pytest
import shapely
import sqlalchemy.exc
from sqlalchemy import Engine, text

from caz import reference as r
from caz.boundaries import distance_to_zone_m, read_boundaries
from caz.db import get_engine

pytestmark = pytest.mark.db
TEST_DB = "caz_test"


def recreate_test_database(admin: Engine) -> None:
    # CREATE/DROP DATABASE cannot run inside a transaction, hence AUTOCOMMIT
    with admin.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.execute(text(f"DROP DATABASE IF EXISTS {TEST_DB} WITH (FORCE)"))
        conn.execute(text(f"CREATE DATABASE {TEST_DB}"))


@pytest.fixture(scope="module")
def engine() -> Iterator[Engine]:
    """A fresh test database with the extensions and the reference data loaded."""
    admin = get_engine()
    recreate_test_database(admin)
    test_engine = get_engine(TEST_DB)
    with test_engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION postgis"))
    r.load_reference(test_engine)
    yield test_engine
    test_engine.dispose()
    with admin.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.execute(text(f"DROP DATABASE IF EXISTS {TEST_DB} WITH (FORCE)"))
    admin.dispose()


def scalar(engine: Engine, sql: str, **params):
    with engine.connect() as conn:
        return conn.execute(text(sql), params).scalar_one()


def test_all_rows_are_loaded(engine: Engine) -> None:
    assert scalar(engine, "SELECT count(*) FROM ref.zones") == 11
    assert scalar(engine, "SELECT count(*) FROM ref.control_cities") == 18
    assert scalar(engine, "SELECT count(*) FROM ref.zone_boundaries") == 11
    assert scalar(engine, "SELECT count(*) FROM ref.zones WHERE include") == 9


def test_values_survive_the_round_trip(engine: Engine) -> None:
    sql = "SELECT charging_start::text || ' ' || class FROM ref.zones WHERE zone_id = 'tyneside'"
    assert scalar(engine, sql) == "2023-01-30 C"
    sql = "SELECT exclusion_reason FROM ref.zones WHERE zone_id = 'bristol'"
    assert scalar(engine, sql) is None
    sql = "SELECT ST_AsText(centre) FROM ref.control_cities WHERE city_id = 'leeds'"
    assert scalar(engine, sql) == "POINT(-1.5478 53.7965)"


def test_boundaries_are_stored_exactly(engine: Engine) -> None:
    # WKB transfer: PostGIS holds the same coordinates as the GeoJSON file
    files = read_boundaries(r.REFERENCE_DIR / "zone_boundaries.geojson")
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT zone_id, ST_AsBinary(geom) FROM ref.zone_boundaries"))
        stored = {zone_id: shapely.from_wkb(bytes(wkb)) for zone_id, wkb in rows}
    for zone_id, geometry in files.items():
        # same vertices in the same order, to the last bit (tolerance 0)
        assert stored[zone_id].equals_exact(r._as_multipolygon(geometry), tolerance=0)


def test_excluded_zone_without_reason_is_refused_by_the_database(engine: Engine) -> None:
    with pytest.raises(sqlalchemy.exc.IntegrityError), engine.begin() as conn:
        conn.execute(text("UPDATE ref.zones SET exclusion_reason = NULL WHERE zone_id = 'bath'"))


def test_failed_load_leaves_the_previous_tables(engine: Engine, monkeypatch) -> None:
    # Break the last step (boundaries insert): the schema was already dropped and
    # zones re-inserted, so only a rollback can leave the old tables in place.
    monkeypatch.setattr(r, "INSERT_BOUNDARIES", text("INSERT INTO ref.no_such_table VALUES (1)"))
    with pytest.raises(sqlalchemy.exc.ProgrammingError):
        r.load_reference(engine)
    assert scalar(engine, "SELECT count(*) FROM ref.zone_boundaries") == 11


def test_postgis_agrees_with_python_on_inside_and_distance(engine: Engine) -> None:
    """For a grid of points around every zone: same inside/outside answer, and
    distances within 0.5% (PostGIS uses the WGS84 ellipsoid, our Python a sphere)."""
    boundaries = read_boundaries(r.REFERENCE_DIR / "zone_boundaries.geojson")
    sql = text("""
        SELECT ST_Covers(geom, pt), ST_Distance(geom::geography, pt::geography)
        FROM ref.zone_boundaries,
             LATERAL (SELECT ST_SetSRID(ST_MakePoint(:lon, :lat), 4326) AS pt) p
        WHERE zone_id = :zone_id
    """)
    n_inside = 0
    with engine.connect() as conn:
        for zone_id, zone in boundaries.items():
            lon0, lat0 = zone.centroid.x, zone.centroid.y
            for dlat in np.linspace(-0.02, 0.02, 9):  # about +-2.2 km
                for dlon in np.linspace(-0.035, 0.035, 9):  # about +-2.2 km
                    lat, lon = lat0 + dlat, lon0 + dlon
                    covers, d_postgis = conn.execute(
                        sql, {"zone_id": zone_id, "lat": lat, "lon": lon}
                    ).one()
                    d_python = distance_to_zone_m(lat, lon, zone)
                    assert covers == (d_python == 0.0), (zone_id, lat, lon)
                    assert d_postgis == pytest.approx(d_python, rel=0.005, abs=1.0)
                    n_inside += covers
    assert n_inside > 50  # the grid really tests both sides of the boundaries
