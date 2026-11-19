"""Building the database (needs the database container).

Run with:  uv run --env-file .env pytest -m db

Uses a separate, throwaway database (caz_test) on the same server, so the
development database is never touched. The monitoring sites are made up (a grid
of points around every zone), so the tests need no downloads.
"""

from collections.abc import Iterator

import numpy as np
import pandas as pd
import pytest
import shapely
import sqlalchemy.exc
from sqlalchemy import Engine, text

from caz import database, hourly
from caz import reference as r
from caz.boundaries import EDGE_M, locate, read_boundaries
from caz.db import get_engine

pytestmark = pytest.mark.db
TEST_DB = "caz_test"
BOUNDARIES = read_boundaries(r.REFERENCE_DIR / "zone_boundaries.geojson")
VIEW_LIMIT_M = 5000  # aq.site_zone_location keeps sites within this of a boundary


def grid_sites() -> pd.DataFrame:
    """Metadata for a 9 x 9 grid of NO2 sites (about +-2.2 km) around every zone."""
    rows = []
    for zone_id, zone in BOUNDARIES.items():
        for i, dlat in enumerate(np.linspace(-0.02, 0.02, 9)):
            for j, dlon in enumerate(np.linspace(-0.035, 0.035, 9)):
                rows.append(
                    {
                        "network": "LMAM",
                        "site_id": f"{zone_id.upper()}_{i}{j}",
                        "latitude": zone.centroid.y + dlat,
                        "longitude": zone.centroid.x + dlon,
                    }
                )
    # The same instrument in two networks, 5 m apart, in the middle of Bath:
    # only the AURN record (kept) may appear in the view.
    bath = BOUNDARIES["bath"].representative_point()
    rows += [
        {"network": "AURN", "site_id": "DUP", "latitude": bath.y, "longitude": bath.x},
        {"network": "SAQN", "site_id": "DUPCOPY", "latitude": bath.y + 0.000045,
         "longitude": bath.x},
    ]  # fmt: skip
    df = pd.DataFrame(rows).assign(
        site_name="test site",
        location_type="Urban Traffic",
        provider_code=lambda d: (d["network"] == "LMAM").map({True: "test", False: None}),
        parameter="NO2",
        start_date=pd.Timestamp("2015-01-01"),
        end_date=pd.NaT,
    )
    df["provider_code"] = df["provider_code"].astype("string")
    return df


def recreate_test_database(admin: Engine) -> None:
    # CREATE/DROP DATABASE cannot run inside a transaction, hence AUTOCOMMIT
    with admin.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.execute(text(f"DROP DATABASE IF EXISTS {TEST_DB} WITH (FORCE)"))
        conn.execute(text(f"CREATE DATABASE {TEST_DB}"))


@pytest.fixture(scope="module")
def engine() -> Iterator[Engine]:
    """A fresh test database, built with the real reference data and the grid sites."""
    admin = get_engine()
    recreate_test_database(admin)
    test_engine = get_engine(TEST_DB)
    with test_engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION postgis"))
    database.build_database(test_engine, grid_sites())
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
    assert scalar(engine, "SELECT count(*) FROM ref.replication_sites") == 32
    assert scalar(engine, "SELECT count(*) FROM aq.sites") == 11 * 81 + 2
    assert scalar(engine, "SELECT count(*) FROM aq.site_parameters") == 11 * 81 + 2


def test_values_survive_the_round_trip(engine: Engine) -> None:
    sql = "SELECT charging_start::text || ' ' || class FROM ref.zones WHERE zone_id = 'tyneside'"
    assert scalar(engine, sql) == "2023-01-30 C"
    sql = "SELECT exclusion_reason FROM ref.zones WHERE zone_id = 'bristol'"
    assert scalar(engine, sql) is None
    sql = "SELECT ST_AsText(centre) FROM ref.control_cities WHERE city_id = 'leeds'"
    assert scalar(engine, sql) == "POINT(-1.5478 53.7965)"
    sql = "SELECT canon_network || ' ' || canon_site_id FROM aq.sites WHERE site_id = 'DUPCOPY'"
    assert scalar(engine, sql) == "AURN DUP"
    sql = "SELECT end_date FROM aq.site_parameters WHERE site_id = 'DUP'"
    assert scalar(engine, sql) is None  # still running


def test_boundaries_are_stored_exactly(engine: Engine) -> None:
    # WKB transfer: PostGIS holds the same coordinates as the GeoJSON file
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT zone_id, ST_AsBinary(geom) FROM ref.zone_boundaries"))
        stored = {zone_id: shapely.from_wkb(bytes(wkb)) for zone_id, wkb in rows}
    for zone_id, geometry in BOUNDARIES.items():
        # same vertices in the same order, to the last bit (tolerance 0)
        assert stored[zone_id].equals_exact(r._as_multipolygon(geometry), tolerance=0)


def test_excluded_zone_without_reason_is_refused_by_the_database(engine: Engine) -> None:
    with pytest.raises(sqlalchemy.exc.IntegrityError), engine.begin() as conn:
        conn.execute(text("UPDATE ref.zones SET exclusion_reason = NULL WHERE zone_id = 'bath'"))


def test_site_pointing_to_a_missing_instrument_is_refused_at_commit(engine: Engine) -> None:
    with pytest.raises(sqlalchemy.exc.IntegrityError), engine.begin() as conn:
        conn.execute(text("UPDATE aq.sites SET canon_site_id = 'NOPE' WHERE site_id = 'DUPCOPY'"))


def test_failed_build_leaves_the_previous_database(engine: Engine, monkeypatch) -> None:
    # Break the last step (views): schemas were already dropped and every table
    # refilled, so only a rollback can leave the old tables and view in place.
    def broken_views(conn, folder):
        if folder.name == "views":
            conn.exec_driver_sql("SELECT * FROM no_such_table")
        return run_sql_files(conn, folder)

    run_sql_files = database.run_sql_files
    monkeypatch.setattr(database, "run_sql_files", broken_views)
    with pytest.raises(sqlalchemy.exc.ProgrammingError):
        database.build_database(engine, grid_sites().head(3))
    assert scalar(engine, "SELECT count(*) FROM aq.sites") == 11 * 81 + 2
    assert scalar(engine, "SELECT count(*) FROM aq.site_zone_location") > 0


def test_view_labels_agree_with_python(engine: Engine) -> None:
    """Every grid site: same inside / edge / outside label and distance as caz.boundaries.

    PostGIS measures on the ellipsoid, Python on a sphere (about 0.2% apart), so
    sites within 0.5% of a threshold (1000 m, 5000 m) are not compared.
    """
    with engine.connect() as conn:
        view = pd.read_sql("SELECT * FROM aq.site_zone_location", conn)
        sites = pd.read_sql(
            "SELECT network, site_id, ST_Y(geom) AS lat, ST_X(geom) AS lon FROM aq.sites", conn
        )
    view = view.set_index(["network", "site_id", "zone_id"])
    compared = 0
    for site in sites.itertuples():
        for zone_id, zone in BOUNDARIES.items():
            label, d = locate(site.lat, site.lon, zone)
            near_threshold = any(abs(d - t) < 0.005 * t for t in (EDGE_M, VIEW_LIMIT_M))
            key = (site.network, site.site_id, zone_id)
            if near_threshold or site.site_id == "DUPCOPY":
                continue
            if d > VIEW_LIMIT_M:
                assert key not in view.index, key
                continue
            row = view.loc[key]
            assert row["zone_location"] == label, key
            # 1 m for ellipsoid vs sphere, + 0.5 m because the view rounds to whole metres
            assert row["distance_to_zone_m"] == pytest.approx(d, rel=0.005, abs=1.5), key
            compared += 1
    assert compared > 800
    labels = view["zone_location"].value_counts()
    assert min(labels.get(k, 0) for k in ("inside", "edge", "outside")) > 50  # all exercised


def test_copies_of_an_instrument_are_left_out_of_the_view(engine: Engine) -> None:
    sql = "SELECT count(*) FROM aq.site_zone_location WHERE site_id = :s"
    assert scalar(engine, sql, s="DUP") == 1  # Bath only
    assert scalar(engine, sql, s="DUPCOPY") == 0


def hourly_frame(n: int, no2: float = 30.0, year: int = 2023) -> pd.DataFrame:
    """n hours of validated hourly data from 1 January of `year`, 00:00 UTC."""
    times = pd.date_range(f"{year}-01-01", periods=n, freq="h", tz="UTC")
    times = times.astype("datetime64[ns, UTC]")
    return pd.DataFrame(
        {"measured_at": times, "no2": no2, "pm25": float("nan"), "ws": 2.0, "wd": 180.0,
         "air_temp": 5.0}
    )  # fmt: skip


def test_loading_a_site_year_twice_replaces_it(engine: Engine) -> None:
    hourly.load_site_year(engine, "AURN", "DUP", 2023, hourly_frame(48), "https://x/DUP_2023")
    hourly.load_site_year(engine, "AURN", "DUP", 2023, hourly_frame(24, no2=50.0), "https://x")
    sql = "SELECT count(*) FROM obs.hourly WHERE site_id = 'DUP'"
    assert scalar(engine, sql) == 24
    assert scalar(engine, "SELECT avg(no2) FROM obs.hourly WHERE site_id = 'DUP'") == 50.0
    sql = "SELECT concat_ws(' ', status, n_hours, n_no2) FROM obs.hourly_loads WHERE year = 2023"
    assert scalar(engine, sql) == "loaded 24 24"
    sql = "SELECT measured_at::text FROM obs.hourly WHERE site_id = 'DUP' ORDER BY 1 LIMIT 1"
    with engine.connect() as conn:
        conn.execute(text("SET TIME ZONE 'UTC'"))
        assert conn.execute(text(sql)).scalar_one() == "2023-01-01 00:00:00+00"


def test_not_found_is_recorded_and_clears_the_year(engine: Engine) -> None:
    hourly.load_site_year(engine, "AURN", "DUP", 2023, hourly_frame(5), "https://x")
    hourly.record_outcome(engine, "AURN", "DUP", 2023, "not_found", "https://x", "HTTP 404")
    assert scalar(engine, "SELECT status FROM obs.hourly_loads WHERE year = 2023") == "not_found"
    sql = "SELECT count(*) FROM obs.hourly WHERE site_id = 'DUP' AND measured_at >= '2023-01-01'"
    assert scalar(engine, sql) == 0


def test_rebuild_keeps_the_hourly_data(engine: Engine) -> None:
    hourly.load_site_year(engine, "AURN", "DUP", 2024, hourly_frame(10, year=2024), "https://x")
    before = scalar(engine, "SELECT count(*) FROM obs.hourly")
    database.build_database(engine, grid_sites())  # drops and recreates ref and aq
    assert scalar(engine, "SELECT count(*) FROM obs.hourly") == before > 0
    assert scalar(engine, "SELECT count(*) FROM obs.hourly_loads") > 0


def test_study_site_years_lists_included_zones_and_copies(engine: Engine) -> None:
    with engine.connect() as conn:
        years = pd.read_sql("SELECT * FROM aq.study_site_years", conn)
    # Bath is excluded, so its grid sites are not needed; Bristol's are
    assert not years["site_id"].str.startswith("BATH_").any()
    assert years["site_id"].str.startswith("BRISTOL_").any()
    assert years["year"].min() == 2015  # grid sites start in 2015 (data from 2010 on)
    assert years.query("site_id.str.startswith('BRISTOL_')")["provider_code"].eq("test").all()


def test_rows_outside_the_year_are_refused(engine: Engine) -> None:
    with pytest.raises(ValueError, match="outside 2022"):
        hourly.load_site_year(engine, "AURN", "DUP", 2022, hourly_frame(5), "https://x")


def day_hours(day: str, values: dict[int, float]) -> pd.DataFrame:
    """Hourly rows for one UTC day: {hour: no2}."""
    start = pd.Timestamp(day, tz="UTC")
    times = [start + pd.Timedelta(hours=h) for h in values]
    frame = pd.DataFrame({"measured_at": pd.DatetimeIndex(times).astype("datetime64[ns, UTC]")})
    return frame.assign(
        no2=list(values.values()), pm25=float("nan"), ws=float("nan"), wd=float("nan"),
        air_temp=float("nan"),
    )  # fmt: skip


def test_cleaning_rules_and_pooling(engine: Engine) -> None:
    """Week of Monday 1 March 2021 at DUP (AURN), with its copy DUPCOPY (SAQN).

    Mon: DUP has hours 0-17 (10); the copy also has 18-23 (40) -> pooled 24 hours,
         mean (18*10 + 6*40) / 24 = 17.5
    Tue: 18 hours of 20, but one is -10 -> removed (rule 1) -> 17 hours -> no daily
         value (rule 2)
    Wed-Sat: 24 hours of 30 -> 4 valid days
    Week: 5 valid days -> mean (17.5 + 4*30) / 5 = 27.5 (rule 3)
    Next week: 4 valid days only -> no weekly value
    """
    dup = pd.concat(
        [
            day_hours("2021-03-01", dict.fromkeys(range(18), 10.0)),
            day_hours("2021-03-02", {0: -10.0} | dict.fromkeys(range(1, 18), 20.0)),
            *[day_hours(f"2021-03-0{d}", dict.fromkeys(range(24), 30.0)) for d in range(3, 7)],
            *[day_hours(f"2021-03-{d:02d}", dict.fromkeys(range(24), 30.0)) for d in range(8, 12)],
        ],
        ignore_index=True,
    )
    copy = day_hours("2021-03-01", {h: 10.0 if h < 18 else 40.0 for h in range(24)})
    hourly.load_site_year(engine, "AURN", "DUP", 2021, dup, "https://x")
    hourly.load_site_year(engine, "SAQN", "DUPCOPY", 2021, copy, "https://x")
    database.refresh_panels(engine)

    with engine.connect() as conn:
        daily = pd.read_sql(
            "SELECT day::text, no2_hours, no2 FROM panel.site_daily "
            "WHERE site_id = 'DUP' AND day < '2021-03-08' ORDER BY day",
            conn,
        ).set_index("day")
        weekly = pd.read_sql(
            "SELECT week_start::text, valid_days, no2 FROM panel.site_weekly "
            "WHERE site_id = 'DUP' AND week_start >= '2021-03-01' ORDER BY week_start",
            conn,
        ).set_index("week_start")
    assert "DUPCOPY" not in daily.index  # copies are pooled into the kept record
    assert daily.loc["2021-03-01", "no2_hours"] == 24
    assert daily.loc["2021-03-01", "no2"] == pytest.approx(17.5)
    assert daily.loc["2021-03-02", "no2_hours"] == 17
    assert pd.isna(daily.loc["2021-03-02", "no2"])
    assert weekly.loc["2021-03-01", "valid_days"] == 5
    assert weekly.loc["2021-03-01", "no2"] == pytest.approx(27.5)
    assert weekly.loc["2021-03-08", "valid_days"] == 4
    assert pd.isna(weekly.loc["2021-03-08", "no2"])
    summary = database.panel_summary(engine)
    assert summary["removed_below_minus_5"] == 1
    assert summary["no2_hours_from_copies"] == 6


def test_area_panel_averages_instruments_by_location(engine: Engine) -> None:
    # The grid site at Bristol's centroid is inside the zone (Bristol is included)
    week = pd.concat(
        [day_hours(f"2021-03-0{d}", dict.fromkeys(range(24), 25.0)) for d in range(1, 6)],
        ignore_index=True,
    )
    hourly.load_site_year(engine, "LMAM", "BRISTOL_44", 2021, week, "https://x")
    database.refresh_panels(engine)
    sql = """SELECT location || ' ' || n_sites || ' ' || round(no2::numeric, 1)
             FROM panel.area_weekly WHERE area_id = 'bristol' AND week_start = '2021-03-01'"""
    assert scalar(engine, sql) == "inside 1 25.0"


def test_build_gives_up_when_another_session_holds_the_tables(engine, monkeypatch) -> None:
    monkeypatch.setattr(database, "LOCK_TIMEOUT", "1s")
    with engine.connect() as other:  # like a table left open in VS Code
        other.execute(text("SELECT 1 FROM ref.zones LIMIT 1"))  # transaction stays open
        with pytest.raises(database.DatabaseBusyError, match="another session"):
            database.build_database(engine, grid_sites())
        other.rollback()
