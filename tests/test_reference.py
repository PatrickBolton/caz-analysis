"""Validation of the reference files before they are loaded (no database needed)."""

import shutil
from pathlib import Path

import pandas as pd
import pandera.errors
import pytest

from caz import reference as r


def test_committed_reference_files_are_valid() -> None:
    zones = r.read_zones()
    assert len(zones) == 11
    assert len(r.read_control_cities()) == 18
    boundaries = r.read_zone_boundaries(zone_ids=zones["zone_id"])
    assert set(boundaries["zone_id"]) == set(zones["zone_id"])
    assert (boundaries["geometry"].map(lambda g: g.geom_type) == "MultiPolygon").all()


def edited_zones(tmp_path: Path, **changes) -> Path:
    """A copy of zones.csv with some values of the first row (Bath) changed."""
    zones = pd.read_csv(r.REFERENCE_DIR / "zones.csv", dtype=str, keep_default_na=False)
    for column, value in changes.items():
        zones.loc[0, column] = value
    path = tmp_path / "zones.csv"
    zones.to_csv(path, index=False)
    return path


@pytest.mark.parametrize(
    "changes",
    [
        {"include": "true"},  # included, but still has an exclusion reason
        {"exclusion_reason": ""},  # excluded without a reason
        {"class": "E"},  # no such class
        {"charging_start": "2019-01-01"},  # before the council decided
        {"charging_start": "15/03/2021"},  # wrong date format
        {"includes_private_cars": "true"},  # Bath is Class C
        {"approx_centre_lat": "40.0"},  # not in Great Britain
        {"source_url": "http://insecure.example"},
    ],
)
def test_bad_zone_rows_are_rejected(tmp_path: Path, changes: dict) -> None:
    with pytest.raises((pandera.errors.SchemaError, ValueError)):
        r.read_zones(edited_zones(tmp_path, **changes))


def test_duplicate_zone_ids_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(pandera.errors.SchemaError):
        r.read_zones(edited_zones(tmp_path, zone_id="birmingham"))


def test_boundary_for_an_unknown_zone_is_rejected() -> None:
    zones = r.read_zones()
    with pytest.raises(ValueError, match="unknown zones"):
        r.read_zone_boundaries(zone_ids=zones["zone_id"][zones["zone_id"] != "bath"])


def test_records_turn_missing_values_into_none_and_dates_into_dates() -> None:
    zones = r.read_zones()
    record = r._records(zones)[zones.index[zones["zone_id"] == "tyneside"][0]]
    assert record["exclusion_reason"] is None
    assert str(record["charging_start"]) == "2023-01-30"


def test_load_reference_validates_before_touching_the_database(tmp_path: Path) -> None:
    # A bad file must fail validation before any connection is made: the engine
    # here is a placeholder that would raise a different error if used.
    for name in ("zones.csv", "control_cities.csv", "zone_boundaries.geojson"):
        shutil.copy(r.REFERENCE_DIR / name, tmp_path / name)
    edited_zones(tmp_path, **{"class": "E"})
    with pytest.raises(pandera.errors.SchemaError):
        r.load_reference(engine=None, reference_dir=tmp_path)  # type: ignore[arg-type]
