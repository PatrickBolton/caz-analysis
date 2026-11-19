"""Validation of the reference files before they are loaded (no database needed)."""

import shutil
from pathlib import Path

import pandas as pd
import pandera.errors
import pytest

from caz import reference as r
from caz.db import records


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
    record = records(zones)[zones.index[zones["zone_id"] == "tyneside"][0]]
    assert record["exclusion_reason"] is None
    assert str(record["charging_start"]) == "2023-01-30"


def copy_reference(tmp_path: Path) -> None:
    for name in (
        "zones.csv",
        "control_cities.csv",
        "zone_boundaries.geojson",
        "replication_liu2023.csv",
    ):
        shutil.copy(r.REFERENCE_DIR / name, tmp_path / name)  # fmt: skip


def test_read_reference_validates_all_files(tmp_path: Path) -> None:
    copy_reference(tmp_path)
    assert {k: len(v) for k, v in r.read_reference(tmp_path).items()} == {
        "replication_sites": 32,
        "zones": 11,
        "control_cities": 18,
        "zone_boundaries": 11,
    }
    edited_zones(tmp_path, **{"class": "E"})
    with pytest.raises(pandera.errors.SchemaError):
        r.read_reference(tmp_path)


def test_replication_sites_from_the_paper() -> None:
    sites = r.read_replication_sites()
    assert set(sites["study"]) == {"liu2023"}
    assert sites["role"].value_counts().to_dict() == {"donor": 30, "treated": 2}
    donors = sites[sites["role"] == "donor"]
    assert donors["site_class"].value_counts().to_dict() == {"roadside": 15, "background": 15}


def test_bad_replication_row_is_rejected(tmp_path: Path) -> None:
    copy_reference(tmp_path)
    path = tmp_path / "replication_liu2023.csv"
    sites = pd.read_csv(path, dtype=str)
    sites.loc[0, "role"] = "control"  # not 'treated' or 'donor'
    sites.to_csv(path, index=False)
    with pytest.raises(pandera.errors.SchemaError):
        r.read_replication_sites(tmp_path)
