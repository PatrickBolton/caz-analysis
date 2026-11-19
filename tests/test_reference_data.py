"""Sanity checks for the hand-made reference files in data/reference/.

These files define the treatment (which zone started when), so a typo here would
silently bias every later result. The checks run in CI on every push.
"""

import csv
import datetime as dt
from pathlib import Path

import pytest

from caz.boundaries import read_boundaries
from caz.geo import haversine_km

REFERENCE_DIR = Path(__file__).resolve().parents[1] / "data" / "reference"

# Rough bounding box for Great Britain.
LAT_RANGE = (49.8, 60.9)
LON_RANGE = (-8.7, 1.8)

# A control city closer than this to a zone centre risks spillover contamination.
MIN_CONTROL_DISTANCE_KM = 15.0
# Known, documented exceptions: (control city, zone). Leeds is ~13 km from Bradford;
# it is kept as a control for other zones but must not be a donor for Bradford.
ALLOWED_CLOSE_PAIRS = {("leeds", "bradford")}


def read_csv(name: str) -> list[dict[str, str]]:
    with open(REFERENCE_DIR / name, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


ZONES = read_csv("zones.csv")
CONTROLS = read_csv("control_cities.csv")


def parse_date(value: str) -> dt.date:
    return dt.date.fromisoformat(value)  # requires YYYY-MM-DD


def test_zone_columns() -> None:
    expected = [
        "zone_id", "city", "country", "scheme_type", "class", "charging_start",
        "decision_date", "start_date_announced", "includes_private_cars",
        "approx_centre_lat", "approx_centre_lon", "source_url", "decision_source_url",
        "include", "exclusion_reason", "notes",
    ]  # fmt: skip
    assert list(ZONES[0].keys()) == expected


@pytest.mark.parametrize("zone", ZONES, ids=lambda z: z["zone_id"])
def test_excluded_zones_give_a_reason(zone: dict[str, str]) -> None:
    assert zone["include"] in {"true", "false"}
    assert bool(zone["exclusion_reason"]) == (zone["include"] == "false")


def test_every_included_zone_has_a_boundary() -> None:
    # Phase 1 labels sites inside / edge / outside from these polygons.
    boundaries = read_boundaries(REFERENCE_DIR / "zone_boundaries.geojson")
    included = {z["zone_id"] for z in ZONES if z["include"] == "true"}
    assert included <= set(boundaries)


@pytest.mark.parametrize("rows,id_col", [(ZONES, "zone_id"), (CONTROLS, "city_id")])
def test_ids_unique(rows: list[dict[str, str]], id_col: str) -> None:
    ids = [r[id_col] for r in rows]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("zone", ZONES, ids=lambda z: z["zone_id"])
def test_zone_row_valid(zone: dict[str, str]) -> None:
    decided = parse_date(zone["decision_date"])
    announced = parse_date(zone["start_date_announced"])
    charging = parse_date(zone["charging_start"])
    # The council decides on a zone, then the start date is announced, then charging begins.
    assert decided <= announced < charging
    assert dt.date(2018, 1, 1) <= charging <= dt.date.today()
    assert zone["country"] in {"England", "Scotland"}
    assert zone["scheme_type"] in {"charging_caz", "lez"}
    assert zone["class"] in {"B", "C", "D", "LEZ"}
    assert zone["includes_private_cars"] in {"true", "false"}
    # Class D and Scottish LEZs cover private cars; Classes B and C do not.
    assert (zone["includes_private_cars"] == "true") == (zone["class"] in {"D", "LEZ"})
    assert zone["source_url"].startswith("https://")
    assert zone["decision_source_url"].startswith("https://")


@pytest.mark.parametrize("row", ZONES + CONTROLS, ids=lambda r: r.get("zone_id") or r["city_id"])
def test_coordinates_in_great_britain(row: dict[str, str]) -> None:
    lat, lon = float(row["approx_centre_lat"]), float(row["approx_centre_lon"])
    assert LAT_RANGE[0] <= lat <= LAT_RANGE[1]
    assert LON_RANGE[0] <= lon <= LON_RANGE[1]


@pytest.mark.parametrize("city", CONTROLS, ids=lambda c: c["city_id"])
def test_control_row_valid(city: dict[str, str]) -> None:
    assert city["include"] in {"true", "false"}
    assert city["role"] in {"control", "never_charged", "excluded"}
    if city["include"] == "false":
        assert city["role"] == "excluded" and city["exclusion_reason"]
    else:
        assert city["role"] != "excluded" and not city["exclusion_reason"]


@pytest.mark.parametrize(
    "city", [c for c in CONTROLS if c["include"] == "true"], ids=lambda c: c["city_id"]
)
def test_controls_not_too_close_to_a_zone(city: dict[str, str]) -> None:
    lat, lon = float(city["approx_centre_lat"]), float(city["approx_centre_lon"])
    for zone in ZONES:
        if (city["city_id"], zone["zone_id"]) in ALLOWED_CLOSE_PAIRS:
            continue
        d = haversine_km(
            lat, lon, float(zone["approx_centre_lat"]), float(zone["approx_centre_lon"])
        )
        assert d >= MIN_CONTROL_DISTANCE_KM, (
            f"{city['city_id']} is {d:.0f} km from {zone['zone_id']}"
        )
