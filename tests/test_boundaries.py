"""Tests for caz.boundaries: building polygons and locating monitors."""

import json

import pytest
from shapely.geometry import Polygon, box

from caz import boundaries as b

# A 0.02 x 0.02 degree square near Bristol: about 1.39 km (E-W) x 2.22 km (N-S).
LAT, LON = 51.45, -2.60
SQUARE = box(LON, LAT, LON + 0.02, LAT + 0.02)
M_PER_DEG_LAT = 6371008.8 * 3.141592653589793 / 180  # ~111.2 km


def test_polygon_from_lines_joins_pieces_in_any_order() -> None:
    (x0, y0), (x1, y1) = (0, 0), (1, 1)
    pieces = [
        [(x1, y0), (x1, y1)],
        [(x0, y0), (x1, y0)],
        [(x0, y1), (x0, y0)],
        [(x1, y1), (x0, y1)],
    ]
    assert b.polygon_from_lines(pieces).equals(box(0, 0, 1, 1))


def test_polygon_from_lines_cuts_holes_from_inner_rings() -> None:
    outer = [list(box(0, 0, 10, 10).exterior.coords)]
    inner = [list(box(4, 4, 6, 6).exterior.coords)]
    assert b.polygon_from_lines(outer, inner).area == pytest.approx(100 - 4)


def test_polygon_from_lines_rejects_an_open_boundary() -> None:
    with pytest.raises(ValueError, match="closed ring"):
        b.polygon_from_lines([[(0, 0), (1, 0), (1, 1)]])  # three sides only


def test_polygon_from_lines_drops_slivers() -> None:
    zone = list(box(0, 0, 10, 10).exterior.coords)
    sliver = list(box(20, 0, 20.1, 0.1).exterior.coords)  # 0.01% of the zone's area
    assert b.polygon_from_lines([zone, sliver]).equals(box(0, 0, 10, 10))


def test_clean_geometry_drops_extra_coordinate_values() -> None:
    # e.g. Birmingham's file has [lon, lat, 0, null] points
    ring = [[0, 0, 0, None], [1, 0, 0, None], [1, 1, 0, None], [0, 0, 0, None]]
    geometry = b.clean_geometry({"type": "Polygon", "coordinates": [ring]})
    assert geometry.equals(Polygon([(0, 0), (1, 0), (1, 1)]))
    assert not geometry.has_z


def test_area_km2_of_a_known_square() -> None:
    expected = (0.02 * M_PER_DEG_LAT) ** 2 * 0.6234 / 1e6  # cos(51.46 deg) ~ 0.6234
    assert b.area_km2(SQUARE) == pytest.approx(expected, rel=2e-3)


def test_distance_to_zone_is_zero_inside_and_metres_outside() -> None:
    assert b.distance_to_zone_m(LAT + 0.01, LON + 0.01, SQUARE) == 0.0
    # 0.005 deg of latitude south of the square's southern edge
    d = b.distance_to_zone_m(LAT - 0.005, LON + 0.01, SQUARE)
    assert d == pytest.approx(0.005 * M_PER_DEG_LAT, rel=1e-3)  # ~556 m


@pytest.mark.parametrize(
    "dlat,expected",
    # 1000 m is ~0.00899 deg of latitude: test either side of the edge limit
    [(0.01, "inside"), (-0.005, "edge"), (-0.0089, "edge"), (-0.0091, "outside")],
)
def test_locate_classifies_by_distance(dlat: float, expected: str) -> None:
    location, _ = b.locate(LAT + dlat, LON + 0.01, SQUARE)
    assert location == expected


def test_locate_without_a_boundary_is_unknown() -> None:
    assert b.locate(LAT, LON, None) == ("unknown", None)


def test_boundaries_survive_a_write_read_round_trip(tmp_path) -> None:
    path = tmp_path / "zones.geojson"
    b.write_boundaries([{"geometry": SQUARE, "zone_id": "bristol", "licence": "x"}], path)
    back = b.read_boundaries(path)
    assert set(back) == {"bristol"}
    # same shape (vertex order may change) to within the 1e-6 degree rounding
    assert back["bristol"].hausdorff_distance(SQUARE) < 1e-6
    assert json.loads(path.read_text())["features"][0]["properties"]["licence"] == "x"
