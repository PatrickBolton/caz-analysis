"""Tests for caz.geo."""

import pytest

from caz.geo import haversine_km

# Approximate city-centre coordinates (decimal degrees).
BIRMINGHAM = (52.4797, -1.9027)
BRISTOL = (51.4545, -2.5879)


def test_distance_to_self_is_zero() -> None:
    assert haversine_km(*BIRMINGHAM, *BIRMINGHAM) == 0.0


def test_distance_is_symmetric() -> None:
    assert haversine_km(*BIRMINGHAM, *BRISTOL) == pytest.approx(haversine_km(*BRISTOL, *BIRMINGHAM))


def test_one_degree_of_latitude_is_about_111_km() -> None:
    assert haversine_km(52.0, -1.0, 53.0, -1.0) == pytest.approx(111.2, abs=0.1)


def test_birmingham_to_bristol() -> None:
    # Straight-line distance is about 123 km (the road distance is longer, about 140 km).
    assert haversine_km(*BIRMINGHAM, *BRISTOL) == pytest.approx(123, abs=2)
