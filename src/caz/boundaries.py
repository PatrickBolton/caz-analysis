"""Zone boundary polygons: building them from line pieces, and locating monitors.

Coordinates are (longitude, latitude) in WGS84, as in GeoJSON. Distances are in
metres, computed in a local flat projection centred on the zone. Over a few km
this is accurate to well under 0.1%, so no map-projection library is needed.
"""

import json
import math
from pathlib import Path

import shapely
from shapely.geometry import LineString, Point, mapping, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import polygonize, unary_union

from caz.geo import EARTH_RADIUS_KM

EDGE_M = 1000.0  # monitors outside a zone but within this distance of its boundary
LOCATIONS = ("inside", "edge", "outside", "unknown")  # see locate()


MIN_PART_FRACTION = 0.01  # pieces smaller than this share of a zone are drawing artefacts


def polygon_from_lines(
    outer: list[list[tuple[float, float]]], inner: list[list[tuple[float, float]]] | None = None
) -> BaseGeometry:
    """Join boundary line pieces (e.g. the OpenStreetMap ways of a relation) into a polygon.

    `outer` pieces must together form one or more closed rings; `inner` pieces form
    holes. Raises ValueError if the outer pieces do not close, so a broken boundary
    cannot silently become an empty or partial zone. Pieces smaller than
    MIN_PART_FRACTION of the total area (slivers where mapped lines overlap) are dropped.
    """
    outer_polygon = _faces(outer)
    if outer_polygon is None:
        raise ValueError("boundary pieces do not form a closed ring")
    holes = _faces(inner or [])
    polygon = outer_polygon.difference(holes) if holes is not None else outer_polygon
    parts = list(getattr(polygon, "geoms", [polygon]))
    total = sum(p.area for p in parts)
    return unary_union([p for p in parts if p.area >= MIN_PART_FRACTION * total])


def _faces(lines: list[list[tuple[float, float]]]) -> BaseGeometry | None:
    """Union of all areas enclosed by the lines, or None if they enclose nothing."""
    if not lines:
        return None
    polygons = list(polygonize(unary_union([LineString(line) for line in lines])))
    return unary_union(polygons) if polygons else None


def clean_geometry(geojson_geometry: dict) -> BaseGeometry:
    """Shapely geometry from a GeoJSON geometry, dropping any 3rd/4th coordinate values."""

    def two_d(coords):  # nested lists down to [x, y, (z), (m)] points
        if coords and isinstance(coords[0], (int, float)):
            return list(coords[:2])
        return [two_d(c) for c in coords]

    geometry = shape({**geojson_geometry, "coordinates": two_d(geojson_geometry["coordinates"])})
    if not geometry.is_valid:
        geometry = shapely.make_valid(geometry)
    return geometry


def to_local_metres(geometry: BaseGeometry, lat0: float, lon0: float) -> BaseGeometry:
    """Project (lon, lat) degrees to flat (x, y) metres around (lat0, lon0)."""
    metres_per_degree = EARTH_RADIUS_KM * 1000 * math.pi / 180
    scale_x = metres_per_degree * math.cos(math.radians(lat0))

    def project(coords):  # (n, 2) array of lon, lat
        out = coords.copy()
        out[:, 0] = (coords[:, 0] - lon0) * scale_x
        out[:, 1] = (coords[:, 1] - lat0) * metres_per_degree
        return out

    return shapely.transform(geometry, project)


def area_km2(zone: BaseGeometry) -> float:
    """Area of a (lon, lat) polygon in square kilometres."""
    centre = zone.centroid
    return to_local_metres(zone, centre.y, centre.x).area / 1e6


def distance_to_zone_m(lat: float, lon: float, zone: BaseGeometry) -> float:
    """Distance in metres from a point to a zone: 0 inside, else to the nearest boundary point."""
    local = to_local_metres(zone, lat, lon)  # the point is the origin
    return float(local.distance(Point(0.0, 0.0)))


def locate(lat: float, lon: float, zone: BaseGeometry | None) -> tuple[str, float | None]:
    """("inside" | "edge" | "outside" | "unknown", distance to zone in metres or None).

    "edge" means outside but within EDGE_M of the boundary; "unknown" means no boundary.
    """
    if zone is None:
        return "unknown", None
    d = distance_to_zone_m(lat, lon, zone)
    if d == 0.0:
        return "inside", 0.0
    return ("edge" if d <= EDGE_M else "outside"), d


def write_boundaries(features: list[dict], path: Path) -> None:
    """Write [{"geometry": shapely geometry, **properties}] as a GeoJSON FeatureCollection.

    Coordinates are rounded to 6 decimal places (about 0.1 m), to keep the file small.
    """
    out = []
    for feature in features:
        props = {k: v for k, v in feature.items() if k != "geometry"}
        geometry = shapely.set_precision(feature["geometry"], 1e-6)
        out.append({"type": "Feature", "properties": props, "geometry": mapping(geometry)})
    # one feature per line: compact, but diffs still show which zone changed
    lines = [json.dumps(f, separators=(",", ":")) for f in out]
    path.write_text('{"type":"FeatureCollection","features":[\n' + ",\n".join(lines) + "\n]}\n")


def read_boundaries(path: Path) -> dict[str, BaseGeometry]:
    """zone_id -> polygon, from a file written by write_boundaries."""
    data = json.loads(path.read_text())
    return {f["properties"]["zone_id"]: shape(f["geometry"]) for f in data["features"]}
