"""Build data/reference/zone_boundaries.geojson: one charging-zone polygon per zone.

Run from the project root:
    uv run python scripts/build_zone_boundaries.py            # uses cached downloads
    uv run python scripts/build_zone_boundaries.py --refresh  # re-download

Sources (see docs/DECISIONS.md, 2026-11-09 and 2026-11-19):
- the council's own published boundary where one could be found: GIS files for
  Birmingham, Bradford and Sheffield; for Portsmouth, the polygon behind the
  official map on the council's Cleaner Air Portsmouth website (served by the
  WP Google Maps plugin's public REST API);
- otherwise OpenStreetMap (ODbL), via the OSM editing API (an element and all
  its parts). OSM is cross-checked against the official file where both exist,
  and the result is printed.
"""

import argparse
import datetime as dt
import json
from pathlib import Path

import requests
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from caz.boundaries import area_km2, clean_geometry, polygon_from_lines, write_boundaries

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "boundaries"
OUT = ROOT / "data" / "reference" / "zone_boundaries.geojson"
OSM_API = "https://api.openstreetmap.org/api/0.6"
HEADERS = {"User-Agent": "caz-analysis/0.1 (+https://github.com/PatrickBolton)"}
OSM_LICENCE = "ODbL 1.0, (c) OpenStreetMap contributors"


def arcgis(url: str, licence: str, page: str) -> dict:
    return {"kind": "arcgis", "url": url, "licence": licence, "source_page": page}


def osm(element: str, osm_id: int) -> dict:
    return {
        "kind": "osm",
        "element": element,
        "osm_id": osm_id,
        "url": f"https://www.openstreetmap.org/{element}/{osm_id}",
        "licence": OSM_LICENCE,
        "source_page": f"https://www.openstreetmap.org/{element}/{osm_id}",
    }


def wp_map(url: str, page: str, polygon_name: str) -> dict:
    """A polygon drawn on a WordPress map (WP Google Maps plugin), found by its name."""
    return {
        "kind": "wp_map",
        "url": url,
        "polygon_name": polygon_name,
        "licence": "not stated",
        "source_page": page,
    }


ARCGIS_QUERY = "/query?where=1%3D1&outFields=*&outSR=4326&f=geojson"
SOURCES = {
    "bath": osm("way", 1146120795),
    "birmingham": arcgis(
        "https://maps.birmingham.gov.uk/server/rest/services/mybrummap/"
        "mybrummap_Transportation/MapServer/27" + ARCGIS_QUERY,
        "not stated",
        "https://maps.birmingham.gov.uk/server/rest/services/mybrummap/"
        "mybrummap_Transportation/MapServer/27",
    ),
    "bradford": arcgis(
        "https://gis.bradford.gov.uk/server/rest/services/Open_Data/"
        "Clean_Air_Zone_Boundary/MapServer/0" + ARCGIS_QUERY,
        "not stated",
        "https://www.data.gov.uk/dataset/7235c856-557d-4def-9b7f-02cc86260149/clean-air-zone-boundary",
    ),
    "portsmouth": wp_map(
        "https://cleanerairportsmouth.co.uk/wp-json/wpgmza/v1/polygons",
        "https://cleanerairportsmouth.co.uk/clean-air-zone-map/",
        "Portsmouth Clean Air Zone",
    ),
    "bristol": osm("relation", 15479588),
    "tyneside": osm("way", 1338896773),
    "sheffield": arcgis(
        "https://sheffieldcitycouncil.cloud.esriuk.com/server/rest/services/AGOL/"
        "OpenData/MapServer/28" + ARCGIS_QUERY,
        "Open Government Licence v3.0",
        "https://www.data.gov.uk/dataset/sheffield-clean-air-zone",
    ),
    "glasgow": osm("relation", 17878889),
    "dundee": osm("way", 1432941996),
    "aberdeen": osm("relation", 17878642),
    "edinburgh": osm("relation", 17296491),
}
# OSM versions of zones that have an official source, to measure OSM's accuracy.
OSM_CROSS_CHECKS = {"birmingham": osm("way", 770591846), "sheffield": osm("relation", 16389126)}


def fetch_json(name: str, refresh: bool, **request) -> dict:
    """GET once and cache the JSON under data/raw/boundaries/."""
    path = RAW / f"{name}.json"
    if path.exists() and not refresh:
        return json.loads(path.read_text())
    response = requests.get(headers=HEADERS, timeout=180, **request)
    response.raise_for_status()
    RAW.mkdir(parents=True, exist_ok=True)
    path.write_text(response.text)
    return response.json()


def build(name: str, source: dict, refresh: bool) -> BaseGeometry:
    """Polygon for one source."""
    if source["kind"] == "arcgis":
        data = fetch_json(name, refresh, url=source["url"])
        parts = [clean_geometry(f["geometry"]) for f in data["features"]]
        if not parts:
            raise ValueError(f"{name}: no features returned")
        return unary_union(parts)
    if source["kind"] == "wp_map":
        polygons = fetch_json(name, refresh, url=source["url"])["polygons"]
        (polygon,) = [p for p in polygons if p["polyname"] == source["polygon_name"]]
        if polygon["innerpolydata"]:
            raise ValueError(f"{name}: polygon has holes, which this code does not read")
        ring = [(float(p["lng"]), float(p["lat"])) for p in polygon["polydata"]]
        return clean_geometry({"type": "Polygon", "coordinates": [ring + ring[:1]]})
    url = f"{OSM_API}/{source['element']}/{source['osm_id']}/full.json"
    elements = fetch_json(name, refresh, url=url)["elements"]
    nodes = {e["id"]: (e["lon"], e["lat"]) for e in elements if e["type"] == "node"}
    ways = {e["id"]: e["nodes"] for e in elements if e["type"] == "way"}
    if source["element"] == "way":
        boundary_ways = [source["osm_id"]]
    else:
        (relation,) = [e for e in elements if e["type"] == "relation"]
        members = [m for m in relation["members"] if m["type"] == "way"]  # skip label nodes
        unknown = {m["role"] for m in members} - {"outer", "inner", ""}
        if unknown:
            raise ValueError(f"{name}: unexpected member roles {unknown}")
        boundary_ways = [m["ref"] for m in members if m["role"] != "inner"]
        hole_ways = [m["ref"] for m in members if m["role"] == "inner"]
        return polygon_from_lines(
            [[nodes[n] for n in ways[w]] for w in boundary_ways],
            [[nodes[n] for n in ways[w]] for w in hole_ways],
        )
    return polygon_from_lines([[nodes[n] for n in ways[w]] for w in boundary_ways])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--refresh", action="store_true", help="re-download cached files")
    args = parser.parse_args()

    features = []
    for zone_id, source in SOURCES.items():
        name = f"{zone_id}_{source['kind']}"
        geometry = build(name, source, args.refresh)
        # date the cached download was made, not today's date
        retrieved = dt.date.fromtimestamp((RAW / f"{name}.json").stat().st_mtime).isoformat()
        features.append(
            {
                "geometry": geometry,
                "zone_id": zone_id,
                "source": "openstreetmap" if source["kind"] == "osm" else "council",
                "source_url": source["source_page"],
                "licence": source["licence"],
                "retrieved": retrieved,
                "area_km2": round(area_km2(geometry), 2),
            }
        )
        print(f"{zone_id:11s} {features[-1]['source']:13s} {features[-1]['area_km2']:6.2f} km2")
    write_boundaries(features, OUT)
    print(f"\nWrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1024:.0f} KB)")

    print("\nOpenStreetMap vs council boundary (IoU = shared area / combined area; 1 = identical):")
    official = {f["zone_id"]: f["geometry"] for f in features}
    for zone_id, source in OSM_CROSS_CHECKS.items():
        osm_geometry = build(f"{zone_id}_osm", source, args.refresh)
        a, b = official[zone_id], osm_geometry
        iou = area_km2(a.intersection(b)) / area_km2(a.union(b))
        print(f"  {zone_id:11s} council {area_km2(a):5.2f} km2, OSM {area_km2(b):5.2f} km2, "
              f"IoU {iou:.3f}")  # fmt: skip


if __name__ == "__main__":
    main()
