"""Figures for the README, from the database.

Run from the project root (database built):
    uv run --env-file .env python scripts/make_figures.py

assets/zone_monitors.png: every zone at the same scale, with its boundary and the
NO2 monitors within 5 km of it, coloured by position (inside / edge / outside).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from caz.boundaries import read_boundaries, to_local_metres
from caz.db import get_engine

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets"
HALF_WIDTH_KM = 7.0  # every panel is 14 km wide, so zone sizes compare directly

# Colours: categorical slots 1-2 of the project palette plus a neutral grey;
# text stays in ink colours, never series colours
INSIDE, EDGE, OUTSIDE = "#2a78d6", "#eb6834", "#8a8984"
ZONE_FILL, ZONE_LINE, INK, MUTED = "#e9e7e1", "#52514e", "#2b2a28", "#6f6e69"
LOCATION_COLOUR = {"inside": INSIDE, "edge": EDGE, "outside": OUTSIDE}
SHAPE = {"roadside": "o", "background": "s"}  # anything else: triangle

ZONES = """
    SELECT zone_id, city, class AS zone_class, charging_start, include,
           ST_Y(ST_Centroid(b.geom)) AS lat0, ST_X(ST_Centroid(b.geom)) AS lon0
    FROM ref.zones JOIN ref.zone_boundaries b USING (zone_id)
    ORDER BY charging_start
"""
MONITORS = """
    SELECT l.zone_id, l.site_id, l.site_class, l.zone_location,
           ST_Y(s.geom) AS lat, ST_X(s.geom) AS lon
    FROM aq.site_zone_location l
    JOIN aq.sites s USING (network, site_id)
    WHERE EXISTS (SELECT 1 FROM aq.site_parameters p
                  WHERE (p.network, p.site_id) = (l.network, l.site_id)
                    AND p.parameter = 'NO2' AND coalesce(p.end_date, 'infinity') >= '2010-01-01')
"""


def to_km(lat: np.ndarray, lon: np.ndarray, lat0: float, lon0: float) -> tuple:
    """Flat km around (lat0, lon0), the same projection as caz.boundaries."""
    x = (lon - lon0) * 111.195 * np.cos(np.radians(lat0))
    return x, (lat - lat0) * 111.195


def draw(ax, zone, boundary, monitors: pd.DataFrame) -> None:
    local = to_local_metres(boundary, zone.lat0, zone.lon0)
    for poly in getattr(local, "geoms", [local]):
        xs, ys = np.array(poly.exterior.coords).T / 1000
        ax.fill(xs, ys, color=ZONE_FILL, lw=0, zorder=1)
        ax.plot(xs, ys, color=ZONE_LINE, lw=1.0, zorder=2)
    for _, m in monitors.iterrows():
        x, y = to_km(m.lat, m.lon, zone.lat0, zone.lon0)
        ax.scatter(x, y, s=46, marker=SHAPE.get(m.site_class, "^"),
                   color=LOCATION_COLOUR[m.zone_location], edgecolor="white",
                   linewidth=1.2, zorder=3)  # fmt: skip
    counts = monitors["zone_location"].value_counts()
    status = "" if zone.include else " · excluded"
    started = f"{zone.charging_start:%b %Y}"
    ax.set_title(
        f"{zone.zone_id.capitalize()}  ·  Class {zone.zone_class}  ·  {started}{status}",
        fontsize=9, color=INK if zone.include else MUTED, loc="left",
    )  # fmt: skip
    ax.text(0.02, 0.03, f"{counts.get('inside', 0)} inside · {counts.get('edge', 0)} edge · "
            f"{counts.get('outside', 0)} outside", transform=ax.transAxes, fontsize=8,
            color=MUTED)  # fmt: skip
    ax.set_xlim(-HALF_WIDTH_KM, HALF_WIDTH_KM)
    ax.set_ylim(-HALF_WIDTH_KM, HALF_WIDTH_KM)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#d8d6cf")


def legend(ax) -> None:
    ax.axis("off")
    handles = [
        Patch(facecolor=ZONE_FILL, edgecolor=ZONE_LINE, label="charging zone"),
        *[Line2D([], [], ls="", marker="o", ms=7, color=c, label=f"monitor {k}")
          for k, c in [("inside", INSIDE), ("at edge (≤ 1 km out)", EDGE),
                       ("outside (1–5 km)", OUTSIDE)]],
        Line2D([], [], ls="", marker="o", ms=7, color=MUTED, label="roadside"),
        Line2D([], [], ls="", marker="s", ms=7, color=MUTED, label="urban background"),
        Line2D([], [], ls="", marker="^", ms=7, color=MUTED, label="other"),
    ]  # fmt: skip
    ax.legend(handles=handles, loc="center left", frameon=False, fontsize=9, labelcolor=INK)
    note = "Excluded: no monitor inside with 3+ years\nof data before charging started"
    ax.text(0.02, 0.10, note,
            transform=ax.transAxes, fontsize=8, color=MUTED)  # fmt: skip
    ax.text(0.02, 0.0, f"Each panel {2 * HALF_WIDTH_KM:.0f} × {2 * HALF_WIDTH_KM:.0f} km",
            transform=ax.transAxes, fontsize=8, color=MUTED)  # fmt: skip


def main() -> None:
    engine = get_engine()
    with engine.connect() as conn:
        zones = pd.read_sql(ZONES, conn, parse_dates=["charging_start"])
        monitors = pd.read_sql(MONITORS, conn)
    boundaries = read_boundaries(ROOT / "data" / "reference" / "zone_boundaries.geojson")

    fig, axes = plt.subplots(3, 4, figsize=(13, 10), layout="constrained")
    for ax, (_, zone) in zip(axes.flat, zones.iterrows(), strict=False):
        draw(ax, zone, boundaries[zone.zone_id], monitors[monitors["zone_id"] == zone.zone_id])
    legend(axes.flat[-1])
    fig.suptitle("UK Clean Air Zones and Low Emission Zones, with nearby NO2 monitors "
                 "(active since 2010)", x=0.01, ha="left", fontsize=11, color=INK)  # fmt: skip
    OUT.mkdir(exist_ok=True)
    path = OUT / "zone_monitors.png"
    fig.savefig(path, dpi=110, facecolor="white")
    print(f"wrote {path.relative_to(ROOT)} ({path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
