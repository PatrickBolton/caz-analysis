# CAZ-analysis

**Did England's Clean Air Zones clean the air, and which design works best?**

Since 2021, seven English cities have introduced charging Clean Air Zones (CAZs) at different dates and with different strictness (Class B, C, D), and four Scottish cities have introduced Low Emission Zones. Existing evaluations look at one city at a time, each with its own method and usually only one year of data after launch, so their results cannot be compared. This project estimates all eleven zones with one pipeline, and turns the results into a decision tool for cities still considering a zone.

> Status: Phases 0-1 complete. Nine zones have a usable NO2 monitor inside the zone; 10 million hours of NO2, PM2.5 and weather are loaded, validated and cleaned into daily and weekly panels in Postgres/PostGIS. Phase 2 (causal analysis) in progress: the published Birmingham result has been checked with this pipeline (below).

## Questions
1. How much did each zone reduce NO2, after removing the effects of weather and national trends?
2. Does the effect last, or fade as vehicle fleets modernise everywhere? Did it start before charging did, once the council had decided?
3. Does charging private cars (Class D, and the Scottish LEZs) reduce NO2 more than charging only commercial vehicles (Classes B/C)? Of the two English Class D zones only Bristol can be analysed: Birmingham has no usable monitor inside its zone.
4. For a city considering a zone, what NO2 reduction should it expect, with what uncertainty, and at what cost?

## Earlier studies
- **Birmingham:** Liu et al. (2023, *Environmental and Resource Economics*) found modest first-year NO2 reductions using weather normalisation and synthetic control. As a check on this pipeline, their design was repeated at the two of their sites in the national network (A4540 ring road, Ladywood). The estimates have the same sign and lie within each other's 95% intervals, but are 0.4-0.8 µg/m³ larger reductions (e.g. A4540, first 7 months: -1.29 vs -0.84 µg/m³). One of their control sites (Portsmouth) started its own zone during their study period.
- **Sheffield** (Williams et al. 2025): no significant effect; **Glasgow** (Shin et al. 2024): reductions; both with one or two monitors.
- **No causal studies found** for Portsmouth, Tyneside, Edinburgh, Aberdeen or Dundee.

## Approach (summary)
Open data (DEFRA air quality, council zone boundaries, DfT traffic counts) → Postgres/PostGIS → weather normalisation (ML) → staggered difference-in-differences and synthetic control → hierarchical Bayesian pooling across zones → scenario tool → dashboard with an AI assistant that only reports numbers produced by the model.

Details: [`docs/METHODS.md`](docs/METHODS.md) · Data: [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md) · Decisions: [`docs/DECISIONS.md`](docs/DECISIONS.md)

## Running locally
Needs [uv](https://docs.astral.sh/uv/) and Docker.

```bash
uv sync                                   # Python 3.12 and dependencies, as pinned in uv.lock
uv run pytest                             # tests
cp .env.example .env                      # then set POSTGRES_PASSWORD in .env
docker compose up -d --wait               # Postgres 18 + PostGIS + pgvector on localhost:5432
docker compose exec -T db sh -c 'psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$POSTGRES_DB"' < docker/db/check.sql
uv run --env-file .env python scripts/build_database.py   # reference tables, monitoring sites, views
uv run --env-file .env python scripts/load_hourly.py      # hourly NO2, PM2.5, weather (~300 MB; resumable)
uv run --env-file .env python scripts/check_hourly.py     # loaded data vs openair's annual statistics
uv run --env-file .env python scripts/replicate_liu2023.py  # Birmingham check against Liu et al. (2023)
uv run --env-file .env pytest -m db       # database tests (in a throwaway caz_test database)
docker compose down                       # stop (data is kept in a Docker volume; add -v to delete it)
```

## Data licences
Air quality data © Crown copyright (DEFRA), Open Government Licence v3.0. Road traffic data © DfT, OGL v3.0. Weather: ERA5 (Copernicus Climate Change Service) via the Open-Meteo API (CC BY 4.0). Zone boundaries from council open-data portals and maps (Sheffield: OGL; Birmingham, Bradford, Portsmouth: no licence stated) and from OpenStreetMap (© OpenStreetMap contributors, ODbL 1.0; `data/reference/zone_boundaries.geojson` is available under the ODbL). Full attribution in `docs/DATA_SOURCES.md`.
