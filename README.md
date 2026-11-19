# CAZ-analysis

**Did England's Clean Air Zones clean the air, and which design works best?**

Since 2021, seven English cities have introduced charging Clean Air Zones (CAZs) at different dates and with different strictness (Class B, C, D), and four Scottish cities have introduced Low Emission Zones. Existing evaluations look at one city at a time, each with its own method and usually only one year of data after launch, so their results cannot be compared. This project estimates all eleven zones with one pipeline, and turns the results into a decision tool for cities still considering a zone.

> Status: Phase 0 (setup and feasibility check) complete; nine zones have a usable monitor inside the zone and go forward. Next: Phase 1 (data foundation).

## Questions
1. How much did each zone reduce NO2, after removing the effects of weather and national trends?
2. Does the effect last, or fade as vehicle fleets modernise everywhere? Did it start before charging did, once the council had decided?
3. Does charging private cars (Class D: Birmingham, Bristol) reduce NO2 more than charging only commercial vehicles (Classes B/C)?
4. For a city considering a zone, what NO2 reduction should it expect, with what uncertainty, and at what cost?

## Earlier studies
- **Birmingham:** Liu et al. (2023, *Environmental and Resource Economics*) found modest first-year NO2 reductions using synthetic control. This project first replicates their design as a check on its own pipeline, then applies the same approach to every zone.
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
uv run --env-file .env pytest -m db       # database tests (in a throwaway caz_test database)
docker compose down                       # stop (data is kept in a Docker volume; add -v to delete it)
```

## Data licences
Air quality data © Crown copyright (DEFRA), Open Government Licence v3.0. Road traffic data © DfT, OGL v3.0. Zone boundaries from council open-data portals (Sheffield: OGL) and from OpenStreetMap (© OpenStreetMap contributors, ODbL 1.0; `data/reference/zone_boundaries.geojson` is available under the ODbL). Full attribution in `docs/DATA_SOURCES.md`.
