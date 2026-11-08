# CAZ-analysis

**Did England's Clean Air Zones clean the air, and which design works best?**

Since 2021, seven English cities have introduced charging Clean Air Zones (CAZs) at different dates and with different strictness (Class B, C, D), and four Scottish cities have introduced Low Emission Zones. Existing evaluations look at one city at a time. This project compares them all with modern causal-inference methods and turns the results into a decision tool for cities still considering a zone.

> Status: planning. See [`docs/PLAN.md`](docs/PLAN.md).

## Questions
1. How much did each zone reduce NO2, after removing the effects of weather and national trends?
2. Does charging private cars (Class D: Birmingham, Bristol) reduce NO2 more than charging only commercial vehicles (Classes B/C)?
3. For a city considering a zone, what NO2 reduction should it expect, with what uncertainty, and at what cost?

## Approach (summary)
Open data (DEFRA air quality, council zone boundaries, DfT traffic counts) → Postgres/PostGIS → weather normalisation (ML) → staggered difference-in-differences and synthetic control → hierarchical Bayesian pooling across zones → scenario tool → dashboard with an AI assistant that only reports numbers produced by the model.

Details: [`docs/METHODS.md`](docs/METHODS.md) · Data: [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md) · Decisions: [`docs/DECISIONS.md`](docs/DECISIONS.md)

## Data licences
Air quality data © Crown copyright (DEFRA), Open Government Licence v3.0. Road traffic data © DfT, OGL v3.0. Zone boundaries from council open-data portals (OGL). Full attribution in `docs/DATA_SOURCES.md`.
