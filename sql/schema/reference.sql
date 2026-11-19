-- Reference tables: the zones (treatment), control cities, and zone boundaries.
--
-- These are small and fully defined by files in data/reference/, so they are
-- rebuilt from scratch on every load (scripts/load_reference.py), inside one
-- transaction: a failed load leaves the previous tables untouched.
-- Coordinates are WGS84 longitude/latitude (EPSG:4326), as in the source files.

DROP SCHEMA IF EXISTS ref CASCADE;
CREATE SCHEMA ref;

CREATE TABLE ref.zones (
    zone_id               text PRIMARY KEY,
    city                  text NOT NULL,
    country               text NOT NULL CHECK (country IN ('England', 'Scotland')),
    scheme_type           text NOT NULL CHECK (scheme_type IN ('charging_caz', 'lez')),
    class                 text NOT NULL CHECK (class IN ('B', 'C', 'D', 'LEZ')),
    charging_start        date NOT NULL,
    decision_date         date NOT NULL,
    start_date_announced  date NOT NULL,
    includes_private_cars boolean NOT NULL,
    include               boolean NOT NULL,   -- analysed from Phase 1 on
    exclusion_reason      text,
    centre                geometry(Point, 4326) NOT NULL,  -- approximate, for distances only
    source_url            text NOT NULL,
    decision_source_url   text NOT NULL,
    notes                 text,
    CHECK (decision_date <= start_date_announced AND start_date_announced < charging_start),
    CHECK (include = (exclusion_reason IS NULL))  -- excluded zones must say why
);

CREATE TABLE ref.zone_boundaries (
    zone_id    text PRIMARY KEY REFERENCES ref.zones (zone_id),
    geom       geometry(MultiPolygon, 4326) NOT NULL CHECK (ST_IsValid(geom)),
    source     text NOT NULL CHECK (source IN ('council', 'openstreetmap')),
    source_url text NOT NULL,
    licence    text NOT NULL,
    retrieved  date NOT NULL,
    area_km2   double precision NOT NULL CHECK (area_km2 > 0)
);
CREATE INDEX ON ref.zone_boundaries USING gist (geom);

CREATE TABLE ref.control_cities (
    city_id          text PRIMARY KEY,
    city             text NOT NULL,
    country          text NOT NULL,
    include          boolean NOT NULL,
    role             text NOT NULL,
    exclusion_reason text,
    centre           geometry(Point, 4326) NOT NULL,
    notes            text,
    source_url       text,
    CHECK (include = (exclusion_reason IS NULL)),
    CHECK (include = (role <> 'excluded'))
);
