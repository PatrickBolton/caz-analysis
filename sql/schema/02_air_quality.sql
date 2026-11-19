-- Air-quality monitoring sites and what they measured, from the openair site
-- metadata of each network (AURN, SAQN, LMAM; AQE when it is back online).
-- Rebuilt on every build (scripts/build_database.py), inside one transaction.

DROP SCHEMA IF EXISTS aq CASCADE;
CREATE SCHEMA aq;

CREATE TABLE aq.sites (
    network       text NOT NULL CHECK (network IN ('AURN', 'AQE', 'SAQN', 'LMAM')),
    site_id       text NOT NULL,
    site_name     text NOT NULL,
    location_type text NOT NULL,  -- as published, e.g. 'Urban Traffic'
    site_class    text NOT NULL
        CHECK (site_class IN ('roadside', 'background', 'industrial', 'rural', 'other')),
    geom          geometry(Point, 4326) NOT NULL,
    -- The instrument this record belongs to. A site listed in two networks
    -- (within 10 m) is one instrument; the copy points to the kept record, and a
    -- kept record points to itself. See caz.feasibility.duplicate_mapping.
    canon_network text NOT NULL,
    canon_site_id text NOT NULL,
    PRIMARY KEY (network, site_id),
    -- checked at commit, so rows can be inserted in any order
    FOREIGN KEY (canon_network, canon_site_id) REFERENCES aq.sites (network, site_id)
        DEFERRABLE INITIALLY DEFERRED
);
CREATE INDEX ON aq.sites USING gist (geom);

CREATE TABLE aq.site_parameters (
    network    text NOT NULL,
    site_id    text NOT NULL,
    parameter  text NOT NULL,  -- openair code, e.g. 'NO2', 'PM2.5'
    start_date date NOT NULL,
    end_date   date,           -- NULL = still running
    PRIMARY KEY (network, site_id, parameter, start_date),
    FOREIGN KEY (network, site_id) REFERENCES aq.sites (network, site_id),
    CHECK (end_date IS NULL OR end_date >= start_date)
);
