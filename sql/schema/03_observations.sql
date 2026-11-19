-- Hourly measurements. Unlike ref and aq, this schema is NOT dropped on rebuild:
-- the data come from ~1,000 downloads, loaded incrementally by
-- scripts/load_hourly.py. IF NOT EXISTS makes this file safe to run every build.
-- There are no foreign keys to aq.sites, because aq is dropped and recreated
-- on every build; the loader only loads sites that exist in aq.sites.
-- Changing these tables later needs a migration (ALTER TABLE), not an edit here.

CREATE SCHEMA IF NOT EXISTS obs;

-- One row per site and hour. Timestamps are UTC and hour-beginning (openair
-- convention): 00:00 is the mean over 00:00-01:00.
CREATE TABLE IF NOT EXISTS obs.hourly (
    network     text NOT NULL,
    site_id     text NOT NULL,
    measured_at timestamptz NOT NULL,
    no2         double precision,  -- ug/m3
    pm25        double precision,  -- ug/m3
    ws          double precision,  -- wind speed, m/s (modelled; AURN and SAQN only)
    wd          double precision,  -- wind direction, degrees (modelled)
    air_temp    double precision,  -- air temperature, degrees C (modelled)
    PRIMARY KEY (network, site_id, measured_at),
    CHECK (extract(minute FROM measured_at) = 0 AND extract(second FROM measured_at) = 0)
);

-- What happened to each site-year file, so reruns only fetch what is missing.
CREATE TABLE IF NOT EXISTS obs.hourly_loads (
    network    text NOT NULL,
    site_id    text NOT NULL,
    year       integer NOT NULL,
    status     text NOT NULL CHECK (status IN ('loaded', 'not_found', 'rejected')),
    n_hours    integer NOT NULL DEFAULT 0,  -- rows loaded
    n_no2      integer NOT NULL DEFAULT 0,  -- hours with an NO2 value
    detail     text,                        -- reason, for not_found / rejected
    source_url text NOT NULL,
    loaded_at  timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (network, site_id, year)
);
