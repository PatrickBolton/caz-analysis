-- Clean NO2 panels: hourly -> daily -> weekly, per instrument and per area.
--
-- Cleaning rules (docs/DECISIONS.md, 2026-11-19), named here so they are easy to find:
--   1. an hourly NO2 value below -5 ug/m3 is treated as missing
--      (instruments read slightly negative near zero; lower values mean a fault);
--   2. a daily mean needs at least 18 of 24 hours (75%, as for annual means);
--   3. a weekly mean (Monday-Sunday) needs at least 5 of 7 valid days.
-- Days are UTC calendar days. The daily, weekly and area panels are
-- materialized (stored), so queries are fast; they are refreshed after new hourly
-- data is loaded (caz.database.refresh_panels, called by scripts/load_hourly.py).

DROP SCHEMA IF EXISTS panel CASCADE;
CREATE SCHEMA panel;

-- One row per instrument and hour. An instrument listed in two networks is one
-- instrument: each value comes from the kept record if it has it, otherwise from
-- a copy. (Where both have a value they are identical: checked 2026-11-19.)
CREATE VIEW panel.hourly_clean AS
WITH records AS (
    SELECT s.canon_network AS network, s.canon_site_id AS site_id, h.measured_at,
           h.no2, h.pm25, h.ws, h.wd, h.air_temp,
           (s.network, s.site_id) <> (s.canon_network, s.canon_site_id) AS is_copy
    FROM obs.hourly h
    JOIN aq.sites s USING (network, site_id)
),
pooled AS (
    SELECT network, site_id, measured_at,
           -- first non-missing value, kept record before copies
           (array_agg(no2 ORDER BY is_copy) FILTER (WHERE no2 IS NOT NULL))[1] AS no2,
           (array_agg(pm25 ORDER BY is_copy) FILTER (WHERE pm25 IS NOT NULL))[1] AS pm25,
           (array_agg(ws ORDER BY is_copy) FILTER (WHERE ws IS NOT NULL))[1] AS ws,
           (array_agg(wd ORDER BY is_copy) FILTER (WHERE wd IS NOT NULL))[1] AS wd,
           (array_agg(air_temp ORDER BY is_copy) FILTER (WHERE air_temp IS NOT NULL))[1]
               AS air_temp,
           bool_and(is_copy) AS only_in_copy
    FROM records
    GROUP BY network, site_id, measured_at
)
SELECT network, site_id, measured_at,
       CASE WHEN no2 >= -5 THEN no2 END AS no2,  -- rule 1
       no2 < -5 AS no2_removed,
       pm25, ws, wd, air_temp, only_in_copy
FROM pooled;

-- One row per instrument and UTC day with at least one NO2 hour.
CREATE MATERIALIZED VIEW panel.site_daily AS
SELECT network, site_id,
       (measured_at AT TIME ZONE 'UTC')::date AS day,
       count(no2) AS no2_hours,
       CASE WHEN count(no2) >= 18 THEN avg(no2) END AS no2  -- rule 2
FROM panel.hourly_clean
GROUP BY 1, 2, 3
HAVING count(no2) > 0;
CREATE UNIQUE INDEX ON panel.site_daily (network, site_id, day);

-- One row per instrument and week (starting Monday) with at least one NO2 day.
CREATE MATERIALIZED VIEW panel.site_weekly AS
SELECT network, site_id,
       date_trunc('week', day)::date AS week_start,
       count(no2) AS valid_days,
       CASE WHEN count(no2) >= 5 THEN avg(no2) END AS no2  -- rule 3; mean of daily means
FROM panel.site_daily
GROUP BY 1, 2, 3;
CREATE UNIQUE INDEX ON panel.site_weekly (network, site_id, week_start);

-- The weekly panel for all areas: mean over the instruments with a valid week,
-- split by location (inside / edge / outside / control) and site class.
-- Materialized too: the area lookup measures distances to every zone boundary,
-- which takes seconds per query.
CREATE MATERIALIZED VIEW panel.area_weekly AS
SELECT a.area_id, a.area_type, a.location, a.site_class, w.week_start,
       count(w.no2) AS n_sites,
       avg(w.no2) AS no2
FROM panel.site_weekly w
JOIN aq.site_areas a USING (network, site_id)
GROUP BY 1, 2, 3, 4, 5;
CREATE UNIQUE INDEX ON panel.area_weekly (area_id, location, site_class, week_start);
