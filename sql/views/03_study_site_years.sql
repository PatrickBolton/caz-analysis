-- The site-years whose hourly data the study needs (scripts/load_hourly.py).
--
-- Instruments that measured NO2 and belong to an area (aq.site_areas: within
-- 5 km of an included zone's boundary or of an included control city's centre),
-- plus every copy of those instruments in other networks, so no year is lost if
-- one network's record is incomplete. Years: 2010 to the last complete year,
-- where the site measured NO2 or PM2.5 at some point in that year.

CREATE VIEW aq.study_site_years AS
WITH instruments AS (  -- see 02_site_areas.sql
    SELECT DISTINCT network, site_id FROM aq.site_areas
),
records AS (  -- the instruments' own records and their copies
    SELECT s.network, s.site_id, s.provider_code
    FROM aq.sites s
    JOIN instruments i ON (s.canon_network, s.canon_site_id) = (i.network, i.site_id)
)
SELECT DISTINCT r.network, r.site_id, r.provider_code, y.year
FROM records r
JOIN aq.site_parameters p USING (network, site_id)
CROSS JOIN generate_series(2010, extract(year FROM current_date)::integer - 1) AS y (year)
WHERE p.parameter IN ('NO2', 'PM2.5')
  AND p.start_date <= make_date(y.year, 12, 31)
  AND coalesce(p.end_date, 'infinity'::date) >= make_date(y.year, 1, 1)
  AND EXISTS (SELECT 1 FROM aq.site_parameters n
              WHERE (n.network, n.site_id) = (r.network, r.site_id) AND n.parameter = 'NO2');
