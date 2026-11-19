-- Which area each instrument belongs to, for the panels and the hourly download.
--
-- Zones: instruments within 5 km of an included zone's boundary, labelled
-- inside / edge / outside (aq.site_zone_location).
-- Control cities: instruments within 5 km of an included control city's centre,
-- labelled 'control'.
-- Only kept records (an instrument's copies in other networks are pooled into
-- it in panel.hourly_clean).

CREATE VIEW aq.site_areas AS
SELECT l.network, l.site_id, l.site_class,
       l.zone_id AS area_id, 'zone' AS area_type,
       l.zone_location AS location, l.distance_to_zone_m AS distance_m
FROM aq.site_zone_location l
JOIN ref.zones z USING (zone_id)
WHERE z.include
UNION ALL
SELECT s.network, s.site_id, s.site_class,
       c.city_id, 'control',
       'control', round(ST_Distance(c.centre::geography, s.geom::geography))::integer
FROM aq.sites s
JOIN ref.control_cities c
  ON c.include AND ST_DWithin(c.centre::geography, s.geom::geography, 5000)
WHERE s.network = s.canon_network AND s.site_id = s.canon_site_id;
