-- Where each monitoring instrument is relative to each zone it is near.
--
-- One row per (instrument, zone) for instruments within 5 km of the zone
-- boundary. Copies of an instrument in a second network are left out (only
-- kept records, where the site is its own canon). Labels, as in
-- caz.boundaries.locate:
--   inside  - in the zone or on its boundary
--   edge    - outside, within 1000 m of the boundary (caz.boundaries.EDGE_M)
--   outside - 1000 m to 5 km from the boundary
-- Distances are on the WGS84 ellipsoid (geography); the Python code uses a
-- sphere, so they differ by about 0.2% (tests/test_db.py checks agreement).

CREATE VIEW aq.site_zone_location AS
WITH instruments AS (
    SELECT network, site_id, site_name, site_class, geom
    FROM aq.sites
    WHERE network = canon_network AND site_id = canon_site_id
),
pairs AS (
    SELECT i.network, i.site_id, i.site_name, i.site_class, b.zone_id,
           ST_Covers(b.geom, i.geom) AS covered,
           ST_Distance(b.geom::geography, i.geom::geography) AS distance_m
    FROM instruments i
    JOIN ref.zone_boundaries b
      ON ST_DWithin(b.geom::geography, i.geom::geography, 5000)
)
SELECT network, site_id, site_name, site_class, zone_id,
       CASE WHEN covered THEN 'inside'
            WHEN distance_m <= 1000 THEN 'edge'
            ELSE 'outside'
       END AS zone_location,
       CASE WHEN covered THEN 0 ELSE round(distance_m)::integer END AS distance_to_zone_m
FROM pairs;
