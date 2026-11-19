-- Smoke test for the database container: the extensions are installed and work.
-- Run:  docker compose exec -T db sh -c 'psql -v ON_ERROR_STOP=1 -U "$POSTGRES_USER" -d "$POSTGRES_DB"' < docker/db/check.sql
-- Any failed check raises an error, so psql exits non-zero (used in CI).
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'postgis') THEN
        RAISE EXCEPTION 'postgis extension missing';
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_extension WHERE extname = 'vector') THEN
        RAISE EXCEPTION 'vector extension missing';
    END IF;
    -- a point in a 1x1 box is inside it; one outside is not
    IF NOT ST_Contains(ST_MakeEnvelope(0, 0, 1, 1, 4326), ST_SetSRID(ST_Point(0.5, 0.5), 4326))
       OR ST_Contains(ST_MakeEnvelope(0, 0, 1, 1, 4326), ST_SetSRID(ST_Point(2, 2), 4326)) THEN
        RAISE EXCEPTION 'PostGIS point-in-polygon check failed';
    END IF;
    -- Euclidean distance between (1, 2) and (4, 6) is 5
    IF ('[1,2]'::vector <-> '[4,6]'::vector) <> 5 THEN
        RAISE EXCEPTION 'pgvector distance check failed';
    END IF;
END $$;

SELECT extname, extversion FROM pg_extension WHERE extname IN ('postgis', 'vector') ORDER BY 1;
