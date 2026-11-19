-- Runs once, when the database volume is first created (see docker/db/Dockerfile).
CREATE EXTENSION IF NOT EXISTS postgis;  -- geometry types, spatial joins (site inside zone?)
CREATE EXTENSION IF NOT EXISTS vector;   -- pgvector: embeddings for report retrieval (Phase 5)
