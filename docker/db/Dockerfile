# Postgres with the PostGIS (spatial) and pgvector (vector search) extensions.
#
# Why not the ready-made postgis/postgis image: it is published for Intel (amd64)
# only, so on Apple silicon it runs slowly under emulation, and it lacks pgvector.
# The official postgres image is multi-architecture and already has the PostgreSQL
# project's package repository (apt.postgresql.org) configured, so both extensions
# install as ordinary packages built for the same Postgres version.
FROM postgres:18.7-trixie

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        "postgresql-${PG_MAJOR}-postgis-3" \
        "postgresql-${PG_MAJOR}-pgvector" \
    && rm -rf /var/lib/apt/lists/*

# Scripts here run once, when the data volume is first created.
COPY init/ /docker-entrypoint-initdb.d/
