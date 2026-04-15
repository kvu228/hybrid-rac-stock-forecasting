FROM postgres:16-bookworm

# Install TimescaleDB + pgvector for PostgreSQL 16
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends ca-certificates curl gnupg lsb-release; \
    echo "deb https://packagecloud.io/timescale/timescaledb/debian/ $(lsb_release -cs) main" > /etc/apt/sources.list.d/timescaledb.list; \
    curl -fsSL https://packagecloud.io/timescale/timescaledb/gpgkey | gpg --dearmor -o /usr/share/keyrings/timescaledb-archive-keyring.gpg; \
    printf "deb [signed-by=/usr/share/keyrings/timescaledb-archive-keyring.gpg] https://packagecloud.io/timescale/timescaledb/debian/ %s main\n" "$(lsb_release -cs)" > /etc/apt/sources.list.d/timescaledb.list; \
    apt-get update; \
    apt-get install -y --no-install-recommends timescaledb-2-postgresql-16 postgresql-16-pgvector; \
    rm -rf /var/lib/apt/lists/*

# Make TimescaleDB available to PostgreSQL
RUN echo "shared_preload_libraries = 'timescaledb'" >> /usr/share/postgresql/postgresql.conf.sample

