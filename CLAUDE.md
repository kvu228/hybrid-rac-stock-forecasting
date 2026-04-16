# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Hybrid stock pattern recognition & forecasting system built on PostgreSQL 16+ combining TimescaleDB (time-series) + pgvector (vector search) + RAC (Retrieval-Augmented Classification). This is a **database engineering** thesis project — the focus is on schema design, indexing strategies, and query planning, NOT AI/ML.

All components run on a single PostgreSQL instance: OHLCV time-series in a TimescaleDB hypertable, 128-d pattern embeddings with HNSW index via pgvector, and stored procedures that combine vector KNN search with B-Tree metadata lookups in one execution plan.

## Commands

```bash
# Setup
uv sync --dev                                    # Install dependencies (uses uv, not pip/poetry)
cp .env.example .env                             # Configure environment
docker compose --env-file .env up -d --build     # Start PostgreSQL + TimescaleDB + pgvector
alembic upgrade head                             # Apply all migrations

# ETL pipeline
python -m etl.pipeline seed                      # Load small fixture dataset
python -m etl.pipeline backfill --symbols VCB FPT --start 2024-01-01 --end 2024-03-31 --chunk-days 365 --concurrency 2
python -m etl.pipeline incremental --symbols-file etl/tickers_vn100.txt --end 2026-04-15 --chunk-days 365 --concurrency 4

# Quality checks
uv run ruff check .                              # Lint
uv run mypy .                                    # Type check

# Tests (require running DB + migrations applied)
uv run pytest -q                                 # All tests
uv run pytest tests/test_etl_smoke.py::test_ingestion_idempotent_with_unique_key -v  # Single test
```

## Architecture

```
Application Layer (Python)
  ├── etl/pipeline.py          CLI entry: seed | backfill | incremental
  ├── etl/vnstock_fetcher.py   Fetches OHLCV from Vnstock API (VCI provider)
  ├── etl/data_cleaner.py      Normalizes, deduplicates, validates
  └── etl/ingestion.py         Bulk COPY + upsert (idempotent on symbol+time)
         │
         ▼
PostgreSQL 16+ (single instance)
  ├── stock_ohlcv             TimescaleDB hypertable (auto-chunking, compression)
  ├── pattern_embeddings      pgvector vector(128) + HNSW index
  ├── rac_predictions         Forecast results with evidence
  ├── support_resistance_zones  S/R metadata for context enrichment
  └── Stored Procedures:
      ├── find_similar_patterns()       KNN via HNSW + threshold filter
      ├── compute_rac_context()         Aggregate K-neighbor stats
      ├── compute_full_rac_context()    Hybrid HNSW + B-Tree query
      └── get_distance_to_nearest_sr()  S/R level distance calc
```

**Data flow:** Vnstock API → fetcher → cleaner → bulk COPY + upsert → hypertable. Ingestion is idempotent via unique constraint on `(symbol, time)`.

**Database migrations** are in `alembic/versions/` (8 sequential migrations). Always use Alembic — never modify schema manually.

## Key Details

- **Python 3.13**, managed by **uv** (not pip/poetry)
- Tests are integration tests requiring a live database — start Docker and run migrations first
- The `.cursorrules` file contains extensive project context in Vietnamese including schema details, tuning parameters, and experimental scenarios
- `PROJECT_CONTEXT.md` has the full technical design document
- `IMPLEMENTATION_PLAN.md` tracks phased delivery (Phase 1-2 complete, 3-5 planned)
- ETL pipeline handles concurrency, date chunking, and Vnstock API error recovery
- CI runs: ruff → mypy → docker compose up → alembic upgrade → pytest