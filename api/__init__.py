"""FastAPI application package for Hybrid-RAC-Stock.

This package exposes a small HTTP layer for demo/visualization:
- OHLCV time-series reads from TimescaleDB hypertable `stock_ohlcv`
- Metadata reads from `support_resistance_zones`
- RAC endpoints that call stored procedures combining pgvector (HNSW) + B-Tree
  lookups in a single database round-trip.
"""