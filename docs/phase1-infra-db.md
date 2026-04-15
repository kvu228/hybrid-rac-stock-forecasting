## Phase 1 — Infrastructure & DB Schema (Alembic)

### Prereqs

- Docker Desktop
- Python (project `.venv`)
- `uv`

### 1) Configure env

- Copy `.env.example` → `.env` and adjust values if needed.

### 2) Start database

```powershell
docker compose --env-file .env up -d --build
```

### 3) Run Alembic migrations

```powershell
.\.venv\Scripts\Activate.ps1
alembic upgrade head
```

### 4) Verify (quick checks)

Inside DB (psql):

- Extensions: `timescaledb`, `vector`
- Hypertable exists: `stock_ohlcv`
- HNSW index exists: `idx_embedding_hnsw`
- Functions exist: `find_similar_patterns`, `compute_rac_context`, `get_distance_to_nearest_sr`, `compute_full_rac_context`

