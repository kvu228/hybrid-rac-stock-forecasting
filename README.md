# Hybrid-RAC-Stock-Forecasting

## Tiếng Việt

Hệ thống **nhận diện mẫu hình** và **dự báo chứng khoán** dựa trên kiến trúc **CSDL lai** chạy trên **một PostgreSQL 16+**:
- **TimescaleDB**: lưu trữ chuỗi thời gian OHLCV (hypertable, chunking, compression)
- **pgvector**: lưu trữ embedding `vector(128)` và tìm kiếm tương đồng bằng **HNSW**
- **RAC (Retrieval‑Augmented Classification)**: dự báo kèm **bằng chứng định lượng** từ Top‑K mẫu hình lịch sử tương đồng (giảm “black‑box”)

> Trọng tâm của repo là **Database Engineering** (schema, index, query plan, benchmark), **không** tập trung vào AI/Deep Learning.

## Mục tiêu

- Thiết kế schema và stored procedures để thực hiện **hybrid query**: vừa **vector search** (HNSW) vừa **lookup metadata** (B‑Tree) trong **một execution plan / một round‑trip**.
- Benchmark và tuning các tham số quan trọng:
  - TimescaleDB: `chunk_time_interval`, compression policy
  - pgvector HNSW: `m`, `ef_construction`, `ef_search`

## Dữ liệu

- **Nguồn**: Vnstock (VCI)
- **Phạm vi**: VN100 (~100 mã)
- **Tần suất**: Daily OHLCV

## Trạng thái hiện tại

Hiện repo chủ yếu chứa **tài liệu thiết kế** (xem `PROJECT_CONTEXT.md`). Phần code triển khai (ETL/API/benchmark) sẽ được bổ sung dần.

## Chạy Phase 1 (Infrastructure & DB Schema)

### Yêu cầu

- Docker Desktop
- Python project venv: `.venv`
- `uv`

### 1) Cấu hình biến môi trường

- Copy `.env.example` → `.env` và chỉnh nếu cần.
- ETL sẽ **tự load** `.env` (nếu có) để lấy `DATABASE_URL` và `VNSTOCK_API_KEY`.

### 2) Khởi động PostgreSQL (TimescaleDB + pgvector)

```powershell
docker compose --env-file .env up -d --build
```

### 3) Chạy migrations (Alembic)

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

# dùng DATABASE_URL trong .env (khuyến nghị), hoặc set tạm trong shell:
$env:DATABASE_URL="postgresql+psycopg://postgres:postgres@localhost:5432/hybrid_rac"
alembic upgrade head
```

### 4) Verify nhanh

```powershell
docker exec -it hybrid_rac_db psql -U postgres -d hybrid_rac
```

Trong `psql`:

- `\dx` (có `timescaledb`, `vector`)
- `SELECT hypertable_name FROM timescaledb_information.hypertables;` (có `stock_ohlcv`)
- `\dt` (có các bảng schema Phase 1)
- `SELECT indexname FROM pg_indexes WHERE tablename='pattern_embeddings' ORDER BY indexname;` (có `idx_embedding_hnsw`)

## Chạy Phase 2 (ETL: fetch → clean → ingest)

### Seed dataset nhỏ (khuyến nghị cho CI/dev)

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
$env:DATABASE_URL="postgresql+psycopg://postgres:postgres@localhost:5432/hybrid_rac"

# đảm bảo schema mới nhất (bao gồm unique key cho ingest idempotent)
alembic upgrade head

python -m db.seed_small_dataset
```

### Chạy ETL pipeline (gọi Vnstock)

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
$env:DATABASE_URL="postgresql+psycopg://postgres:postgres@localhost:5432/hybrid_rac"

# Backfill theo khoảng thời gian
python -m etl.pipeline backfill --symbols VCB FPT --start 2024-01-01 --end 2024-03-31 --chunk-days 365 --concurrency 2

# Incremental (tự lấy từ MAX(time) trong DB theo từng mã → end)
python -m etl.pipeline incremental --symbols VCB FPT --end 2026-04-15 --chunk-days 365 --concurrency 2
```

### Fetch full VN100 (lần đầu: 2010 → nay)

Repo có sẵn danh sách VN100 ở `etl/tickers_vn100.txt`.

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
$env:DATABASE_URL="postgresql+psycopg://postgres:postgres@localhost:5432/hybrid_rac"

alembic upgrade head

# Backfill toàn bộ VN100 (chia theo năm để tránh timeout)
python -m etl.pipeline backfill --symbols-file etl/tickers_vn100.txt --start 2010-01-01 --end 2026-04-15 --chunk-days 365 --concurrency 4
```

Các lần sau chỉ cần incremental:

```powershell
python -m etl.pipeline incremental --symbols-file etl/tickers_vn100.txt --end 2026-04-15 --chunk-days 365 --concurrency 4
```

### Tests / Lint / Type-check

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
$env:DATABASE_URL="postgresql+psycopg://postgres:postgres@localhost:5432/hybrid_rac"

ruff check .
mypy .
pytest -q
```

---

## English

This repo implements an **equity pattern recognition** and **stock forecasting** system built on a **hybrid database architecture** running on a **single PostgreSQL 16+** instance:
- **TimescaleDB**: OHLCV time-series storage (hypertables, chunking, compression)
- **pgvector**: `vector(128)` embeddings with **HNSW** similarity search
- **RAC (Retrieval‑Augmented Classification)**: predictions with **quantitative evidence** from Top‑K similar historical patterns (reducing “black-box” behavior)

> The main focus is **Database Engineering** (schema, indexes, query plans, benchmarks), not AI/Deep Learning.

## Goals

- Design schema + stored procedures to support **hybrid queries**: **vector search** (HNSW) + **structured metadata lookup** (B‑Tree) within **one execution plan / one round-trip**.
- Benchmark and tune key parameters:
  - TimescaleDB: `chunk_time_interval`, compression policies
  - pgvector HNSW: `m`, `ef_construction`, `ef_search`

## Data

- **Source**: Vnstock (VCI)
- **Universe**: VN100 (~100 tickers)
- **Frequency**: Daily OHLCV

## Current status

The repo currently contains mostly **design documentation** (see `PROJECT_CONTEXT.md`). Implementation code (ETL/API/benchmarks) will be added incrementally.

## Run Phase 1 (Infrastructure & DB Schema)

### Requirements

- Docker Desktop
- Project virtualenv: `.venv`
- `uv`

### 1) Configure env

- Copy `.env.example` → `.env`.

### 2) Start PostgreSQL (TimescaleDB + pgvector)

```powershell
docker compose --env-file .env up -d --build
```

### 3) Run migrations (Alembic)

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1

$env:DATABASE_URL="postgresql+psycopg://postgres:postgres@localhost:5432/hybrid_rac"
alembic upgrade head
```

### 4) Quick verify

```powershell
docker exec -it hybrid_rac_db psql -U postgres -d hybrid_rac
```

## Run Phase 2 (ETL: fetch → clean → ingest)

### Seed a small dataset (recommended for CI/dev)

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
$env:DATABASE_URL="postgresql+psycopg://postgres:postgres@localhost:5432/hybrid_rac"

alembic upgrade head
python -m db.seed_small_dataset
```

### Run ETL pipeline (calls Vnstock)

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
$env:DATABASE_URL="postgresql+psycopg://postgres:postgres@localhost:5432/hybrid_rac"

# Backfill a date range
python -m etl.pipeline backfill --symbols VCB FPT --start 2024-01-01 --end 2024-03-31 --chunk-days 365 --concurrency 2

# Incremental (from MAX(time) in DB per symbol → end)
python -m etl.pipeline incremental --symbols VCB FPT --end 2026-04-15 --chunk-days 365 --concurrency 2
```

### Fetch full VN100 (first run: 2010 → present)

The repo includes a VN100 ticker list at `etl/tickers_vn100.txt`.

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
$env:DATABASE_URL="postgresql+psycopg://postgres:postgres@localhost:5432/hybrid_rac"

alembic upgrade head

python -m etl.pipeline backfill --symbols-file etl/tickers_vn100.txt --start 2010-01-01 --end 2026-04-15 --chunk-days 365 --concurrency 4
```

After that, use incremental mode:

```powershell
python -m etl.pipeline incremental --symbols-file etl/tickers_vn100.txt --end 2026-04-15 --chunk-days 365 --concurrency 4
```

### Tests / Lint / Type-check

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
$env:DATABASE_URL="postgresql+psycopg://postgres:postgres@localhost:5432/hybrid_rac"

ruff check .
mypy .
pytest -q
```

