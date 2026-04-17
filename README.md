# Hybrid-RAC-Stock-Forecasting

## Tiếng Việt

Hệ thống **nhận diện mẫu hình** và **dự báo chứng khoán** dựa trên kiến trúc **CSDL lai** chạy trên **một PostgreSQL 16+**:
- **TimescaleDB**: lưu trữ chuỗi thời gian OHLCV (hypertable, chunking, compression)
- **pgvector**: lưu trữ embedding `vector(128)` và tìm kiếm tương đồng bằng **HNSW**
- **RAC (Retrieval‑Augmented Classification)**: dự báo kèm **bằng chứng định lượng** từ Top‑K mẫu hình lịch sử tương đồng (giảm "black‑box")

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

- **Phase 1** ✅ Infrastructure & DB Schema (TimescaleDB, pgvector, stored procedures, migrations)
- **Phase 2** ✅ ETL Pipeline (Vnstock fetch → clean → bulk ingest, idempotent upsert)
- **Phase 3** ✅ Feature Engineering (sliding windows, S/R detection, train/test split)
- **Phase 4** ✅ (core) ML & Embeddings (CNN encoder + embedding generator into `pattern_embeddings`)
- **Phase 5** ✅ RAC application layer (stored procedure wrappers + persist predictions)
- **Phase 7** ✅ (partial) FastAPI REST API (healthcheck, OHLCV, metadata/SR, RAC)
- **Phase 6, 8** — Xem `IMPLEMENTATION_PLAN.md`

## Chuẩn bị môi trường

### Yêu cầu

- Docker Desktop
- Python 3.12
- `uv` (package manager)

### 0) Cài dependencies (uv)

```bash
uv sync --dev
```

### 1) Cấu hình biến môi trường

```bash
cp .env.example .env   # chỉnh nếu cần
```

ETL sẽ **tự load** `.env` (nếu có) để lấy `DATABASE_URL` và `VNSTOCK_API_KEY`.

### 2) Activate virtualenv & set DATABASE_URL

<details>
<summary><strong>macOS / Linux (bash/zsh)</strong></summary>

```bash
source .venv/bin/activate
export DATABASE_URL="postgresql+psycopg://postgres:postgres@localhost:5432/hybrid_rac"
```

</details>

<details>
<summary><strong>Windows (PowerShell)</strong></summary>

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
$env:DATABASE_URL="postgresql+psycopg://postgres:postgres@localhost:5432/hybrid_rac"
```

</details>

> **Lưu ý:** Nếu đã khai báo `DATABASE_URL` trong `.env`, bước export/set có thể bỏ qua — pipeline tự đọc `.env`.
>
> Các lệnh bên dưới giả sử bạn đã activate venv và set `DATABASE_URL`.

### 3) Khởi động PostgreSQL (TimescaleDB + pgvector)

```bash
docker compose --env-file .env up -d --build
```

### 4) Chạy migrations (Alembic)

```bash
alembic upgrade head
```

### 5) Verify nhanh

```bash
docker exec -it hybrid_rac_db psql -U postgres -d hybrid_rac
```

Trong `psql`:

- `\dx` (có `timescaledb`, `vector`)
- `SELECT hypertable_name FROM timescaledb_information.hypertables;` (có `stock_ohlcv`)
- `\dt` (có các bảng schema Phase 1)
- `SELECT indexname FROM pg_indexes WHERE tablename='pattern_embeddings' ORDER BY indexname;` (có `idx_embedding_hnsw`)

## Chạy Phase 2 (ETL: fetch → clean → ingest)

### Seed dataset nhỏ (khuyến nghị cho CI/dev)

```bash
alembic upgrade head
python -m db.seed_small_dataset
```

### Chạy ETL pipeline (gọi Vnstock)

```bash
# Backfill theo khoảng thời gian
python -m etl.pipeline backfill --symbols VCB FPT --start 2024-01-01 --end 2024-03-31 --chunk-days 365 --concurrency 2 --requests-per-minute 55

# Incremental (tự lấy từ MAX(time) trong DB theo từng mã → end)
python -m etl.pipeline incremental --symbols VCB FPT --end 2026-04-15 --chunk-days 365 --concurrency 2 --requests-per-minute 55
```

> **Chống rate limit (Community 60 req/phút):**
> - Pipeline có **global rate limiter** dùng chung cho mọi thread + **retry/backoff** khi gặp 429.
> - Mặc định `--requests-per-minute` lấy từ `VNSTOCK_REQUESTS_PER_MINUTE` hoặc **55** (để chừa headroom).
> - `--rate-limit-burst` (mặc định 1) cho phép “burst” ngắn tối đa N request.

### Fetch full VN100 (lần đầu: 2010 → nay)

Repo có sẵn danh sách VN100 ở `etl/tickers_vn100.txt`.

```bash
alembic upgrade head

# Backfill toàn bộ VN100 (chia theo năm để tránh timeout)
python -m etl.pipeline backfill \
  --symbols-file etl/tickers_vn100.txt \
  --start 2010-01-01 --end 2026-04-15 \
  --chunk-days 365 --concurrency 4 \
  --requests-per-minute 55
```

Các lần sau chỉ cần incremental:

```bash
python -m etl.pipeline incremental \
  --symbols-file etl/tickers_vn100.txt \
  --end 2026-04-15 --chunk-days 365 --concurrency 4 \
  --requests-per-minute 55
```

> **Windows PowerShell:** để xuống dòng, dùng backtick `` ` `` thay vì `\`.
>
> Ví dụ:
>
> ```powershell
> uv run python -m etl.pipeline incremental `
>   --symbols-file etl/tickers_vn100.txt `
>   --end 2026-04-15 `
>   --chunk-days 365 `
>   --concurrency 4 `
>   --requests-per-minute 55
> ```

## Chạy Phase 3 (Feature Engineering & Preprocessing)

### Tạo sliding windows từ OHLCV trong DB

Đọc dữ liệu OHLCV từ DB, forward-fill ngày thiếu, tạo windows [30 phiên × 5 kênh OHLCV], gắn label theo T+5 return, z-score normalize, và chia train/test theo thời gian (80/20).

```bash
# Tạo windows cho một vài mã
python -m etl.pipeline generate-windows --symbols VCB FPT --output-dir data/windows

# Tạo windows cho toàn bộ VN100
python -m etl.pipeline generate-windows \
  --symbols-file etl/tickers_vn100.txt --output-dir data/windows

# Tuỳ chỉnh tham số
python -m etl.pipeline generate-windows --symbols VCB FPT \
  --start 2015-01-01 --end 2025-12-31 \
  --window-size 30 --horizon 5 --stride 1 \
  --up-threshold 0.02 --down-threshold -0.02 \
  --train-ratio 0.8 --output-dir data/windows
```

Output lưu tại `data/windows/`:
- `train_windows.npz` / `test_windows.npz` — mảng numpy `(N, 30, 5)`
- `train_metadata.csv` / `test_metadata.csv` — symbol, window_start, window_end, label, future_return

### Phát hiện vùng hỗ trợ / kháng cự (S/R Zones)

Dùng thuật toán Pivot Points để phát hiện vùng S/R, tự động INSERT vào bảng `support_resistance_zones`.

```bash
# Detect S/R cho một vài mã
python -m etl.pipeline detect-sr --symbols VCB FPT

# Detect S/R cho toàn bộ VN100
python -m etl.pipeline detect-sr --symbols-file etl/tickers_vn100.txt --order 5
```

### Tests / Lint / Type-check

```bash
ruff check .
mypy .
pytest -q
```

---

## Chạy Phase 7 (FastAPI REST API)

> API auto-load `.env` và cần `DATABASE_URL` trỏ tới PostgreSQL đã chạy + đã `alembic upgrade head`.

### Chạy server

```bash
uv run uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Mở docs:
- Swagger UI: `http://localhost:8000/docs`
- OpenAPI JSON: `http://localhost:8000/openapi.json`

### Endpoints đã có

- `GET /api/health`
- `GET /api/symbols`
- `GET /api/ohlcv/{symbol}?start=YYYY-MM-DD&end=YYYY-MM-DD&limit=...`
- `GET /api/ohlcv/{symbol}/latest?n=...`
- `GET /api/sr-zones/{symbol}?active_only=true|false`
- `GET /api/sr-zones/{symbol}/distance?price=...`
- `POST /api/rac/similar-patterns`
- `POST /api/rac/context`
- `POST /api/rac/full-context`
- `POST /api/rac/predict?persist=true|false`
- `GET /api/rac/predictions/{symbol}?limit=...`

---

## English

This repo implements an **equity pattern recognition** and **stock forecasting** system built on a **hybrid database architecture** running on a **single PostgreSQL 16+** instance:
- **TimescaleDB**: OHLCV time-series storage (hypertables, chunking, compression)
- **pgvector**: `vector(128)` embeddings with **HNSW** similarity search
- **RAC (Retrieval‑Augmented Classification)**: predictions with **quantitative evidence** from Top‑K similar historical patterns (reducing "black-box" behavior)

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

- **Phase 1** ✅ Infrastructure & DB Schema (TimescaleDB, pgvector, stored procedures, migrations)
- **Phase 2** ✅ ETL Pipeline (Vnstock fetch → clean → bulk ingest, idempotent upsert)
- **Phase 3** ✅ Feature Engineering (sliding windows, S/R detection, train/test split)
- **Phase 7** ✅ (partial) FastAPI REST API (healthcheck, OHLCV, metadata/SR)
- **Phase 4–6, 8** — See `IMPLEMENTATION_PLAN.md`

## Environment Setup

### Requirements

- Docker Desktop
- Python 3.12
- `uv` (package manager)

### 0) Install dependencies (uv)

```bash
uv sync --dev
```

### 1) Configure environment

```bash
cp .env.example .env   # edit if needed
```

The ETL pipeline **auto-loads** `.env` (if present) for `DATABASE_URL` and `VNSTOCK_API_KEY`.

### 2) Activate virtualenv & set DATABASE_URL

<details>
<summary><strong>macOS / Linux (bash/zsh)</strong></summary>

```bash
source .venv/bin/activate
export DATABASE_URL="postgresql+psycopg://postgres:postgres@localhost:5432/hybrid_rac"
```

</details>

<details>
<summary><strong>Windows (PowerShell)</strong></summary>

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
$env:DATABASE_URL="postgresql+psycopg://postgres:postgres@localhost:5432/hybrid_rac"
```

</details>

> **Note:** If `DATABASE_URL` is already defined in `.env`, you can skip the export/set step — the pipeline reads `.env` automatically.
>
> All commands below assume you have activated the venv and set `DATABASE_URL`.

### 3) Start PostgreSQL (TimescaleDB + pgvector)

```bash
docker compose --env-file .env up -d --build
```

### 4) Run migrations (Alembic)

```bash
alembic upgrade head
```

### 5) Quick verify

```bash
docker exec -it hybrid_rac_db psql -U postgres -d hybrid_rac
```

## Run Phase 2 (ETL: fetch → clean → ingest)

### Seed a small dataset (recommended for CI/dev)

```bash
alembic upgrade head
python -m db.seed_small_dataset
```

### Run ETL pipeline (calls Vnstock)

```bash
# Backfill a date range
python -m etl.pipeline backfill --symbols VCB FPT --start 2024-01-01 --end 2024-03-31 --chunk-days 365 --concurrency 2 --requests-per-minute 55

# Incremental (from MAX(time) in DB per symbol → end)
python -m etl.pipeline incremental --symbols VCB FPT --end 2026-04-15 --chunk-days 365 --concurrency 2 --requests-per-minute 55
```

> **Rate limiting (Community 60 req/min):**
> - The ETL has a **process-wide rate limiter** shared across all threads + **retry/backoff** on 429.
> - By default, `--requests-per-minute` uses `VNSTOCK_REQUESTS_PER_MINUTE` or **55** (headroom).
> - `--rate-limit-burst` (default 1) allows short bursts up to N requests.

### Fetch full VN100 (first run: 2010 → present)

The repo includes a VN100 ticker list at `etl/tickers_vn100.txt`.

```bash
alembic upgrade head

python -m etl.pipeline backfill \
  --symbols-file etl/tickers_vn100.txt \
  --start 2010-01-01 --end 2026-04-15 \
  --chunk-days 365 --concurrency 4 \
  --requests-per-minute 55
```

After that, use incremental mode:

```bash
python -m etl.pipeline incremental \
  --symbols-file etl/tickers_vn100.txt \
  --end 2026-04-15 --chunk-days 365 --concurrency 4 \
  --requests-per-minute 55
```

> **Windows PowerShell:** use backtick `` ` `` for line continuation (not `\`).

## Run Phase 3 (Feature Engineering & Preprocessing)

### Generate sliding windows from OHLCV in DB

Reads OHLCV from DB, forward-fills missing business days, creates [30 sessions × 5 OHLCV channels] windows, labels by T+5 return, z-score normalizes, and splits train/test chronologically (80/20).

```bash
# Generate windows for a few symbols
python -m etl.pipeline generate-windows --symbols VCB FPT --output-dir data/windows

# Generate windows for full VN100
python -m etl.pipeline generate-windows \
  --symbols-file etl/tickers_vn100.txt --output-dir data/windows

# Custom parameters
python -m etl.pipeline generate-windows --symbols VCB FPT \
  --start 2015-01-01 --end 2025-12-31 \
  --window-size 30 --horizon 5 --stride 1 \
  --up-threshold 0.02 --down-threshold -0.02 \
  --train-ratio 0.8 --output-dir data/windows
```

Output saved to `data/windows/`:
- `train_windows.npz` / `test_windows.npz` — numpy arrays `(N, 30, 5)`
- `train_metadata.csv` / `test_metadata.csv` — symbol, window_start, window_end, label, future_return

### Detect Support / Resistance zones

Uses Pivot Points algorithm to detect S/R zones and INSERTs into the `support_resistance_zones` table.

```bash
# Detect S/R for a few symbols
python -m etl.pipeline detect-sr --symbols VCB FPT

# Detect S/R for full VN100
python -m etl.pipeline detect-sr --symbols-file etl/tickers_vn100.txt --order 5
```

### Tests / Lint / Type-check

```bash
ruff check .
mypy .
pytest -q
```

---

## Run Phase 7 (FastAPI REST API)

> The API auto-loads `.env` and requires `DATABASE_URL` pointing to a running PostgreSQL with migrations applied (`alembic upgrade head`).

### Start the server

```bash
uv run uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Open docs:
- Swagger UI: `http://localhost:8000/docs`
- OpenAPI JSON: `http://localhost:8000/openapi.json`

### Available endpoints

- `GET /api/health`
- `GET /api/symbols`
- `GET /api/ohlcv/{symbol}?start=YYYY-MM-DD&end=YYYY-MM-DD&limit=...`
- `GET /api/ohlcv/{symbol}/latest?n=...`
- `GET /api/sr-zones/{symbol}?active_only=true|false`
- `GET /api/sr-zones/{symbol}/distance?price=...`
- `POST /api/rac/similar-patterns`
- `POST /api/rac/context`
- `POST /api/rac/full-context`
- `POST /api/rac/predict?persist=true|false`
- `GET /api/rac/predictions/{symbol}?limit=...`