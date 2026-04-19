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
- **Phase 6** ✅ DB benchmarks (HNSW vs exact, hybrid search, in-DB vs app, HNSW sweep, chunk intervals → `benchmark/results/`)
- **Phase 7** ✅ FastAPI REST API (OHLCV, metadata/SR, RAC, ETL jobs, benchmark, `query-embedding`)
- **Phase 8** ✅ Streamlit dashboard (`streamlit_app/`, Plotly, gọi API — `make streamlit`)

## Chuẩn bị môi trường

### Yêu cầu

- Docker Desktop
- Python 3.12
- `uv` (package manager)

### 0) Cài dependencies (uv)

```bash
uv sync --dev
```

### 0b) Makefile (tuỳ chọn — gom các CLI hay dùng)

Repo có [`Makefile`](Makefile) để gom các lệnh thường dùng (`db-up`, `migrate`, ETL VN100, Phase 3/4 — windows + S/R + train/embedding, **Phase 6 benchmarks**, lint/test, **API**, **Streamlit**…).

```bash
make help
```

Ví dụ nhanh:

```bash
make db-up
make migrate
make etl-backfill-vn100          # END mặc định = ngày hôm nay (theo `uv run python`)
make etl-backfill-vn100 END=2026-04-19
make etl-incremental-vn100 END=2026-04-19
make api
make streamlit                    # Phase 8 — cần API (mặc định http://127.0.0.1:8000)
```

### Makefile — thứ tự gợi ý trên DB thật (Phase 2 → 3 → 4, VN100)

Giả sử `DATABASE_URL` đã đúng (`.env` hoặc export), TimescaleDB/pgvector đã chạy và đã migrate.

1. **Khởi tạo DB (một lần / khi đổi môi trường)**  
   `make db-up` → `make migrate`

2. **Phase 2 — nạp OHLCV đủ lịch sử**  
   Lần đầu: `make etl-backfill-vn100` (tuỳ chỉnh `START=`, `END=`, `CONCURRENCY=`, `RPM=`).  
   Các lần sau: `make etl-incremental-vn100 END=YYYY-MM-DD`.

3. **Phase 3 — feature + metadata cho hybrid context**  
   Gói một lệnh (đúng thứ tự: export windows → S/R DB):  
   `make phase3-vn100`  
   Tách riêng nếu cần:  
   - `make etl-generate-windows-vn100` — ghi `data/windows/` (`WINDOWS_OUT=...`, tùy chọn lọc ngày `WIN_START=` / `WIN_END=`).  
   - `make etl-detect-sr-vn100` — đổ `support_resistance_zones` (`SR_ORDER=5` mặc định).  
   Theo từng mã: `make etl-generate-windows-symbol SYMBOL=VCB`, `make etl-detect-sr-symbol SYMBOL=VCB`.  
   Tuỳ chọn (sau nhiều lần chạy `detect-sr`, để bảng không phình): `make etl-purge-inactive-sr-vn100` hoặc `make etl-purge-inactive-sr-all` (hoặc API `POST /api/sr-zones/purge-inactive`).

4. **Phase 4 — encoder → `pattern_embeddings`**  
   - Smoke nhanh: `make ml-train-encoder-synthetic` (ghi `ML_ENCODER_OUT`, mặc định `ml/model_store/cnn_encoder.pt`).  
   - Train trực tiếp từ bảng **`stock_ohlcv`** (không cần file TSV): `make ml-train-encoder-db` (theo `TICKERS_VN100`) hoặc `make ml-train-encoder-db-symbol SYMBOL=VCB`; có thể lọc ngày `WIN_START=` / `WIN_END=` như bước export windows.  
   - Hoặc train từ TSV OHLCV (cột `time` + OHLCV):  
     `make ml-train-encoder OHLCV_TSV=đường/dẫn/export.tsv ML_ENCODER_EPOCHS=8 ML_ENCODER_BATCH=256`  
   - Sinh embedding vào DB (xoá embedding cũ của mã rồi insert lại):  
     - Cả danh sách VN100 (tuần tự, dễ đo thời gian từng mã): `make ml-embed-vn100`  
     - Một mã: `make ml-embed-symbol SYMBOL=VCB`  
   Tuỳ chỉnh: `ML_EMBED_BATCH=`, `ML_DEVICE=` (hoặc biến môi trường `TORCH_DEVICE`).

5. **Quan sát tải HNSW / kích thước chỉ mục (tuỳ chọn)**  
   Trong `psql`, ví dụ:  
   `SELECT indexname, pg_size_pretty(pg_relation_size(indexrelid::regclass)) AS index_size FROM pg_stat_user_indexes WHERE relname = 'pattern_embeddings' ORDER BY indexname;`

6. **Phase 6 — benchmark hiệu năng DB (sau khi đã có `pattern_embeddings` + `stock_ohlcv`)**  
   Chi tiết: [`benchmark/README.md`](benchmark/README.md). **Cảnh báo:** các target HNSW/sweep **DROP và CREATE lại** `idx_embedding_hnsw` — chỉ chạy trên DB dev.  
   - `make bench-smoke` — test nhẹ (parse `EXPLAIN`; có thêm test DB nếu có `DATABASE_URL`).  
   - `make bench-hnsw-vs-seqscan` — HNSW vs quét chính xác, P50/P95/P99, recall (override: `BENCH_K=`, `BENCH_N_QUERIES=`, `BENCH_SEED=`, `BENCH_EF_SEARCH=`).  
   - `make bench-hybrid`, `make bench-indb-appside` — hybrid symbol filter; in-DB vs app-side RAC context.  
   - `make bench-hnsw-sweep-quick` / `make bench-hnsw-sweep` — lưới tham số HNSW (bản đầy đủ rất lâu).  
   - `make bench-chunk` — so sánh chunk 1 tuần / 1 tháng / 3 tháng (tạo bảng `bench_stock_ohlcv_*`); `make bench-chunk-teardown` để xóa các bảng đó.

Xem `make help` để liệt kê đầy đủ target và biến override.

`make ml-embed-vn100` dùng `grep` và vòng lặp `while` trong shell (POSIX/Git Bash). Nếu GNU Make của bạn gọi `cmd.exe` làm shell mặc định, dùng Git Bash hoặc lặp tay `make ml-embed-symbol SYMBOL=...` / `uv run python -m ml.embedding_generator ...` theo từng mã.

**Windows:** PowerShell thường **không có `make` sẵn**. Cách phổ biến là dùng **Git Bash** (kèm Git for Windows) hoặc cài GNU Make riêng, rồi chạy `make ...` từ đó.

Nếu bạn muốn copy/paste one-liner PowerShell (tránh lỗi line continuation), chạy:

```bash
make windows-ps-backfill-vn100
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

Makefile tương ứng: `make etl-generate-windows-vn100`, `make etl-generate-windows-symbol SYMBOL=VCB`, `make etl-detect-sr-vn100`, `make etl-detect-sr-symbol SYMBOL=VCB`, hoặc gói `make phase3-vn100`; dọn inactive: `make etl-purge-inactive-sr-vn100` / `make etl-purge-inactive-sr-all` (xem mục **Makefile — thứ tự** ở trên).

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

### Dọn dẹp zone không còn active (`is_active = FALSE`)

Mỗi lần `detect-sr` chạy lại, các zone cũ bị đánh dấu inactive và zone mới được insert, nên bảng có thể tích lũy dòng lịch sử. Để xóa hẳn các dòng inactive:

```bash
# Chỉ các mã trong file (ví dụ VN100)
python -m etl.pipeline purge-inactive-sr --symbols-file etl/tickers_vn100.txt

# Toàn bộ bảng (mọi mã)
python -m etl.pipeline purge-inactive-sr --all-inactive
```

Makefile: `make etl-purge-inactive-sr-vn100`, `make etl-purge-inactive-sr-all`.

API (nên bảo vệ ở môi trường production): `POST /api/sr-zones/purge-inactive` với JSON `{"symbols": ["VCB"]}` hoặc `{"all_inactive": true}` (không dùng đồng thời `symbols` với `all_inactive`).

### Tests / Lint / Type-check

```bash
ruff check .
mypy .
pytest -q
```

## Chạy Phase 4 (ML & embeddings → `pattern_embeddings`)

Makefile: `make ml-train-encoder-synthetic`, `make ml-train-encoder`, `make ml-train-encoder-db`, `make ml-train-encoder-db-symbol SYMBOL=...`, `make ml-embed-symbol SYMBOL=...`, `make ml-embed-vn100` (thứ tự tổng thể nằm ở mục **Makefile — thứ tự** phía trên).

**Train encoder (tạo file weights, mặc định `ml/model_store/cnn_encoder.pt`):**

```bash
# Smoke (OHLCV tổng hợp)
uv run python -m ml.train_pipeline --synthetic --epochs 1 --batch-size 32 --out ml/model_store/cnn_encoder.pt

# Từ Postgres (bảng stock_ohlcv; cần DATABASE_URL hoặc --database-url)
uv run python -m ml.train_pipeline --from-db --symbols-file etl/tickers_vn100.txt --epochs 8 --batch-size 256 --out ml/model_store/cnn_encoder.pt
# Một mã + lọc ngày:
uv run python -m ml.train_pipeline --from-db --symbols VCB --start 2015-01-01 --end 2025-12-31 --epochs 8 --out ml/model_store/cnn_encoder.pt

# Từ file TSV (cột time + OHLCV; xem fixture tests/fixtures/ohlcv_small.tsv)
uv run python -m ml.train_pipeline --ohlcv-tsv path/to/ohlcv.tsv --epochs 8 --batch-size 256 --out ml/model_store/cnn_encoder.pt
```

**Sinh embedding và INSERT vào Postgres (`--truncate-symbol` xoá embedding cũ của mã đó trước khi nạp lại):**

```bash
uv run python -m ml.embedding_generator --symbol VCB --truncate-symbol --model ml/model_store/cnn_encoder.pt --batch-size 512
```

---

## Chạy Phase 6 (Benchmark DB)

Cần `DATABASE_URL`, schema đã migrate, dữ liệu **`pattern_embeddings`** (các script vector) và **`stock_ohlcv`** (chunk benchmark). Kết quả: `benchmark/results/` + `benchmark/results/explain/` (gitignored).

**Makefile (khuyến nghị):** `make bench-smoke`, `make bench-hnsw-vs-seqscan`, `make bench-hybrid`, `make bench-indb-appside`, `make bench-hnsw-sweep-quick`, `make bench-hnsw-sweep`, `make bench-chunk`, `make bench-chunk-teardown`. Biến: `BENCH_K`, `BENCH_N_QUERIES`, `BENCH_SEED`, `BENCH_EF_SEARCH`.

**Tương đương `uv run`:**

```bash
uv run pytest -q tests/test_benchmark_smoke.py
uv run python -m benchmark.hnsw_vs_seqscan --k 20 --n-queries 30 --seed 42 --ef-search 100
uv run python -m benchmark.hybrid_search_bench --k 20 --n-queries 30
uv run python -m benchmark.indb_vs_appside --k 20 --n-queries 30
uv run python -m benchmark.hnsw_param_sweep --quick
uv run python -m benchmark.chunk_size_bench
```

Xem thêm [`benchmark/README.md`](benchmark/README.md).

---

## Chạy Phase 7 (FastAPI REST API)

> API auto-load `.env` và cần `DATABASE_URL` trỏ tới PostgreSQL đã chạy + đã `alembic upgrade head`.

### Chạy server

```bash
make api
# hoặc:
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
- `POST /api/sr-zones/purge-inactive` (xóa dòng S/R có `is_active=false`; xem mục dọn dẹp Phase 3)
- `POST /api/rac/similar-patterns`
- `POST /api/rac/context`
- `POST /api/rac/full-context`
- `POST /api/rac/predict?persist=true|false`
- `GET /api/rac/predictions/{symbol}?limit=...`
- `POST /api/etl/seed` — nạp fixture (background job → `GET /api/etl/status/{job_id}`)
- `POST /api/etl/backfill` — body: `symbols`, `start`, `end`, …
- `POST /api/etl/incremental` — body: `symbols`, optional `end`
- `GET /api/etl/status/{job_id}`
- `POST /api/rac/query-embedding` — `symbol` + `window_end` → embedding 128 + OHLCV 30 phiên (cần file encoder: `ml/model_store/cnn_encoder.pt` hoặc env `CNN_ENCODER_PATH`)
- `POST /api/benchmark/explain` — `query_kind`: `hnsw_knn` | `seqscan_knn` | `hybrid_context`
- `GET /api/benchmark/stats` — tóm tắt `pg_stat_statements` (nếu extension có)
- `GET /api/benchmark/results` — danh sách file JSON trong `benchmark/results/`
- `GET /api/benchmark/results/{name}`

---

## Chạy Phase 8 (Streamlit)

> Giao diện chỉ gọi **FastAPI**, không kết nối PostgreSQL trực tiếp. Cần encoder đã train cho trang **Similar patterns** / **RAC** (`POST /api/rac/query-embedding`).

1. Chạy API (terminal 1): `make api` hoặc `uv run uvicorn api.main:app --reload --host 0.0.0.0 --port 8000`
2. Chạy dashboard (terminal 2):

```bash
make streamlit
```

Tuỳ chọn Makefile: `ST_PORT=8501` `ST_HOST=127.0.0.1` `ST_APP=streamlit_app/app.py`.

Trong sidebar từng trang có thể đổi **API base URL** (mặc định `http://127.0.0.1:8000`). Biến môi trường tương đương: `API_BASE_URL`.

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
- **Phase 4** ✅ (core) ML & Embeddings (CNN encoder + embedding generator into `pattern_embeddings`)
- **Phase 5** ✅ RAC application layer (stored procedure wrappers + persist predictions)
- **Phase 6** ✅ Database benchmarks (HNSW vs exact, hybrid search, in-DB vs app, HNSW sweep, chunk intervals → `benchmark/results/`)
- **Phase 7** ✅ FastAPI REST API (OHLCV, metadata/SR, RAC, ETL jobs, benchmark, `query-embedding`)
- **Phase 8** ✅ Streamlit dashboard (`streamlit_app/`, Plotly, API-backed — `make streamlit`)

## Environment Setup

### Requirements

- Docker Desktop
- Python 3.12
- `uv` (package manager)
- (Optional) GNU **Make** if you want to use the [`Makefile`](Makefile) shortcuts

### 0) Install dependencies (uv)

```bash
uv sync --dev
```

### 0b) Makefile (optional — common CLI shortcuts)

This repo includes a [`Makefile`](Makefile) that wraps frequent commands (`db-up`, `migrate`, VN100 ETL, Phase 3/4 windows + S/R + train/embeddings, **Phase 6 benchmarks**, lint/tests, **API**, **Streamlit**, …).

```bash
make help
```

Quick examples:

```bash
make db-up
make migrate
make etl-backfill-vn100          # END defaults to "today" (via `uv run python`)
make etl-backfill-vn100 END=2026-04-19
make etl-incremental-vn100 END=2026-04-19
make api
make streamlit                    # Phase 8 — requires API (default http://127.0.0.1:8000)
```

### Makefile — suggested order on a real DB (Phase 2 → 4, VN100)

Assume `DATABASE_URL` is set (via `.env` or your shell), Postgres (TimescaleDB + pgvector) is running, and migrations are applied.

1. **Bootstrap DB** — `make db-up` → `make migrate`
2. **Phase 2 — load enough OHLCV history** — first run: `make etl-backfill-vn100` (override `START=`, `END=`, `CONCURRENCY=`, `RPM=` as needed). Later: `make etl-incremental-vn100 END=YYYY-MM-DD`.
3. **Phase 3 — windows export + S/R metadata for hybrid context** — one shot (ordered: windows → S/R): `make phase3-vn100`. Or split: `make etl-generate-windows-vn100` then `make etl-detect-sr-vn100`. Per symbol: `make etl-generate-windows-symbol SYMBOL=VCB`, `make etl-detect-sr-symbol SYMBOL=VCB`. Optional date filters: `WIN_START=`, `WIN_END=`. Output dir: `WINDOWS_OUT=` (default `data/windows`). Optional (after many `detect-sr` runs): `make etl-purge-inactive-sr-vn100` or `make etl-purge-inactive-sr-all`, or `POST /api/sr-zones/purge-inactive`.
4. **Phase 4 — encoder → `pattern_embeddings`** — smoke: `make ml-train-encoder-synthetic`. Train straight from **`stock_ohlcv`**: `make ml-train-encoder-db` or `make ml-train-encoder-db-symbol SYMBOL=VCB` (optional `WIN_START=` / `WIN_END=`). Or train from a TSV: `make ml-train-encoder OHLCV_TSV=path/to/export.tsv ML_ENCODER_EPOCHS=8`. Generate vectors + insert: `make ml-embed-vn100` (sequential over `TICKERS_VN100`) or `make ml-embed-symbol SYMBOL=VCB`. Tune `ML_EMBED_BATCH=`, `ML_DEVICE=` (or `TORCH_DEVICE`).
5. **Optional: inspect HNSW / index sizes in `psql`** — e.g. `SELECT indexname, pg_size_pretty(pg_relation_size(indexrelid::regclass)) AS index_size FROM pg_stat_user_indexes WHERE relname = 'pattern_embeddings' ORDER BY indexname;`

6. **Phase 6 — database benchmarks** (after **`pattern_embeddings`** + **`stock_ohlcv`** exist) — see [`benchmark/README.md`](benchmark/README.md). **Warning:** HNSW/sweep targets **DROP and recreate** `idx_embedding_hnsw` — use a dev database only.  
   - `make bench-smoke` — lightweight tests (optional live DB checks when `DATABASE_URL` is set).  
   - `make bench-hnsw-vs-seqscan` — HNSW vs exact KNN + `EXPLAIN` (override `BENCH_K`, `BENCH_N_QUERIES`, `BENCH_SEED`, `BENCH_EF_SEARCH`).  
   - `make bench-hybrid`, `make bench-indb-appside` — symbol-filtered KNN; in-DB vs app-side RAC context.  
   - `make bench-hnsw-sweep-quick` / `make bench-hnsw-sweep` — HNSW parameter grid (full grid is very slow).  
   - `make bench-chunk` — compare 1w / 1m / 3m chunk intervals (creates `bench_stock_ohlcv_*` tables); `make bench-chunk-teardown` drops them.

Run `make help` for the full target list.

`make ml-embed-vn100` uses `grep` + a `while` read loop (POSIX / Git Bash). If your GNU Make defaults to `cmd.exe` as the recipe shell, use Git Bash or run `make ml-embed-symbol SYMBOL=...` per ticker.

**Windows:** PowerShell often **does not ship with `make`**. Typical options are **Git Bash** (ships with Git for Windows) or installing GNU Make separately, then run `make ...` from that environment.

For a PowerShell-friendly one-liner (avoids continuation pitfalls), run:

```bash
make windows-ps-backfill-vn100
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

Makefile equivalents: `make etl-generate-windows-vn100`, `make etl-generate-windows-symbol SYMBOL=VCB`, `make etl-detect-sr-vn100`, `make etl-detect-sr-symbol SYMBOL=VCB`, or `make phase3-vn100`; purge inactive rows: `make etl-purge-inactive-sr-vn100` / `make etl-purge-inactive-sr-all` (see **Makefile — suggested order** above).

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

### Purge inactive S/R rows (`is_active = FALSE`)

Each `detect-sr` run deactivates prior zones and inserts new ones, so inactive rows accumulate. To delete inactive rows permanently:

```bash
# Only symbols listed in a file (e.g. VN100)
python -m etl.pipeline purge-inactive-sr --symbols-file etl/tickers_vn100.txt

# Entire table (all symbols)
python -m etl.pipeline purge-inactive-sr --all-inactive
```

Makefile: `make etl-purge-inactive-sr-vn100`, `make etl-purge-inactive-sr-all`.

API (protect in production): `POST /api/sr-zones/purge-inactive` with JSON `{"symbols": ["VCB"]}` or `{"all_inactive": true}` (do not combine `symbols` with `all_inactive`).

### Tests / Lint / Type-check

```bash
ruff check .
mypy .
pytest -q
```

## Run Phase 4 (ML & embeddings → `pattern_embeddings`)

Makefile: `make ml-train-encoder-synthetic`, `make ml-train-encoder`, `make ml-train-encoder-db`, `make ml-train-encoder-db-symbol SYMBOL=...`, `make ml-embed-symbol SYMBOL=...`, `make ml-embed-vn100`.

**Train encoder (writes weights, default `ml/model_store/cnn_encoder.pt`):**

```bash
# Smoke (synthetic OHLCV)
uv run python -m ml.train_pipeline --synthetic --epochs 1 --batch-size 32 --out ml/model_store/cnn_encoder.pt

# From Postgres (stock_ohlcv; needs DATABASE_URL or --database-url)
uv run python -m ml.train_pipeline --from-db --symbols-file etl/tickers_vn100.txt --epochs 8 --batch-size 256 --out ml/model_store/cnn_encoder.pt
# Single symbol + date filter:
uv run python -m ml.train_pipeline --from-db --symbols VCB --start 2015-01-01 --end 2025-12-31 --epochs 8 --out ml/model_store/cnn_encoder.pt

# From a TSV (must include `time` + OHLCV columns; see tests/fixtures/ohlcv_small.tsv)
uv run python -m ml.train_pipeline --ohlcv-tsv path/to/ohlcv.tsv --epochs 8 --batch-size 256 --out ml/model_store/cnn_encoder.pt
```

**Generate embeddings and INSERT into Postgres (`--truncate-symbol` deletes prior rows for that symbol):**

```bash
uv run python -m ml.embedding_generator --symbol VCB --truncate-symbol --model ml/model_store/cnn_encoder.pt --batch-size 512
```

---

## Run Phase 6 (Database benchmarks)

Requires `DATABASE_URL`, applied migrations, populated **`pattern_embeddings`** (vector scripts) and **`stock_ohlcv`** (chunk benchmark). Outputs go to `benchmark/results/` and `benchmark/results/explain/` (gitignored).

**Makefile (recommended):** `make bench-smoke`, `make bench-hnsw-vs-seqscan`, `make bench-hybrid`, `make bench-indb-appside`, `make bench-hnsw-sweep-quick`, `make bench-hnsw-sweep`, `make bench-chunk`, `make bench-chunk-teardown`. Override with `BENCH_K`, `BENCH_N_QUERIES`, `BENCH_SEED`, `BENCH_EF_SEARCH`.

**Equivalent `uv run`:**

```bash
uv run pytest -q tests/test_benchmark_smoke.py
uv run python -m benchmark.hnsw_vs_seqscan --k 20 --n-queries 30 --seed 42 --ef-search 100
uv run python -m benchmark.hybrid_search_bench --k 20 --n-queries 30
uv run python -m benchmark.indb_vs_appside --k 20 --n-queries 30
uv run python -m benchmark.hnsw_param_sweep --quick
uv run python -m benchmark.chunk_size_bench
```

See also [`benchmark/README.md`](benchmark/README.md).

---

## Run Phase 7 (FastAPI REST API)

> The API auto-loads `.env` and requires `DATABASE_URL` pointing to a running PostgreSQL with migrations applied (`alembic upgrade head`).

### Start the server

```bash
make api
# or:
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
- `POST /api/sr-zones/purge-inactive` (delete S/R rows with `is_active=false`; see Phase 3 purge section)
- `POST /api/rac/similar-patterns`
- `POST /api/rac/context`
- `POST /api/rac/full-context`
- `POST /api/rac/predict?persist=true|false`
- `GET /api/rac/predictions/{symbol}?limit=...`
- `POST /api/etl/seed` — load fixture (async job → `GET /api/etl/status/{job_id}`)
- `POST /api/etl/backfill`, `POST /api/etl/incremental`, `GET /api/etl/status/{job_id}`
- `POST /api/rac/query-embedding` — `symbol` + `window_end` → 128-d embedding + 30 OHLCV bars (weights: `ml/model_store/cnn_encoder.pt` or `CNN_ENCODER_PATH`)
- `POST /api/benchmark/explain` — whitelisted `query_kind`: `hnsw_knn`, `seqscan_knn`, `hybrid_context`
- `GET /api/benchmark/stats`, `GET /api/benchmark/results`, `GET /api/benchmark/results/{name}`

---

## Run Phase 8 (Streamlit)

> The UI talks to **FastAPI only** (no direct Postgres). Train an encoder first for **Similar patterns** / **RAC** (`POST /api/rac/query-embedding`).

1. Start the API (terminal 1): `make api` or `uv run uvicorn api.main:app --reload --host 0.0.0.0 --port 8000`
2. Start the dashboard (terminal 2): `make streamlit`

Makefile overrides: `ST_PORT`, `ST_HOST`, `ST_APP`. Sidebar **API base URL** per page (default `http://127.0.0.1:8000`); env: `API_BASE_URL`.