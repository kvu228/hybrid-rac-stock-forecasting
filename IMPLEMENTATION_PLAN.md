# Implementation Plan: Hybrid-RAC-Stock

> Cập nhật: 2026-04-19

---

## Phase 1: Infrastructure & DB Schema ✅

- [x] 1.1 Docker Compose (PG16 + TimescaleDB + pgvector)
- [x] 1.2 Custom Dockerfile (`docker/Dockerfile`)
- [x] 1.3 Alembic scaffold + `env.py` đọc `DATABASE_URL` từ env
- [x] 1.4 Migration 0001–0005: Extensions, Hypertable, `pattern_embeddings`, `rac_predictions`, `support_resistance_zones`
- [x] 1.5 Migration 0006: B-Tree + HNSW indexes
- [x] 1.6 Migration 0007: 4 Stored Procedures (`find_similar_patterns`, `compute_rac_context`, `compute_full_rac_context`, `get_distance_to_nearest_sr`)
- [x] 1.7 Migration 0008: Unique constraint `(symbol, time)` cho idempotent ingestion
- [x] 1.8 CI pipeline (GitHub Actions): ruff → mypy → docker compose → alembic → pytest

---

## Phase 2: Data Engineering (ETL) ✅

- [x] 2.1 `etl/vnstock_fetcher.py` — Fetch OHLCV từ Vnstock API (VCI provider), error recovery cho no-data
- [x] 2.2 `etl/data_cleaner.py` — Normalize timezone (UTC), coerce types, deduplicate `(symbol, time)`, validate columns
- [x] 2.3 `etl/ingestion.py` — Bulk COPY vào temp table → UPSERT `ON CONFLICT (symbol, time)` (idempotent)
- [x] 2.4 `etl/pipeline.py` — CLI orchestrator: `seed | backfill | incremental`, ThreadPoolExecutor, date chunking
- [x] 2.5 `etl/tickers_vn100.txt` — 101 mã VN100
- [x] 2.6 `db/seed_small_dataset.py` + `tests/fixtures/ohlcv_small.tsv` — Fixture data cho CI
- [x] 2.7 `tests/test_etl_smoke.py` — Test cleaner normalization + ingestion idempotency
- [x] 2.8 `tests/test_phase1_db_schema.py` — Verify extensions, hypertable, HNSW index, stored procedures
- [x] 2.9 `.env` loading + vnstock API key registration

---

## Phase 3: Feature Engineering & Preprocessing

> **Mục tiêu:** Chuẩn bị dữ liệu đầu vào cho CNN Encoder — từ raw OHLCV tạo sliding windows + labels.

- [x] 3.1 **Preprocessing module** (`etl/feature_engineer.py`)
    - [x] Z-score normalization per window (chuẩn hóa mỗi cửa sổ độc lập)
    - [x] Relative return transformation (close-to-close %)
    - [x] Forward-fill giá cho missing sessions (align trading calendar)
- [x] 3.2 **Sliding Window Generator**
    - [x] Tạo windows [30 sessions × 5 channels (OHLCV)] với stride=1
    - [x] Gắn label cho mỗi window: T+5 return → {0: Down, 1: Neutral, 2: Up} theo threshold
    - [x] Lưu metadata: symbol, window_start, window_end, label, future_return
- [x] 3.3 **S/R Zone Detection** (`etl/sr_detector.py`)
    - [x] Thuật toán phát hiện Pivot Points / đảo chiều trên chuỗi giá
    - [x] Tính strength (số lần giá test vùng)
    - [x] INSERT vào bảng `support_resistance_zones`
- [x] 3.4 **Train/Test Split**
    - [x] Phân chia theo thời gian (80% đầu / 20% cuối), KHÔNG random
    - [x] Export metadata split để các phase sau dùng chung

---

## Phase 4: ML & Embedding Pipeline

> **Mục tiêu:** Train CNN Encoder → tạo embeddings → insert vào pgvector. Lưu ý: ML chỉ phục vụ tạo dữ liệu cho DB, không phải trọng tâm đề tài.

- [x] 4.1 **1D-CNN Encoder** (`ml/cnn_encoder.py`)
    - [x] Kiến trúc: Input [batch × 30 × 5] → Conv1D layers → Output vector(128)
    - [x] Framework: PyTorch
    - [x] Loss function: **Triplet** (`--loss triplet`, `TripletMarginLoss` trên embedding); contrastive tùy chọn mở rộng sau
- [x] 4.2 **Training Pipeline** (`ml/train_pipeline.py`)
    - [x] Tối thiểu: smoke-train để tạo encoder weights (phục vụ sinh embeddings cho DB)
    - [x] Train/evaluate theo train/test split Phase 3 (`test_accuracy` với CE; `--metrics-out` ghi JSON)
- [x] 4.3 **Embedding Generator** (`ml/embedding_generator.py`)
    - [x] Load trained CNN → encode windows → vector(128)
    - [x] Batch INSERT vào `pattern_embeddings` (kèm label + future_return)
    - [x] Sau insert: `verify_hnsw_index_used()` (EXPLAIN KNN mẫu) → field `hnsw_index_used` trong `InsertStats`
- [x] 4.4 **SVM Classifier** (`ml/svm_classifier.py`)
    - [x] Framework: scikit-learn (wrapper train/load/predict)
    - [x] Train/eval trên split thời gian: CLI `python -m ml.svm_eval` (embedding CNN + `classification_report`)
- [x] 4.5 **Dependencies**
    - [x] `uv add torch scikit-learn numpy pandas`

---

## Phase 5: RAC Application Layer

> **Mục tiêu:** Python wrapper gọi Stored Procedures + kết nối ML output → prediction pipeline hoàn chỉnh.

- [x] 5.1 **RAC Retriever** (`rac/retriever.py`)
    - [x] Wrapper gọi `find_similar_patterns()` từ Python
    - [x] Input: query embedding (128-d), k, threshold, optional symbol filter
    - [x] Output: danh sách neighbors (id, symbol, label, return, distance, window range)
- [x] 5.2 **RAC Context Enricher** (`rac/context_enricher.py`)
    - [x] Wrapper gọi `compute_rac_context()` — aggregate KNN stats
    - [x] Wrapper gọi `compute_full_rac_context()` — hybrid HNSW + B-Tree query
- [x] 5.3 **RAC Classifier** (`rac/rac_classifier.py`)
    - [x] Orchestrate: compute_full_rac_context → (optional) SVM predict → persist `rac_predictions`
- [x] 5.4 **Explainer** (`rac/explainer.py`)
    - [x] Format evidence payload (neighbors + label_distribution + confidence)
- [x] 5.5 **Tests**
    - [x] Test stored procedures + RAC endpoints integration

---

## Phase 6: Benchmarking ✅

> **Mục tiêu:** Đo lường hiệu suất DB — đây là phần cốt lõi của báo cáo. Mọi claim phải có `EXPLAIN ANALYZE`.

- [x] 6.1 **HNSW vs Sequential Scan** (`benchmark/hnsw_vs_seqscan.py`)
    - [x] Tắt/bật index scan → so sánh latency
    - [x] Đo: query latency P50/P95/P99, pages read (buffers hit/read), Recall@K vs exact
- [x] 6.2 **Hybrid Search** (`benchmark/hybrid_search_bench.py`)
    - [x] So sánh: HNSW only vs B-Tree filter + HNSW
    - [x] Chạy `EXPLAIN ANALYZE` → capture execution plan
- [x] 6.3 **In-DB vs App-side Computing** (`benchmark/indb_vs_appside.py`)
    - [x] Phương án A: Query K raw vectors → Python tính stats
    - [x] Phương án B: Gọi `compute_rac_context()` → DB trả kết quả tổng hợp
    - [x] Đo: end-to-end time, bytes transferred, round-trips
- [x] 6.4 **HNSW Parameter Sweep** (`benchmark/hnsw_param_sweep.py`)
    - [x] Grid: m=[8,16,32,64] × ef_construction=[64,128,200,400] × ef_search=[40,100,200,400]
    - [x] Đo: index build time, index size, query latency, Recall@10/20, RAM usage
- [x] 6.5 **TimescaleDB Chunking** (`benchmark/chunk_size_bench.py`)
    - [x] So sánh chunk intervals: 1 week, 1 month, 3 months
    - [x] Đo: range query latency, compression ratio, chunk exclusion effectiveness
- [x] 6.6 **Export kết quả** → `benchmark/results/` (CSV/JSON cho báo cáo + Streamlit)

---

## Phase 7: FastAPI REST API ✅

> **Mục tiêu:** HTTP API thay thế CLI, phục vụ làm backend cho Streamlit dashboard.

- [x] 7.1 **App skeleton** (`api/main.py`)
    - [x] FastAPI app + lifespan (async connection pool via `psycopg_pool`)
    - [x] CORS middleware
    - [x] `api/deps.py` — shared dependency `get_db_conn`
    - [x] `api/schemas.py` — Pydantic request/response models
- [x] 7.2 **ETL Router** (`api/routers/etl.py`)
    - [x] `POST /api/etl/seed` — load fixture
    - [x] `POST /api/etl/backfill` — background task, trả job_id
    - [x] `POST /api/etl/incremental` — background task
    - [x] `GET /api/etl/status/{job_id}` — poll trạng thái
- [x] 7.3 **OHLCV Router** (`api/routers/ohlcv.py`)
    - [x] `GET /api/symbols` — danh sách symbols trong DB
    - [x] `GET /api/ohlcv/{symbol}?start=...&end=...` — OHLCV theo time range
    - [x] `GET /api/ohlcv/{symbol}/latest?n=30` — N phiên gần nhất
- [x] 7.4 **RAC Router** (`api/routers/rac.py`)
    - [x] `POST /api/rac/query-embedding` — cửa sổ OHLCV 30 phiên → embedding (phục vụ Streamlit)
    - [x] `POST /api/rac/similar-patterns` — gọi `find_similar_patterns()`
    - [x] `POST /api/rac/context` — gọi `compute_rac_context()`
    - [x] `POST /api/rac/full-context` — gọi `compute_full_rac_context()` (★ Hybrid Query)
    - [x] `POST /api/rac/predict` — dự báo + (optional) persist `rac_predictions`
    - [x] `GET /api/rac/predictions/{symbol}?limit=20` — lịch sử predictions
- [x] 7.5 **Metadata Router** (`api/routers/metadata.py`)
    - [x] `GET /api/sr-zones/{symbol}` — S/R zones active
    - [x] `GET /api/sr-zones/{symbol}/distance?price=...` — gọi `get_distance_to_nearest_sr()`
- [x] 7.6 **Benchmark Router** (`api/routers/benchmark.py`)
    - [x] `POST /api/benchmark/explain` — `EXPLAIN (ANALYZE, BUFFERS)` (SQL whitelist: `hnsw_knn`, `seqscan_knn`, `hybrid_context`)
    - [x] `GET /api/benchmark/stats` — `pg_stat_statements` (graceful nếu extension chưa bật)
    - [x] `GET /api/benchmark/results` + `GET /api/benchmark/results/{name}` — đọc JSON trong `benchmark/results/` (phục vụ Streamlit)
- [x] 7.7 **Dependencies**: `uv add uvicorn psycopg-pool`
- [x] 7.8 **Tests**: API integration tests với `httpx.AsyncClient`

---

## Phase 8: Streamlit Dashboard ✅

> **Mục tiêu:** Visualization cho demo bảo vệ đề tài. Gọi FastAPI, KHÔNG kết nối DB trực tiếp.

- [x] 8.1 **App entry** (`streamlit_app/app.py`) — multipage + sidebar URL API
- [x] 8.2 **Trang 1: OHLCV Explorer** (`pages/1_ohlcv_chart.py`)
    - [x] Dropdown symbol + date range picker
    - [x] Plotly Candlestick + Volume bars
    - [x] S/R zones overlay (horizontal lines)
- [x] 8.3 **Trang 2: Similar Patterns** (`pages/2_similar_patterns.py`) ★ Highlight
    - [x] Chọn query window (symbol + session slider)
    - [x] Grid hiển thị K neighbors: mini candlestick charts
    - [x] Normalized overlay chart (so sánh hình dáng cùng scale [0,1])
    - [x] Bảng: cosine_distance, label, future_return
    - [x] Pie chart: label distribution (neighbors)
- [x] 8.4 **Trang 3: RAC Prediction** (`pages/3_rac_prediction.py`)
    - [x] Hiển thị 3 blocks từ `compute_full_rac_context()`:
        - Block 1 — KNN Stats: metric cards + pie chart
        - Block 2 — S/R Context: gauge chart (sr_position_ratio)
        - Block 3 — Evidence: neighbor IDs (+ caption trang Similar patterns)
- [x] 8.5 **Trang 4: Benchmark** (`pages/4_benchmark.py`)
    - [x] Bar chart từ JSON benchmark (HNSW vs exact p50/p95/p99 khi có trong file)
    - [x] `pg_stat_statements` + danh sách artifact JSON qua API
    - [x] Query plan viewer (`POST /api/benchmark/explain`)
- [x] 8.6 **Shared components**
    - [x] `components/candlestick.py` — Plotly candlestick helper
    - [x] `components/similarity.py` — Normalized overlay helper
- [x] 8.7 **Dependencies**: `uv sync --dev` (streamlit, plotly; httpx đã có trong dev)

---

## Thứ tự triển khai đề xuất

```
Phase 3 (Feature Eng.)  ──→  Phase 4 (ML + Embeddings)  ──→  Phase 5 (RAC Layer)
                                                                      │
Phase 7 (API) ◄──────────────────────────────────────────────────────┘
    │
    ▼
Phase 8 (Streamlit)  ←──  Phase 6 (Benchmark)
```

**Ghi chú:**
- Phase 3 → 4 → 5 là đường critical path (phải có embeddings trước khi chạy RAC)
- Phase 7 (API) có thể bắt đầu song song với Phase 4 cho phần OHLCV + ETL routers (không cần embeddings)
- Phase 6 (Benchmark) cần Phase 5 hoàn thành để có dữ liệu embeddings đầy đủ
- Phase 8 (Streamlit) bắt đầu sau khi API có ít nhất OHLCV + RAC endpoints