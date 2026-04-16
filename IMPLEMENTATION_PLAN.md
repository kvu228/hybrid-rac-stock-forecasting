# Implementation Plan: Hybrid-RAC-Stock

> Cập nhật: 2026-04-16

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

- [ ] 4.1 **1D-CNN Encoder** (`ml/cnn_encoder.py`)
    - [ ] Kiến trúc: Input [batch × 30 × 5] → Conv1D layers → Output vector(128)
    - [ ] Framework: PyTorch
    - [ ] Loss function: Triplet loss hoặc contrastive loss (để embedding gần nhau khi pattern giống)
- [ ] 4.2 **Training Pipeline** (`ml/train_pipeline.py`)
    - [ ] Dataloader từ sliding windows (Phase 3)
    - [ ] Train trên tập train (80%)
    - [ ] Save model weights → `ml/model_store/`
- [ ] 4.3 **Embedding Generator** (`ml/embedding_generator.py`)
    - [ ] Load trained CNN → encode toàn bộ windows → vector(128)
    - [ ] Batch INSERT vào `pattern_embeddings` (với label + future_return)
    - [ ] Verify HNSW index rebuilt / updated sau insert
- [ ] 4.4 **SVM Classifier** (`ml/svm_classifier.py`)
    - [ ] Input: original_embedding(128) + context_features (từ stored proc) + S/R metadata
    - [ ] Output: predicted_label (0/1/2) + confidence
    - [ ] Framework: scikit-learn
    - [ ] Train trên tập train, evaluate trên tập test
- [ ] 4.5 **Dependencies**
    - [ ] `uv add torch scikit-learn numpy pandas`

---

## Phase 5: RAC Application Layer

> **Mục tiêu:** Python wrapper gọi Stored Procedures + kết nối ML output → prediction pipeline hoàn chỉnh.

- [ ] 5.1 **RAC Retriever** (`rac/retriever.py`)
    - [ ] Wrapper gọi `find_similar_patterns()` từ Python
    - [ ] Input: query embedding (128-d), k, threshold, optional symbol filter
    - [ ] Output: danh sách neighbors (id, symbol, label, return, distance, window range)
- [ ] 5.2 **RAC Context Enricher** (`rac/context_enricher.py`)
    - [ ] Wrapper gọi `compute_rac_context()` — aggregate KNN stats
    - [ ] Wrapper gọi `compute_full_rac_context()` — hybrid HNSW + B-Tree query
    - [ ] Wrapper gọi `get_distance_to_nearest_sr()` — S/R distance
- [ ] 5.3 **RAC Classifier** (`rac/rac_classifier.py`)
    - [ ] Orchestrate: encode window → retrieve neighbors → enrich context → SVM predict
    - [ ] Lưu kết quả vào `rac_predictions` (với neighbor_ids, label_dist, confidence)
- [ ] 5.4 **Explainer** (`rac/explainer.py`)
    - [ ] Format Top-K neighbors + OHLCV data gốc → human-readable evidence
    - [ ] Dùng cho Streamlit visualization
- [ ] 5.5 **Tests**
    - [ ] Test retriever với mock embeddings (insert fake vectors → verify KNN results)
    - [ ] Test full RAC pipeline end-to-end

---

## Phase 6: Benchmarking

> **Mục tiêu:** Đo lường hiệu suất DB — đây là phần cốt lõi của báo cáo. Mọi claim phải có `EXPLAIN ANALYZE`.

- [ ] 6.1 **HNSW vs Sequential Scan** (`benchmark/hnsw_vs_seqscan.py`)
    - [ ] Tắt/bật index scan → so sánh latency
    - [ ] Đo: query latency P50/P95/P99, pages read (buffers hit/read), Recall@K vs exact
- [ ] 6.2 **Hybrid Search** (`benchmark/hybrid_search_bench.py`)
    - [ ] So sánh: HNSW only vs B-Tree filter + HNSW
    - [ ] Chạy `EXPLAIN ANALYZE` → capture execution plan
- [ ] 6.3 **In-DB vs App-side Computing** (`benchmark/indb_vs_appside.py`)
    - [ ] Phương án A: Query K raw vectors → Python tính stats
    - [ ] Phương án B: Gọi `compute_rac_context()` → DB trả kết quả tổng hợp
    - [ ] Đo: end-to-end time, bytes transferred, round-trips
- [ ] 6.4 **HNSW Parameter Sweep** (`benchmark/hnsw_param_sweep.py`)
    - [ ] Grid: m=[8,16,32,64] × ef_construction=[64,128,200,400] × ef_search=[40,100,200,400]
    - [ ] Đo: index build time, index size, query latency, Recall@10/20, RAM usage
- [ ] 6.5 **TimescaleDB Chunking** (`benchmark/chunk_size_bench.py`)
    - [ ] So sánh chunk intervals: 1 week, 1 month, 3 months
    - [ ] Đo: range query latency, compression ratio, chunk exclusion effectiveness
- [ ] 6.6 **Export kết quả** → `benchmark/results/` (CSV/JSON cho báo cáo + Streamlit)

---

## Phase 7: FastAPI REST API

> **Mục tiêu:** HTTP API thay thế CLI, phục vụ làm backend cho Streamlit dashboard.

- [ ] 7.1 **App skeleton** (`api/main.py`)
    - [ ] FastAPI app + lifespan (async connection pool via `psycopg_pool`)
    - [ ] CORS middleware
    - [ ] `api/deps.py` — shared dependency `get_db_conn`
    - [ ] `api/schemas.py` — Pydantic request/response models
- [ ] 7.2 **ETL Router** (`api/routers/etl.py`)
    - [ ] `POST /api/etl/seed` — load fixture
    - [ ] `POST /api/etl/backfill` — background task, trả job_id
    - [ ] `POST /api/etl/incremental` — background task
    - [ ] `GET /api/etl/status/{job_id}` — poll trạng thái
- [ ] 7.3 **OHLCV Router** (`api/routers/ohlcv.py`)
    - [ ] `GET /api/symbols` — danh sách symbols trong DB
    - [ ] `GET /api/ohlcv/{symbol}?start=...&end=...` — OHLCV theo time range
    - [ ] `GET /api/ohlcv/{symbol}/latest?n=30` — N phiên gần nhất
- [ ] 7.4 **RAC Router** (`api/routers/rac.py`)
    - [ ] `POST /api/rac/similar-patterns` — gọi `find_similar_patterns()`
    - [ ] `POST /api/rac/context` — gọi `compute_rac_context()`
    - [ ] `POST /api/rac/full-context` — gọi `compute_full_rac_context()` (★ Hybrid Query)
    - [ ] `GET /api/rac/predictions/{symbol}?limit=20` — lịch sử predictions
- [ ] 7.5 **Metadata Router** (`api/routers/metadata.py`)
    - [ ] `GET /api/sr-zones/{symbol}` — S/R zones active
    - [ ] `GET /api/sr-zones/{symbol}/distance?price=...` — gọi `get_distance_to_nearest_sr()`
- [ ] 7.6 **Benchmark Router** (`api/routers/benchmark.py`)
    - [ ] `POST /api/benchmark/explain` — chạy `EXPLAIN ANALYZE`
    - [ ] `GET /api/benchmark/stats` — `pg_stat_statements` summary
- [ ] 7.7 **Dependencies**: `uv add uvicorn psycopg-pool`
- [ ] 7.8 **Tests**: API integration tests với `httpx.AsyncClient`

---

## Phase 8: Streamlit Dashboard

> **Mục tiêu:** Visualization cho demo bảo vệ đề tài. Gọi FastAPI, KHÔNG kết nối DB trực tiếp.

- [ ] 8.1 **App entry** (`streamlit_app/app.py`) — multi-page layout
- [ ] 8.2 **Trang 1: OHLCV Explorer** (`pages/1_ohlcv_chart.py`)
    - [ ] Dropdown symbol + date range picker
    - [ ] Plotly Candlestick + Volume bars
    - [ ] S/R zones overlay (horizontal lines)
- [ ] 8.3 **Trang 2: Similar Patterns** (`pages/2_similar_patterns.py`) ★ Highlight
    - [ ] Chọn query window (symbol + date range)
    - [ ] Grid hiển thị K neighbors: mini candlestick charts
    - [ ] Normalized overlay chart (so sánh hình dáng cùng scale [0,1])
    - [ ] Bảng: cosine_distance, label, future_return
    - [ ] Pie chart: label distribution + confidence metric
- [ ] 8.4 **Trang 3: RAC Prediction** (`pages/3_rac_prediction.py`)
    - [ ] Hiển thị 3 blocks từ `compute_full_rac_context()`:
        - Block 1 — KNN Stats: metric cards + pie chart
        - Block 2 — S/R Context: gauge chart (sr_position_ratio)
        - Block 3 — Evidence: neighbor IDs + link sang trang 2
- [ ] 8.5 **Trang 4: Benchmark** (`pages/4_benchmark.py`)
    - [ ] Bar chart: HNSW vs SeqScan latency
    - [ ] Bar chart: In-DB vs App-side
    - [ ] Heatmap: HNSW param sweep → Recall@K
    - [ ] Query plan viewer (`EXPLAIN ANALYZE` output)
- [ ] 8.6 **Shared components**
    - [ ] `components/candlestick.py` — Plotly candlestick helper
    - [ ] `components/similarity.py` — Normalized overlay helper
- [ ] 8.7 **Dependencies**: `uv add --dev streamlit plotly httpx`

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