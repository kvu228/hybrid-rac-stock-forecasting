# PROJECT CONTEXT — Hybrid-RAC-Stock

> **Đề tài:** Hệ thống nhận diện mẫu hình và dự báo chứng khoán dựa trên kiến trúc Cơ sở dữ liệu lai (TimescaleDB & pgvector)
> **Môn học:** Cơ sở dữ liệu nâng cao (Advanced Databases) — Chương trình Thạc sĩ CNTT
> **Trọng tâm kỹ thuật:** Database Engineering, KHÔNG phải AI/Deep Learning

---

## 1. BỐI CẢNH VÀ BÀI TOÁN

### 1.1 Vấn đề cần giải quyết

Dự báo chuỗi thời gian tài chính đối mặt ba rào cản kỹ thuật ở tầng cơ sở dữ liệu:

**Index Bloat trên B-Tree:** Dữ liệu OHLCV có tính chất append-only với khóa thời gian tăng đơn điệu (monotonic). B-Tree truyền thống gây phình to chỉ mục khi bảng vượt ngưỡng bộ nhớ đệm, dẫn đến suy giảm I/O theo hàm mũ — đặc biệt nghiêm trọng khi dữ liệu đạt hàng triệu bản ghi.

**SQL không hỗ trợ tính toán khoảng cách vector:** Bài toán nhận diện mẫu hình tương đồng yêu cầu tìm kiếm láng giềng gần nhất (KNN) trong không gian 128 chiều. Các toán tử đại số tuyến tính của SQL truyền thống không hỗ trợ tự nhiên cho phép tính cosine similarity hay L2 distance trên vector.

**Vấn đề hộp đen (Black-Box):** Các mô hình deep learning đưa ra dự báo nhưng không trích xuất được nguồn gốc dữ liệu (data provenance) để kiểm chứng. Thiếu cơ chế giải thích tại sao một dự báo được đưa ra.

### 1.2 Giải pháp đề xuất

Kiến trúc **Hybrid Storage** trên PostgreSQL 16+ kết hợp:

- **TimescaleDB** → quản lý chuỗi thời gian OHLCV (Hypertable, Chunking, Column Compression)
- **pgvector** → lưu trữ và tìm kiếm vector embedding 128 chiều (HNSW Index)
- **RAC (Retrieval-Augmented Classification)** → truy xuất mẫu hình lịch sử tương đồng làm bằng chứng định lượng cho dự báo, giải quyết vấn đề hộp đen

Tất cả cùng chạy trên **một instance PostgreSQL duy nhất** — không đồng bộ giữa nhiều hệ thống.

---

## 2. KIẾN TRÚC HỆ THỐNG

### 2.1 Tổng quan kiến trúc

```
┌─────────────────────────────────────────────────────────────────┐
│                     APPLICATION LAYER (Python)                  │
│   Vnstock ETL │ 1D-CNN Encoder │ SVM Classifier │ RAC Engine   │
└──────┬────────────────┬──────────────────┬──────────────────────┘
       │                │                  │
       ▼                ▼                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                   PostgreSQL 16+ Instance                       │
│                                                                 │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐  │
│  │   TimescaleDB    │  │    pgvector      │  │  Stored Procs │  │
│  │                  │  │                  │  │  (In-DB       │  │
│  │  • Hypertable    │  │  • vector(128)   │  │   Computing)  │  │
│  │  • Auto-Chunk    │  │  • HNSW Index    │  │               │  │
│  │  • Compression   │  │  • Cosine Sim    │  │  • K-NN Stats │  │
│  │  • Cont. Agg.    │  │  • Hybrid Search │  │  • Context    │  │
│  │                  │  │                  │  │    Enrichment │  │
│  └──────────────────┘  └──────────────────┘  └──────────────┘  │
│                                                                 │
│  Indexes: B-Tree (symbol, time) + HNSW (embedding vector)       │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow (Pipeline)

```
Raw OHLCV (Vnstock/VCI API)
    │
    ▼
[1] ETL & Ingestion ──→ TimescaleDB Hypertable (stock_ohlcv)
    │                     • Chunk interval: tunable (1 week default)
    │                     • Compression policy: segments > N days
    │
    ▼
[2] Preprocessing
    │  • Alignment theo trading calendar
    │  • Forward-fill giá, xử lý volume riêng
    │  • Z-score normalization per window
    │  • Relative return transformation
    │
    ▼
[2.5] Metadata Extraction (S/R Zones)
    │  • Thuật toán dò tìm các điểm đảo chiều (Pivot Points/K-Means trên giá)
    │  • Lưu trữ mức giá Hỗ trợ (Support) và Kháng cự (Resistance)
    │  • Tính toán trọng số (strength) dựa trên số lần giá test vùng này
    │
    ▼
[3] Sliding Window (30 sessions)
    │  • Tensor đầu vào: [30 × 5] (OHLCV channels)
    │  • Stride: 1 (overlap tối đa)
    │
    ▼
[4] 1D-CNN Encoder ──→ Embedding vector(128)
    │                     • Lưu vào bảng pattern_embeddings
    │                     • Indexed bằng HNSW
    │
    ▼
[5] RAC Pipeline (tại query time)
    │  a. Input: embedding của window hiện tại
    │  b. pgvector KNN search (cosine similarity, K=10..50)
    │  c. Stored Procedure tính thống kê K-láng giềng
    │  d. Context Enrichment → feature vector mở rộng
    │  e. SVM Classifier → dự báo + bằng chứng lịch sử
    │
    ▼
[6] Output: Prediction + Top-K similar historical patterns (explainable)
```

---

## 3. DATABASE SCHEMA

### 3.1 Bảng chính

```sql
-- Extension setup
CREATE EXTENSION IF NOT EXISTS timescaledb;
CREATE EXTENSION IF NOT EXISTS vector;

-- ============================================================
-- BẢNG 1: Dữ liệu OHLCV (TimescaleDB Hypertable)
-- ============================================================
CREATE TABLE stock_ohlcv (
    time        TIMESTAMPTZ     NOT NULL,
    symbol      TEXT            NOT NULL,
    open        DOUBLE PRECISION NOT NULL,
    high        DOUBLE PRECISION NOT NULL,
    low         DOUBLE PRECISION NOT NULL,
    close       DOUBLE PRECISION NOT NULL,
    volume      BIGINT          NOT NULL
);

-- Chuyển thành hypertable, chunk theo time
SELECT create_hypertable('stock_ohlcv', 'time',
    chunk_time_interval => INTERVAL '1 month'
);

-- Index composite cho truy vấn theo symbol + time range
CREATE INDEX idx_ohlcv_symbol_time
    ON stock_ohlcv (symbol, time DESC);

-- Compression policy (nén chunk cũ hơn 3 tháng)
ALTER TABLE stock_ohlcv SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol',
    timescaledb.compress_orderby = 'time DESC'
);
SELECT add_compression_policy('stock_ohlcv', INTERVAL '90 days');

-- ============================================================
-- BẢNG 2: Pattern Embeddings (pgvector)
-- ============================================================
CREATE TABLE pattern_embeddings (
    id              BIGSERIAL       PRIMARY KEY,
    symbol          TEXT            NOT NULL,
    window_start    TIMESTAMPTZ     NOT NULL,
    window_end      TIMESTAMPTZ     NOT NULL,
    embedding       vector(128)     NOT NULL,
    label           SMALLINT,           -- 0: down, 1: neutral, 2: up (T+5)
    future_return   DOUBLE PRECISION,   -- actual return at T+5
    created_at      TIMESTAMPTZ     DEFAULT NOW()
);

-- HNSW Index cho cosine similarity search
CREATE INDEX idx_embedding_hnsw
    ON pattern_embeddings
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 200);

-- B-Tree index cho filtered search (Hybrid Search)
CREATE INDEX idx_embedding_symbol
    ON pattern_embeddings (symbol);

CREATE INDEX idx_embedding_label
    ON pattern_embeddings (label);

-- ============================================================
-- BẢNG 3: RAC Results (lưu kết quả dự báo + bằng chứng)
-- ============================================================
CREATE TABLE rac_predictions (
    id              BIGSERIAL       PRIMARY KEY,
    query_embedding_id  BIGINT      REFERENCES pattern_embeddings(id),
    predicted_label     SMALLINT    NOT NULL,
    confidence_score    DOUBLE PRECISION,
    k_neighbors         INTEGER     NOT NULL,
    avg_neighbor_dist   DOUBLE PRECISION,
    neighbor_label_dist JSONB,          -- {"0": 3, "1": 2, "2": 5}
    neighbor_ids        BIGINT[],       -- array of matched embedding IDs
    predicted_at        TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================
-- BẢNG 4: Metadata & Vùng giá nhạy cảm (S/R Zones)
-- Mục đích: Làm giàu dữ liệu ngữ cảnh (Context Enrichment) cho RAC
-- ============================================================
CREATE TABLE support_resistance_zones (
    id              SERIAL          PRIMARY KEY,
    symbol          TEXT            NOT NULL,
    zone_type       VARCHAR(10)     NOT NULL, -- 'SUPPORT' hoặc 'RESISTANCE'
    price_level     DOUBLE PRECISION NOT NULL,
    strength        DOUBLE PRECISION,         -- Mật độ điểm đảo chiều / Trọng số
    detected_at     TIMESTAMPTZ     DEFAULT NOW(),
    is_active       BOOLEAN         DEFAULT TRUE
);

-- B-Tree Index tối ưu truy vấn khoảng cách giá theo từng mã
CREATE INDEX idx_sr_symbol_active 
    ON support_resistance_zones(symbol) 
    WHERE is_active = TRUE;
```

### 3.2 Stored Procedures (In-DB Computing)

```sql
-- ============================================================
-- Stored Procedure: Tìm K-láng giềng + tính thống kê
-- Mục đích: Giảm tải network I/O bằng cách tính toán
--           trực tiếp trên DB thay vì trả raw vectors về app
-- ============================================================
CREATE OR REPLACE FUNCTION find_similar_patterns(
    query_vec       vector(128),
    k_neighbors     INTEGER DEFAULT 20,
    similarity_threshold DOUBLE PRECISION DEFAULT 0.7,
    filter_symbol   TEXT DEFAULT NULL
)
RETURNS TABLE (
    neighbor_id         BIGINT,
    neighbor_symbol     TEXT,
    neighbor_label      SMALLINT,
    neighbor_return     DOUBLE PRECISION,
    cosine_distance     DOUBLE PRECISION,
    window_start        TIMESTAMPTZ,
    window_end          TIMESTAMPTZ
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        pe.id,
        pe.symbol,
        pe.label,
        pe.future_return,
        (pe.embedding <=> query_vec) AS cos_dist,
        pe.window_start,
        pe.window_end
    FROM pattern_embeddings pe
    WHERE (filter_symbol IS NULL OR pe.symbol = filter_symbol)
      AND (pe.embedding <=> query_vec) < (1.0 - similarity_threshold)
    ORDER BY pe.embedding <=> query_vec
    LIMIT k_neighbors;
END;
$$ LANGUAGE plpgsql;

-- ============================================================
-- Stored Procedure: Context Enrichment
-- Tính toán thống kê tổng hợp từ K-láng giềng
-- ============================================================
CREATE OR REPLACE FUNCTION compute_rac_context(
    query_vec       vector(128),
    k_neighbors     INTEGER DEFAULT 20
)
RETURNS TABLE (
    total_neighbors     INTEGER,
    avg_cosine_dist     DOUBLE PRECISION,
    label_distribution  JSONB,
    avg_future_return   DOUBLE PRECISION,
    stddev_future_return DOUBLE PRECISION,
    dominant_label      SMALLINT,
    confidence          DOUBLE PRECISION
) AS $$
BEGIN
    RETURN QUERY
    WITH neighbors AS (
        SELECT
            pe.label,
            pe.future_return,
            (pe.embedding <=> query_vec) AS cos_dist
        FROM pattern_embeddings pe
        ORDER BY pe.embedding <=> query_vec
        LIMIT k_neighbors
    ),
    label_counts AS (
        SELECT
            label,
            COUNT(*)::INTEGER AS cnt
        FROM neighbors
        GROUP BY label
    )
    SELECT
        COUNT(*)::INTEGER,
        AVG(n.cos_dist),
        jsonb_object_agg(lc.label::TEXT, lc.cnt),
        AVG(n.future_return),
        STDDEV(n.future_return),
        (SELECT lc2.label FROM label_counts lc2
         ORDER BY lc2.cnt DESC LIMIT 1),
        (SELECT MAX(lc3.cnt)::DOUBLE PRECISION / k_neighbors
         FROM label_counts lc3)
    FROM neighbors n
    CROSS JOIN (SELECT 1) dummy
    LEFT JOIN label_counts lc ON TRUE;
END;
$$ LANGUAGE plpgsql;

-- ============================================================
-- Stored Procedure: Tính khoảng cách đến S/R Zone gần nhất
-- ============================================================
CREATE OR REPLACE FUNCTION get_distance_to_nearest_sr(
    p_symbol        TEXT,
    p_current_price DOUBLE PRECISION
)
RETURNS TABLE (
    dist_to_support     DOUBLE PRECISION,
    dist_to_resistance  DOUBLE PRECISION
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        (SELECT MIN(ABS(p_current_price - price_level)) 
         FROM support_resistance_zones 
         WHERE symbol = p_symbol AND zone_type = 'SUPPORT' AND is_active = TRUE),
        (SELECT MIN(ABS(price_level - p_current_price)) 
         FROM support_resistance_zones 
         WHERE symbol = p_symbol AND zone_type = 'RESISTANCE' AND is_active = TRUE);
END;
$$ LANGUAGE plpgsql;

-- ============================================================
-- Stored Procedure: HYBRID RAC CONTEXT (★ Core của kiến trúc Hybrid)
-- 
-- Kết hợp trong MỘT lần round-trip duy nhất:
--   1. HNSW vector search (pgvector)  → "Hình dáng tương đồng"
--   2. B-Tree metadata lookup (S/R)   → "Ngữ cảnh vị trí giá"
--   3. Thống kê tổng hợp K-láng giềng → "Bằng chứng định lượng"
--
-- Đây là minh chứng trực tiếp cho Hybrid Query:
--   HNSW Index Scan + B-Tree Index Scan trong cùng execution plan
-- ============================================================
CREATE OR REPLACE FUNCTION compute_full_rac_context(
    query_vec       vector(128),
    p_symbol        TEXT,
    p_current_price DOUBLE PRECISION,
    k_neighbors     INTEGER DEFAULT 20
)
RETURNS TABLE (
    -- Block 1: Thống kê KNN (từ HNSW search)
    total_neighbors         INTEGER,
    avg_cosine_dist         DOUBLE PRECISION,
    label_distribution      JSONB,
    avg_future_return       DOUBLE PRECISION,
    stddev_future_return    DOUBLE PRECISION,
    dominant_label          SMALLINT,
    knn_confidence          DOUBLE PRECISION,
    -- Block 2: Metadata S/R (từ B-Tree lookup)
    dist_to_support         DOUBLE PRECISION,
    dist_to_resistance      DOUBLE PRECISION,
    sr_position_ratio       DOUBLE PRECISION,   -- vị trí tương đối trong kênh S/R
    -- Block 3: Danh sách neighbor IDs (để truy xuất bằng chứng)
    neighbor_ids            BIGINT[]
) AS $$
BEGIN
    RETURN QUERY
    WITH 
    -- ═══ PHASE 1: HNSW Index Scan trên pgvector ═══
    knn AS (
        SELECT
            pe.id,
            pe.label,
            pe.future_return,
            (pe.embedding <=> query_vec) AS cos_dist
        FROM pattern_embeddings pe
        ORDER BY pe.embedding <=> query_vec
        LIMIT k_neighbors
    ),
    knn_stats AS (
        SELECT
            COUNT(*)::INTEGER                       AS n_total,
            AVG(cos_dist)                           AS avg_dist,
            AVG(future_return)                      AS avg_ret,
            STDDEV(future_return)                    AS std_ret,
            ARRAY_AGG(id ORDER BY cos_dist)          AS ids
        FROM knn
    ),
    knn_labels AS (
        SELECT
            label,
            COUNT(*)::INTEGER AS cnt
        FROM knn
        GROUP BY label
    ),
    knn_dominant AS (
        SELECT
            label AS dom_label,
            cnt::DOUBLE PRECISION / (SELECT n_total FROM knn_stats) AS conf
        FROM knn_labels
        ORDER BY cnt DESC
        LIMIT 1
    ),

    -- ═══ PHASE 2: B-Tree Index Scan trên S/R Zones ═══
    sr_data AS (
        SELECT
            (SELECT MIN(ABS(p_current_price - price_level))
             FROM support_resistance_zones
             WHERE symbol = p_symbol 
               AND zone_type = 'SUPPORT' 
               AND is_active = TRUE
            ) AS d_support,
            (SELECT MIN(ABS(price_level - p_current_price))
             FROM support_resistance_zones
             WHERE symbol = p_symbol 
               AND zone_type = 'RESISTANCE' 
               AND is_active = TRUE
            ) AS d_resistance
    )

    -- ═══ PHASE 3: Tổng hợp kết quả ═══
    SELECT
        ks.n_total,
        ks.avg_dist,
        (SELECT jsonb_object_agg(kl.label::TEXT, kl.cnt) FROM knn_labels kl),
        ks.avg_ret,
        ks.std_ret,
        kd.dom_label,
        kd.conf,
        sr.d_support,
        sr.d_resistance,
        -- Tỷ lệ vị trí: 0.0 = sát support, 1.0 = sát resistance
        CASE 
            WHEN sr.d_support IS NOT NULL AND sr.d_resistance IS NOT NULL 
                 AND (sr.d_support + sr.d_resistance) > 0
            THEN sr.d_support / (sr.d_support + sr.d_resistance)
            ELSE NULL
        END,
        ks.ids
    FROM knn_stats ks
    CROSS JOIN knn_dominant kd
    CROSS JOIN sr_data sr;
END;
$$ LANGUAGE plpgsql;
```

---

## 4. CẤU TRÚC THƯ MỤC DỰ KIẾN

```
hybrid-rac-stock/
│
├── PROJECT_CONTEXT.md          ← File này
├── README.md
├── pyproject.toml              ← Quản lý dependency bằng uv (PEP 621)
├── uv.lock                     ← Lockfile do uv tạo
├── .env.example                ← DB connection string, API keys
│
├── config/
│   ├── db_config.py            ← PostgreSQL connection, pool settings
│   ├── model_config.py         ← CNN hyperparams, SVM params
│   └── tuning_config.py        ← HNSW params (m, ef_construction, ef_search)
│
├── alembic.ini                  ← Alembic config (DB URL từ env)
├── alembic/
│   ├── env.py                    ← Alembic runtime (online migrations)
│   ├── script.py.mako
│   └── versions/
│       ├── 0001_extensions.py            ← timescaledb, pgvector
│       ├── 0002_stock_ohlcv_hypertable.py← table + create_hypertable + compression policy
│       ├── 0003_pattern_embeddings.py    ← embeddings table
│       ├── 0004_rac_predictions.py       ← predictions + evidence
│       ├── 0005_support_resistance_zones.py
│       ├── 0006_indexes.py               ← B-Tree indexes + HNSW index
│       └── 0007_stored_procedures.py     ← find_similar_patterns / compute_* functions
│
├── db/
│   ├── seeds/
│   │   └── seed_sample_data.py
│   └── tuning/
│       ├── chunk_interval_benchmark.sql
│       ├── hnsw_param_sweep.sql        ← Benchmark m, ef_construction
│       └── compression_analysis.sql
│
├── etl/
│   ├── vnstock_fetcher.py      ← Thu thập OHLCV từ VCI qua Vnstock
│   ├── data_cleaner.py         ← Align trading calendar, forward-fill
│   ├── feature_engineer.py     ← Z-score, relative return, sliding window
│   └── ingestion.py            ← Bulk COPY vào TimescaleDB
│
├── ml/
│   ├── cnn_encoder.py          ← 1D-CNN: [30×5] → vector(128)
│   ├── svm_classifier.py       ← SVM phân loại 3 lớp (down/neutral/up)
│   ├── train_pipeline.py       ← Training orchestration
│   └── model_store/            ← Saved model weights (.pt, .pkl)
│
├── rac/
│   ├── retriever.py            ← Gọi pgvector KNN search
│   ├── context_enricher.py     ← Gọi Stored Proc compute_rac_context()
│   ├── rac_classifier.py       ← RAC pipeline: retrieve → enrich → classify
│   └── explainer.py            ← Format bằng chứng lịch sử cho output
│
├── api/                         ← FastAPI REST API (Demo layer)
│   ├── __init__.py
│   ├── main.py              ← App factory, lifespan (DB pool startup/shutdown)
│   ├── deps.py              ← Shared dependencies (get_db_conn)
│   ├── routers/
│   │   ├── etl.py           ← POST /api/etl/* (trigger ETL qua background task)
│   │   ├── ohlcv.py         ← GET /api/symbols, /api/ohlcv/{symbol}
│   │   ├── rac.py           ← POST /api/rac/* (gọi stored procedures)
│   │   └── metadata.py      ← GET /api/sr-zones/{symbol}
│   └── schemas.py           ← Pydantic request/response models
│
├── streamlit_app/               ← Streamlit Dashboard (Visualization)
│   ├── app.py               ← Entry point (multi-page)
│   ├── pages/
│   │   ├── 1_ohlcv_chart.py     ← Candlestick + S/R overlay
│   │   ├── 2_similar_patterns.py ← So sánh K-neighbor patterns
│   │   ├── 3_rac_prediction.py   ← Full RAC context + prediction
│   │   └── 4_benchmark.py        ← HNSW vs SeqScan, query plan viewer
│   └── components/
│       ├── candlestick.py   ← Plotly candlestick helper
│       └── similarity.py    ← Normalized overlay chart helper
│
├── benchmark/
│   ├── hnsw_vs_seqscan.py      ← So sánh HNSW vs Sequential Scan
│   ├── hybrid_search_bench.py  ← Benchmark Hybrid Search (B-Tree + HNSW)
│   ├── indb_vs_appside.py      ← So sánh In-DB Computing vs App-side
│   ├── chunk_size_bench.py     ← Benchmark chunk interval sizing
│   └── results/                ← CSV/JSON benchmark outputs
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_embedding_visualization.ipynb  ← t-SNE/UMAP plot of embeddings
│   ├── 03_hnsw_tuning.ipynb
│   └── 04_rac_demo.ipynb                ← Demo end-to-end RAC pipeline
│
├── report/
│   ├── CuoiKy_CSDLNC_Hybrid-RAC-Stock.docx
│   ├── figures/
│   └── references.bib
│
└── tests/
    ├── test_etl.py
    ├── test_db_schema.py
    ├── test_hnsw_recall.py     ← Verify recall@K at different ef_search
    └── test_rac_pipeline.py
```

### 4.0 Migration strategy (Alembic)

- Repo sử dụng **Alembic** để quản lý versioning migrations (upgrade/downgrade).
- Ưu tiên dùng Alembic operations cho **DDL chuẩn** (create table/index/constraints).
- Với các phần **đặc thù TimescaleDB/pgvector** (extensions, hypertable, compression policy, HNSW index, stored procedures), migration sẽ dùng `op.execute(...)` để chạy SQL trực tiếp nhưng vẫn nằm trong revision của Alembic (giữ được audit trail + rollback theo `downgrade()`).

### 4.1 Quản lý dependency (uv)

- Repo sử dụng **uv** để quản lý dependency qua `pyproject.toml` và `uv.lock` (không dùng `requirements.txt` làm nguồn chuẩn).

Các lệnh thường dùng:

```bash
# Cài dependency từ lockfile / pyproject
uv sync

# Thêm dependency runtime (tự cập nhật pyproject.toml và uv.lock)
uv add <package>

# Thêm dependency dev
uv add --dev <package>
```

---

## 5. THAM SỐ QUAN TRỌNG CẦN TUNING

### 5.1 TimescaleDB


| Tham số               | Mô tả                           | Giá trị khởi đầu | Ghi chú                                                                                                        |
| --------------------- | ------------------------------- | ---------------- | -------------------------------------------------------------------------------------------------------------- |
| `chunk_time_interval` | Kích thước chunk theo thời gian | `1 month`        | VN100 ~100 mã × ~22 phiên/tháng ≈ 2,200 rows/chunk. Khuyến nghị TimescaleDB: chunk nên chiếm ~25% RAM khả dụng |
| `compress_after`      | Nén chunk sau N ngày            | `90 days`        | Giảm storage 90%+ cho dữ liệu lịch sử                                                                          |
| `compress_segmentby`  | Cột phân đoạn compression       | `symbol`         | Cho phép decompress theo từng mã khi query                                                                     |


### 5.2 pgvector HNSW


| Tham số                | Mô tả                                       | Giá trị khởi đầu | Phạm vi thử nghiệm         |
| ---------------------- | ------------------------------------------- | ---------------- | -------------------------- |
| `m`                    | Số liên kết hai chiều tối đa mỗi node       | `16`             | 8, 16, 32, 64              |
| `ef_construction`      | Kích thước danh sách ứng viên khi xây index | `200`            | 64, 128, 200, 400          |
| `ef_search`            | Kích thước danh sách ứng viên khi tìm kiếm  | `100`            | 40, 100, 200, 400          |
| `maintenance_work_mem` | RAM cho HNSW index build                    | `512MB`          | Phải đủ chứa toàn bộ graph |


**Trade-off cốt lõi:**

- `m` tăng → Recall tăng, RAM tăng, build time tăng
- `ef_construction` tăng → Chất lượng graph tốt hơn, build chậm hơn
- `ef_search` tăng → Recall tăng, query latency tăng (nhưng sublinear)

### 5.3 PostgreSQL General

```ini
# postgresql.conf — các tham số liên quan đến workload này
shared_buffers = '2GB'              # 25% RAM cho shared buffer
effective_cache_size = '6GB'        # 75% RAM
work_mem = '256MB'                  # Cho sort/hash operations
maintenance_work_mem = '512MB'      # Cho HNSW index build
random_page_cost = 1.1              # Nếu dùng SSD
max_parallel_workers_per_gather = 4
```

---

## 6. CÁC KỊCH BẢN THỰC NGHIỆM (BENCHMARK)

### 6.1 Kịch bản 1: HNSW vs Sequential Scan

**Mục tiêu:** Chứng minh HNSW tăng tốc tìm kiếm vector so với quét tuần tự.

```sql
-- Sequential Scan (tắt index)
SET enable_indexscan = off;
EXPLAIN ANALYZE
SELECT id, embedding <=> $query_vec AS distance
FROM pattern_embeddings
ORDER BY embedding <=> $query_vec
LIMIT 20;

-- HNSW Index Scan (bật index)
SET enable_indexscan = on;
SET hnsw.ef_search = 100;
EXPLAIN ANALYZE
SELECT id, embedding <=> $query_vec AS distance
FROM pattern_embeddings
ORDER BY embedding <=> $query_vec
LIMIT 20;
```

**Metrics cần đo:** Query latency (ms), Recall@K (so với exact search), Số pages đọc (buffers hit/read từ EXPLAIN ANALYZE).

### 6.2 Kịch bản 2: Hybrid Search (B-Tree + HNSW)

**Mục tiêu:** So sánh hiệu suất khi kết hợp filter (WHERE symbol = ...) với vector search.

```sql
-- Chỉ HNSW (không filter)
SELECT * FROM pattern_embeddings
ORDER BY embedding <=> $query_vec
LIMIT 20;

-- Hybrid: B-Tree filter trước, HNSW sau
SELECT * FROM pattern_embeddings
WHERE symbol = 'VCB'
ORDER BY embedding <=> $query_vec
LIMIT 20;
```

### 6.3 Kịch bản 3: In-DB Computing vs App-side Computing

**Mục tiêu:** So sánh chi phí mạng khi tính thống kê K-láng giềng.

**Phương án A (App-side):** Query trả về K raw vectors → Python tính thống kê.
**Phương án B (In-DB):** Gọi `compute_rac_context()` → DB trả về kết quả tổng hợp.

**Metrics:** Tổng thời gian end-to-end, dữ liệu truyền qua mạng (bytes), số round-trips.

### 6.4 Kịch bản 4: HNSW Parameter Sweep

**Mục tiêu:** Tìm bộ tham số tối ưu cho dataset cụ thể.

```
Tham số cần sweep:
  m:                [8, 16, 32, 64]
  ef_construction:  [64, 128, 200, 400]
  ef_search:        [40, 100, 200, 400]

Metrics mỗi tổ hợp:
  • Index build time (s)
  • Index size on disk (MB)
  • Query latency P50/P95/P99 (ms)
  • Recall@10, Recall@20 vs exact search
  • RAM usage (pg_stat_activity + OS level)
```

---

## 7. DỮ LIỆU THỰC NGHIỆM

### 7.1 Nguồn dữ liệu

- **Thư viện:** Vnstock (Python), source: VCI (Vietcap Securities)
- **Phạm vi:** Rổ VN100 (~100 mã vốn hóa lớn nhất)
- **Tần suất:** Daily OHLCV (End-of-Day)
- **Khoảng thời gian:** 2010 – nay (~15 năm), bao phủ đầy đủ các chu kỳ thị trường: sideway (2010–2012, 2018–2019), uptrend (2013–2017, 2020–2021, 2024–), downtrend (2022). Độ dài chuỗi quan trắc này đảm bảo mô hình tiếp xúc với đa dạng regime, giảm thiểu bias do chỉ train trên một pha thị trường duy nhất.
- **Phân chia:** 80% train (giai đoạn đầu) / 20% test (giai đoạn cuối) — phân chia theo thời gian, KHÔNG random

### 7.2 Quy mô ước tính

```
100 mã × 15 năm × ~250 phiên/năm = ~375,000 bản ghi OHLCV
(Lưu ý: không phải tất cả 100 mã đều có dữ liệu từ 2010 — các mã IPO sau
sẽ có chuỗi ngắn hơn. Ước tính thực tế: ~300,000–350,000 bản ghi OHLCV)

Sliding window stride 1: ~350,000 - 30 = ~349,970 embeddings
Mỗi embedding: 128 × 4 bytes (float32) = 512 bytes
Tổng vector data: ~349,970 × 512 bytes ≈ 171 MB (raw, chưa tính HNSW overhead)
HNSW index overhead: ~2-4x raw data (m=16) → ~340-680 MB
```

Ở quy mô ~350K vectors, HNSW bắt đầu thể hiện rõ ưu thế so với Sequential Scan (latency chênh lệch 10-50x), TimescaleDB chunking có ý nghĩa hơn khi số chunk đủ lớn để chunk exclusion giảm đáng kể I/O, và HNSW index bắt đầu tạo áp lực lên `maintenance_work_mem` — tạo điều kiện benchmark trade-off RAM/recall có thực chất.

### 7.3 Thuộc tính dữ liệu


| Cột      | Kiểu               | Ý nghĩa                                                      |
| -------- | ------------------ | ------------------------------------------------------------ |
| `time`   | `TIMESTAMPTZ`      | Timestamp phiên giao dịch — partition key cho hypertable     |
| `symbol` | `TEXT`             | Mã cổ phiếu (VCB, FPT, VNM...) — segment key cho compression |
| `open`   | `DOUBLE PRECISION` | Giá mở cửa                                                   |
| `high`   | `DOUBLE PRECISION` | Giá cao nhất phiên                                           |
| `low`    | `DOUBLE PRECISION` | Giá thấp nhất phiên                                          |
| `close`  | `DOUBLE PRECISION` | Giá đóng cửa — cơ sở tính return                             |
| `volume` | `BIGINT`           | Khối lượng khớp lệnh                                         |


---

## 8. ML PIPELINE (TÓM TẮT — CHỈ PHỤC VỤ CONTEXT CHO DB)

> **Lưu ý:** Đề tài tập trung vào database engineering. Phần ML chỉ mô tả đủ để hiểu data flow.

### 8.1 CNN Encoder

- **Kiến trúc:** 1D-CNN
- **Input:** Tensor [batch_size × 30 × 5] (30 phiên × 5 kênh OHLCV)
- **Output:** vector(128) — embedding lưu vào pgvector
- **Vai trò DB:** Embedding được INSERT vào `pattern_embeddings` và indexed bằng HNSW

### 8.2 SVM Classifier

- **Input:** Feature vector = [original_embedding(128)] + [context_features_from_KNN(N)] + [Metadata: dist_to_support, dist_to_resistance]
- **Context features** từ Stored Procedure: avg_distance, label_distribution, avg_return, stddev_return...
- **Output:** Label dự báo (0/1/2) + confidence score
- **Vai trò DB:** Context features được tính IN-DB, giảm round-trip

### 8.3 Nhãn dự báo


| Label | Ý nghĩa              | Cách tính                            |
| ----- | -------------------- | ------------------------------------ |
| 0     | Giảm (Down)          | Return T+5 < -threshold              |
| 1     | Trung tính (Neutral) | -threshold ≤ Return T+5 ≤ +threshold |
| 2     | Tăng (Up)            | Return T+5 > +threshold              |


---

## 9. PHƯƠNG PHÁP RAC (RETRIEVAL-AUGMENTED CLASSIFICATION)

### 9.1 Tổng quan

RAC lấy cảm hứng từ khung RAG (Lewis et al., 2020) nhưng áp dụng cho bài toán **phân loại** thay vì sinh văn bản. Pipeline:

```
Query Embedding → [pgvector KNN] → K Similar Patterns
                                         │
                                         ▼
                              [Stored Proc: Thống kê]
                                         │
                                         ▼
                              Context-Enriched Feature
                                         │
                                         ▼
                              [SVM Classification]
                                         │
                                         ▼
                        Prediction + Historical Evidence
```

### 9.2 Giá trị của RAC từ góc nhìn DB

**Tính minh bạch:** Mỗi dự báo kèm theo Top-K mẫu hình lịch sử tương đồng nhất — người dùng có thể tự kiểm chứng bằng cách xem lại đồ thị giá của các mẫu này.

**In-DB Computing:** Thống kê K-láng giềng (phân bố label, trung bình return, độ lệch chuẩn) được tính bằng Stored Procedure ngay trên DB, chỉ trả về kết quả tổng hợp thay vì raw vectors → giảm network I/O đáng kể.

**Hybrid Query trong một execution plan:** Stored procedure `compute_full_rac_context()` là minh chứng cốt lõi cho kiến trúc "lai" — trong cùng một lần gọi, PostgreSQL thực hiện đồng thời HNSW Index Scan (trên pgvector) để tìm mẫu hình tương đồng và B-Tree Index Scan (trên bảng S/R Zones) để tính khoảng cách đến vùng hỗ trợ/kháng cự. App-layer chỉ cần một round-trip duy nhất để nhận về toàn bộ context: thống kê KNN, metadata S/R, tỷ lệ vị trí giá (`sr_position_ratio`), và danh sách neighbor IDs.

**Làm giàu ngữ cảnh (Context Enrichment):** Sự kết hợp giữa tìm kiếm phi cấu trúc (Embedding HNSW) và tính toán khoảng cách siêu dữ liệu có cấu trúc (Metadata S/R Zones) ngay tại tầng CSDL cung cấp cho mô hình SVM góc nhìn toàn diện: "Mẫu hình này trong quá khứ diễn biến ra sao?" VÀ "Mẫu hình hiện tại đang đứng ở đâu so với các mốc cản tâm lý?".

---

## 10. CÁC VẤN ĐỀ KỸ THUẬT CẦN GIẢI QUYẾT (TỪ BÁO CÁO GIỮA KỲ)

### 10.1 Chi phí Re-indexing HNSW

Khi có dữ liệu mới (daily ingestion), HNSW index cần cập nhật. pgvector hỗ trợ incremental insert (không cần rebuild toàn bộ), nhưng hiệu suất có thể giảm dần khi graph phình to. Chiến lược:

- `REINDEX CONCURRENTLY` khi recall giảm dưới ngưỡng
- Partitioned approach: separate HNSW index theo khoảng thời gian
- Monitor recall định kỳ so với exact search

### 10.2 Market Regime Shift

Khi thị trường chuyển trạng thái (bull → bear → sideways), distribution của embeddings thay đổi. Embedding từ giai đoạn bull market có thể không tương đồng với patterns trong bear market. Chiến lược:

- Time-decay weighting: ưu tiên láng giềng gần hơn về thời gian
- Sliding retraining window cho CNN encoder
- Monitor avg_cosine_distance — nếu tăng đột biến → regime shift detected

### 10.3 Bottleneck bộ nhớ

HNSW index cần nằm hoàn toàn trong RAM để đạt hiệu suất tối ưu. Với `m=16` và ~350K vectors 128-chiều, index size ước tính 340–680 MB — vẫn khả thi trên máy 8GB RAM nhưng đã bắt đầu tạo áp lực lên shared_buffers. Khi scale lên 1M+ vectors:

- Giảm `m` (trade-off recall)
- Dùng `halfvec(128)` (half-precision float16) → giảm 50% memory
- Partitioning: tách index theo thời gian hoặc sector

---

## 11. TÀI LIỆU THAM KHẢO CHÍNH

### Papers

1. **Malkov, Y. A. & Yashunin, D. A. (2018).** "Efficient and Robust Approximate Nearest Neighbor Search Using Hierarchical Navigable Small World Graphs." *IEEE TPAMI*, 42(4), 824–836.
2. **Lewis, P. et al. (2020).** "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." *NeurIPS 2020*, 9459–9474.
3. **Yeo, W. J. et al. (2023).** "A Comprehensive Review on Financial Explainable AI." *arXiv:2309.11960*.

### Technical Documentation

1. **PostgreSQL Global Development Group.** PostgreSQL 16 Documentation — B-Tree Indexes, Index Types. *postgresql.org/docs/16/*
2. **Timescale, Inc.** TimescaleDB Documentation — Hypertables, Compression, Continuous Aggregates. *docs.timescale.com*
3. **pgvector contributors.** pgvector — Open-source vector similarity search for Postgres. *github.com/pgvector/pgvector*

---

## 12. QUY ƯỚC CODING

### 12.1 Database

- Tên bảng: `snake_case`, số ít (`stock_ohlcv`, không phải `stocks_ohlcvs`)
- Tên cột: `snake_case`
- Stored Procedures: `snake_case`, prefix mô tả hành động (`find_`, `compute_`, `get_`)
- Migration files: đánh số tuần tự `001_`, `002_`...
- Mọi query phải có `EXPLAIN ANALYZE` trong giai đoạn benchmark

### 12.2 Python

- Python 3.10+
- DB connection: `psycopg2` hoặc `asyncpg`
- ORM: KHÔNG dùng ORM cho benchmark (raw SQL để control query plan)
- ML framework: PyTorch (CNN), scikit-learn (SVM)
- Data: pandas cho preprocessing, numpy cho vector operations
- Config: `.env` + python-dotenv

### 12.3 Git

- Branch: `main` (stable), `dev` (working), `feature/`*, `benchmark/*`
- Commit message: `[module] action: description` (ví dụ: `[db] add: HNSW parameter sweep benchmark`)
- Không commit model weights > 100MB → dùng `.gitignore`

---

## 13. NGUYÊN TẮC QUAN TRỌNG KHI IMPLEMENT

> **Nguyên tắc #1:** Mọi quyết định thiết kế phải xuất phát từ góc nhìn **database engineering**. Khi viết code hoặc báo cáo, luôn hỏi: "Database đang giải quyết gì ở đây? I/O pattern ra sao? Index nào đang được sử dụng? Query plan trông như thế nào?"

> **Nguyên tắc #2:** KHÔNG sa đà vào giải thích AI/Deep Learning. CNN chỉ là công cụ tạo embedding — điều quan trọng là embedding đó được **lưu trữ, đánh index, và truy vấn** như thế nào trong PostgreSQL.

> **Nguyên tắc #3:** Mọi claim về hiệu suất phải có `EXPLAIN ANALYZE` kèm theo. Không nói "nhanh hơn" mà không có con số cụ thể (latency ms, pages read, buffer hit ratio).

> **Nguyên tắc #4:** Benchmark phải fair — cùng hardware, cùng dataset, warm cache vs cold cache phải ghi rõ. Dùng `pg_stat_statements` để monitor.

---

## 14. DEMO LAYER — FastAPI REST API

### 14.1 Mục đích

Cung cấp HTTP API để trigger ETL, truy vấn dữ liệu OHLCV, và gọi các Stored Procedures (RAC pipeline) — thay thế việc chạy CLI thủ công. API đóng vai trò **backend cho Streamlit dashboard** và cũng phục vụ demo khi bảo vệ đề tài.

### 14.2 Kiến trúc tổng quan

```
Streamlit (UI) ──HTTP──→ FastAPI ──psycopg──→ PostgreSQL
                              │
                              ├── ETL Router      → trigger backfill/incremental
                              ├── OHLCV Router    → đọc dữ liệu chuỗi thời gian
                              ├── RAC Router      → gọi stored procedures
                              └── Metadata Router → S/R zones
```

**Nguyên tắc:** Streamlit KHÔNG kết nối DB trực tiếp — mọi truy vấn đi qua FastAPI để giữ đúng kiến trúc layered.

### 14.3 Danh sách Endpoints

#### Nhóm ETL — Trigger pipeline qua API

| Method | Endpoint | Mô tả | Ghi chú |
|--------|----------|--------|---------|
| `POST` | `/api/etl/seed` | Load fixture dataset nhỏ | Đồng bộ (nhanh) |
| `POST` | `/api/etl/backfill` | Backfill OHLCV theo symbols + date range | **Background task** (chạy lâu) |
| `POST` | `/api/etl/incremental` | Cập nhật dữ liệu mới nhất | **Background task** |
| `GET` | `/api/etl/status/{job_id}` | Kiểm tra trạng thái job đang chạy | Poll từ Streamlit |

**Request body mẫu (backfill):**

```json
{
  "symbols": ["VCB", "FPT", "VNM"],
  "start": "2024-01-01",
  "end": "2024-12-31",
  "chunk_days": 365,
  "concurrency": 2
}
```

**Response mẫu:**

```json
{
  "job_id": "a1b2c3d4-...",
  "status": "started",
  "message": "Backfill job queued for 3 symbols"
}
```

> **Lưu ý:** ETL có thể chạy hàng phút → dùng `BackgroundTasks` của FastAPI, client poll `/api/etl/status/{job_id}` để theo dõi tiến trình.

#### Nhóm Data — Đọc OHLCV

| Method | Endpoint | Mô tả | Query trên DB |
|--------|----------|--------|---------------|
| `GET` | `/api/symbols` | Danh sách symbols có trong DB | `SELECT DISTINCT symbol FROM stock_ohlcv` |
| `GET` | `/api/ohlcv/{symbol}?start=...&end=...` | OHLCV theo symbol + time range | B-Tree index scan trên `(symbol, time)` |
| `GET` | `/api/ohlcv/{symbol}/latest?n=30` | N phiên gần nhất | Phục vụ sliding window preview |

#### Nhóm RAC — Gọi Stored Procedures (★ Core)

| Method | Endpoint | Stored Procedure | Vai trò DB |
|--------|----------|------------------|------------|
| `POST` | `/api/rac/similar-patterns` | `find_similar_patterns()` | HNSW Index Scan |
| `POST` | `/api/rac/context` | `compute_rac_context()` | HNSW + In-DB aggregation |
| `POST` | `/api/rac/full-context` | `compute_full_rac_context()` | **Hybrid Query**: HNSW + B-Tree trong 1 execution plan |
| `GET` | `/api/rac/predictions/{symbol}?limit=20` | `SELECT FROM rac_predictions` | Lịch sử predictions đã lưu |

**Request body mẫu (similar-patterns):**

```json
{
  "query_embedding": [0.12, -0.34, 0.56, ...],
  "k": 20,
  "similarity_threshold": 0.7,
  "filter_symbol": "VCB"
}
```

**Response mẫu:**

```json
{
  "neighbors": [
    {
      "id": 12345,
      "symbol": "VCB",
      "label": 2,
      "future_return": 0.034,
      "cosine_distance": 0.08,
      "window_start": "2021-06-01T00:00:00+07:00",
      "window_end": "2021-07-14T00:00:00+07:00"
    }
  ],
  "query_time_ms": 2.3
}
```

#### Nhóm Metadata — S/R Zones

| Method | Endpoint | Mô tả |
|--------|----------|--------|
| `GET` | `/api/sr-zones/{symbol}` | Các vùng Support/Resistance đang active |
| `GET` | `/api/sr-zones/{symbol}/distance?price=85000` | Gọi `get_distance_to_nearest_sr()` |

#### Nhóm Benchmark — Hỗ trợ báo cáo

| Method | Endpoint | Mô tả |
|--------|----------|--------|
| `POST` | `/api/benchmark/explain` | Chạy `EXPLAIN ANALYZE` cho query bất kỳ, trả về query plan |
| `GET` | `/api/benchmark/stats` | Tóm tắt từ `pg_stat_statements` |

### 14.4 Thiết kế kỹ thuật

**DB Connection Pool (lifespan):**

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
import psycopg_pool

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.pool = psycopg_pool.AsyncConnectionPool(
        conninfo=DATABASE_URL, min_size=2, max_size=10
    )
    yield
    await app.state.pool.close()

app = FastAPI(lifespan=lifespan)
```

**Dependency Injection:**

```python
from fastapi import Depends, Request

async def get_db_conn(request: Request):
    async with request.app.state.pool.connection() as conn:
        yield conn
```

**ETL Background Task pattern:**

```python
from fastapi import BackgroundTasks
from uuid import uuid4

# In-memory job store (đủ cho demo; production dùng Redis/DB)
etl_jobs: dict[str, dict] = {}

@router.post("/api/etl/backfill")
async def backfill(req: BackfillRequest, bg: BackgroundTasks):
    job_id = str(uuid4())
    etl_jobs[job_id] = {"status": "running", "progress": 0}
    bg.add_task(run_backfill, job_id, req)
    return {"job_id": job_id, "status": "started"}

@router.get("/api/etl/status/{job_id}")
async def etl_status(job_id: str):
    return etl_jobs.get(job_id, {"status": "not_found"})
```

### 14.5 Dependencies cần thêm

```bash
uv add uvicorn psycopg-pool
# psycopg[binary] đã có sẵn; psycopg-pool cho async connection pool
# fastapi đã có sẵn trong pyproject.toml
```

### 14.6 Chạy API server

```bash
uv run uvicorn api.main:app --reload --port 8000
# Swagger UI: http://localhost:8000/docs
```

---

## 15. DEMO LAYER — Streamlit Dashboard

### 15.1 Mục đích

Visualization layer cho demo bảo vệ đề tài — trực quan hóa dữ liệu OHLCV, so sánh mẫu hình tương đồng, và hiển thị kết quả RAC pipeline. Streamlit giao tiếp với PostgreSQL **thông qua FastAPI**, không kết nối DB trực tiếp.

### 15.2 Các trang chính

#### Trang 1: OHLCV Explorer (`1_ohlcv_chart.py`)

- **Input:** Dropdown chọn symbol (từ `GET /api/symbols`), date range picker
- **Chart chính:** Plotly Candlestick + Volume bar phía dưới
- **Overlay:** S/R zones hiển thị dưới dạng horizontal lines (từ `GET /api/sr-zones/{symbol}`)
- **Data source:** `GET /api/ohlcv/{symbol}?start=...&end=...`

#### Trang 2: Similar Patterns (`2_similar_patterns.py`) ★ Highlight của đề tài

- **Input:** User chọn symbol + khoảng thời gian (30 phiên) trên chart, hoặc chọn embedding ID có sẵn
- **API call:** `POST /api/rac/similar-patterns`
- **Hiển thị:**

```
┌─────────────────────────────────────────────────┐
│  Query Window (VCB, 2024-03-01 → 2024-04-11)   │
│  [========= Candlestick Chart =========]        │
├────────────────┬────────────────┬───────────────┤
│  Neighbor #1   │  Neighbor #2   │  Neighbor #3  │
│  VCB 2021-06   │  FPT 2023-01   │  VNM 2020-11  │
│  dist: 0.08    │  dist: 0.12    │  dist: 0.15   │
│  label: UP ↑   │  label: UP ↑   │  label: DOWN ↓│
│  [mini chart]  │  [mini chart]  │  [mini chart] │
├────────────────┴────────────────┴───────────────┤
│  Label Distribution         │  Confidence: 70%  │
│  ██████████ UP: 65%                             │
│  ████ DOWN: 20%                                 │
│  ███ NEUTRAL: 15%                               │
└─────────────────────────────────────────────────┘
```

- **Mini charts:** Mỗi neighbor hiển thị normalized price overlay (cùng scale 0–1) để so sánh hình dáng trực quan
- **Bảng chi tiết:** neighbor_id, symbol, cosine_distance, label, future_return, window_start/end

#### Trang 3: RAC Prediction (`3_rac_prediction.py`)

- **API call:** `POST /api/rac/full-context` (★ Hybrid Query)
- **Hiển thị 3 blocks** mapping trực tiếp với output của `compute_full_rac_context()`:

| Block | Nội dung | Visualization |
|-------|----------|---------------|
| **Block 1 — KNN Stats** | total_neighbors, avg_cosine_dist, label_distribution, dominant_label | Pie chart (label dist) + metric cards |
| **Block 2 — S/R Context** | dist_to_support, dist_to_resistance, sr_position_ratio | Gauge chart (0.0 = sát support ↔ 1.0 = sát resistance) |
| **Block 3 — Evidence** | neighbor_ids, link sang trang Similar Patterns | Bảng IDs + nút "Xem chi tiết" |

- **Ý nghĩa DB:** Trang này minh chứng Hybrid Query — một lần gọi API trả về kết quả từ cả HNSW Index Scan và B-Tree Index Scan.

#### Trang 4: Benchmark Dashboard (`4_benchmark.py`)

- **Mục đích:** Trực quan hóa kết quả benchmark cho báo cáo
- **Charts:**
  - Bar chart: HNSW vs SeqScan latency (ms)
  - Bar chart: In-DB vs App-side computing (end-to-end time + bytes transferred)
  - Heatmap: HNSW parameter sweep (m × ef_construction → Recall@K)
  - Tree/text viewer: Raw `EXPLAIN ANALYZE` output từ `POST /api/benchmark/explain`
- **Data source:** Kết quả benchmark lưu trong `benchmark/results/` hoặc query trực tiếp qua API

### 15.3 Components tái sử dụng

```python
# components/candlestick.py
def plot_candlestick(df, title, sr_zones=None):
    """Plotly candlestick + volume + optional S/R horizontal lines."""
    ...

# components/similarity.py
def plot_pattern_overlay(query_df, neighbor_dfs):
    """Normalized price overlay: nhiều patterns cùng scale [0,1]
    để so sánh hình dáng bất kể mức giá tuyệt đối."""
    ...
```

### 15.4 Dependencies cần thêm

```bash
uv add --dev streamlit plotly httpx
# httpx: HTTP client để Streamlit gọi FastAPI
# plotly: Candlestick, pie chart, heatmap
# streamlit: Dashboard framework
```

### 15.5 Chạy Streamlit

```bash
# Yêu cầu: FastAPI đang chạy ở port 8000
uv run streamlit run streamlit_app/app.py --server.port 8501
# Dashboard: http://localhost:8501
```

### 15.6 Thứ tự khởi động (Development)

```bash
# Terminal 1: Database
docker compose --env-file .env up -d --build
alembic upgrade head

# Terminal 2: API server
uv run uvicorn api.main:app --reload --port 8000

# Terminal 3: Streamlit dashboard
uv run streamlit run streamlit_app/app.py
```

