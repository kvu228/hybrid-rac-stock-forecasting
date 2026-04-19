# Phase 6 — Database benchmarks

Scripts measure pgvector HNSW, hybrid symbol filters, in-DB vs app-side RAC context, HNSW parameter trade-offs, and TimescaleDB chunk intervals. Each run writes CSV + JSON under `benchmark/results/` and raw `EXPLAIN (ANALYZE, BUFFERS)` text under `benchmark/results/explain/` (gitignored).

## Makefile (repo root)

```bash
make help                    # lists bench-* targets
make bench-smoke
make bench-hnsw-vs-seqscan   # optional: BENCH_K=20 BENCH_N_QUERIES=30 BENCH_SEED=42 BENCH_EF_SEARCH=100
make bench-hybrid
make bench-indb-appside
make bench-hnsw-sweep-quick
make bench-hnsw-sweep
make bench-chunk
make bench-chunk-teardown
```

## Prerequisites

- `DATABASE_URL` pointing at the project Postgres (TimescaleDB + pgvector), schema migrated (`alembic upgrade head`).
- `pattern_embeddings` populated for vector scripts; `stock_ohlcv` populated for chunk scripts.
- Benchmarks that **drop and recreate** `idx_embedding_hnsw` must run against a **disposable/dev database**, not production.

## Commands

From repo root (`uv sync` once):

```bash
# HNSW vs exact (seq scan path after index drop); restores default index at end
uv run python -m benchmark.hnsw_vs_seqscan --k 20 --n-queries 30 --seed 42
uv run python -m benchmark.hnsw_vs_seqscan --dry-run

# Global KNN vs WHERE symbol + KNN
uv run python -m benchmark.hybrid_search_bench --k 20 --n-queries 30

# Raw K rows + Python vs compute_rac_context()
uv run python -m benchmark.indb_vs_appside --k 20 --n-queries 30

# HNSW grid (long); use --quick for a small grid
uv run python -m benchmark.hnsw_param_sweep --quick
uv run python -m benchmark.hnsw_param_sweep

# Hypertable chunk intervals (creates bench_stock_ohlcv_* tables, copies stock_ohlcv)
uv run python -m benchmark.chunk_size_bench
uv run python -m benchmark.chunk_size_bench --teardown-only
```

## Tests

```bash
uv run pytest tests/test_benchmark_smoke.py -q
```

---

## Hướng dẫn đọc `benchmark/results/`

### Cấu trúc thư mục

| Đường dẫn | Nội dung |
|-----------|----------|
| `benchmark/results/` | File **CSV** + **JSON** mỗi lần chạy script (tên có timestamp để không ghi đè). |
| `benchmark/results/explain/` | Text **EXPLAIN (ANALYZE, BUFFERS, VERBOSE)** từng truy vấn (đính kèm claim trong báo cáo). |

Thư mục thường **gitignore** (chỉ giữ `.gitignore`); bạn tự tạo lại bằng cách chạy benchmark.

### Quy ước tên file

`<tên_script>_<YYYYMMDD_HHMMSS>.csv` cùng cặp `.json` — ví dụ `hnsw_vs_seqscan_20260419_085647.csv`.

File trong `explain/` dạng `<prefix>_q<chỉ_số>_<cùng_slug_timestamp>.txt` (ví dụ `hnsw_vs_seqscan_exact_q0_20260419_085001.txt`).

### JSON (mở bằng editor / Jupyter / `jq`)

Mọi script ghi cùng khung:

- **`summary`**: metadata lần chạy — `script`, `seed`, `k`, `n_queries`, `generated_at_utc`, thêm field riêng từng benchmark (ví dụ `ef_search`, `exact_execution_ms_p50_p95_p99`, `hnsw_execution_ms_p50_p95_p99`, `recall_mean`, `server` gồm phiên bản Postgres / extension `vector` & `timescaledb`).
- **`rows`**: bảng chi tiết từng dòng đo (trùng logic với CSV).

**Cách đọc nhanh:** mở `summary` trước để lấy số tổng hợp cho báo cáo; dùng `rows` khi cần biểu đồ theo `query_idx` hoặc so sánh từng query.

### CSV

Giống cột `rows` trong JSON — mở Excel/LibreOffice hoặc `pandas.read_csv` để vẽ biểu đồ (Phase 8 / báo cáo).

### Cột theo từng benchmark

| Script | CSV / `rows` — ý nghĩa chính |
|--------|------------------------------|
| **hnsw_vs_seqscan** | `phase`: `exact_no_index` (đã DROP HNSW, quét chính xác) vs `hnsw_index`; `execution_time_ms` lấy từ dòng cuối plan (`Execution Time`); `shared_hit` / `shared_read` cộng dồn từ `Buffers:` trong EXPLAIN; `recall_at_k` chỉ có ở phase HNSW (so Top‑K id với ground truth exact). |
| **hybrid_search_bench** | `variant`: `global_knn` vs `filtered_symbol_knn`; `symbol_filter` rỗng hoặc mã đã chọn; cùng nhóm timing/buffers. |
| **indb_vs_appside** | `variant`: `app_side` (SELECT K dòng có vector + pickle size) vs `in_db` (`compute_rac_context`); `wall_ms`, `payload_bytes`, `round_trips`. |
| **hnsw_param_sweep** | Mỗi dòng = một bộ `(m, ef_construction, ef_search)` sau khi build index: `index_build_s`, `index_size_bytes`, `query_ms_p50/p95/p99`, `recall_at_10_mean`, `recall_at_20_mean`. |
| **chunk_size_bench** | Mỗi dòng = một hypertable bench (`table`, `chunk_interval_label`): `num_chunks`, `chunks_excluded` (parse từ EXPLAIN nếu có), `execution_time_ms`, `compression_ratio_estimate` (Timescale `chunk_compression_stats`, có thể `null` nếu chưa nén). |

### Thư mục `explain/*.txt`

Đây là **bằng chứng trực tiếp** cho câu “có `EXPLAIN ANALYZE`”:

- Cuối plan: **`Planning Time`** và **`Execution Time`** (ms) — latency thực tế server cho câu lệnh đó.
- Các dòng **`Buffers:`** — `shared hit` / `shared read` (I/O buffer); dùng so sánh tải đọc giữa hai phương án.
- Phần **plan tree** (Seq Scan, Index Scan, Custom Scan / Hnsw, v.v.) — giải thích *tại sao* nhanh/chậm.

Ghép với CSV: cùng `query_idx` và cùng slug timestamp trong tên file (ví dụ `..._q3_20260419_085647.txt` đi với bộ kết quả `085647`).

### Gợi ý dùng cho báo cáo / Streamlit

1. Chụp **bảng** từ `summary` (JSON) hoặc pivot từ CSV.
2. Chèn **đoạn EXPLAIN** rút gọn hoặc full từ `explain/*.txt` làm phụ lục.
3. Ghi rõ điều kiện: `seed`, `k`, phiên bản DB trong `summary.server` để kết quả có thể tái lập.
