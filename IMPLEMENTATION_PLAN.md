# Implementation Plan: Hybrid-RAC-Stock

- [x] **Phase 1: Infrastructure & DB Schema** (Mục tiêu: Setup môi trường và bảng biểu)
    - [x] 1.1 Docker Compose (PG16, TimescaleDB, pgvector)
    - [x] 1.2 Alembic Migrations (Extensions, Hypertable, HNSW Index, Functions)
- [ ] **Phase 2: Data Engineering (ETL)** (Mục tiêu: Dữ liệu vào DB chuẩn)
    - [ ] 2.1 Vnstock Fetcher & Cleaner (OHLCV)
    - [ ] 2.2 Trading Calendar Alignment + Data Validation
    - [ ] 2.3 Bulk Ingestion (COPY/Batch insert) vào `stock_ohlcv`
    - [ ] 2.4 ETL Smoke Tests (row counts, NULL checks, time ordering)
    - [ ] 2.5 (Optional) Seed small dataset for CI/dev
- [ ] **Phase 3: In-DB Logic (Stored Procedures)** (Mục tiêu: Xử lý RAC tại tầng DB)
    - [ ] 3.1 `compute_full_rac_context` (Hybrid Query)
- [ ] **Phase 4: ML & Embedding Pipeline**
    - [ ] 4.1 1D-CNN Encoder (PyTorch)
    - [ ] 4.2 Embedding Generator & Indexing
- [ ] **Phase 5: Benchmarking & Reports**
    - [ ] 5.1 Latency & Recall Benchmarks