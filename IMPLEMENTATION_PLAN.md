# Implementation Plan: Hybrid-RAC-Stock

- [ ] **Phase 1: Infrastructure & DB Schema** (Mục tiêu: Setup môi trường và bảng biểu)
    - [ ] 1.1 Docker Compose (PG16, TimescaleDB, pgvector)
    - [ ] 1.2 SQL Migrations (Hypertable, HNSW Index)
- [ ] **Phase 2: Data Engineering (ETL)** (Mục tiêu: Dữ liệu vào DB chuẩn)
    - [ ] 2.1 Vnstock Fetcher & Cleaner
    - [ ] 2.2 Bulk Ingestion Logic
- [ ] **Phase 3: In-DB Logic (Stored Procedures)** (Mục tiêu: Xử lý RAC tại tầng DB)
    - [ ] 3.1 `compute_full_rac_context` (Hybrid Query)
- [ ] **Phase 4: ML & Embedding Pipeline**
    - [ ] 4.1 1D-CNN Encoder (PyTorch)
    - [ ] 4.2 Embedding Generator & Indexing
- [ ] **Phase 5: Benchmarking & Reports**
    - [ ] 5.1 Latency & Recall Benchmarks