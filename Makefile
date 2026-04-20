UV ?= uv
PY ?= $(UV) run python
ALEMBIC ?= $(UV) run alembic
UVICORN ?= $(UV) run uvicorn
STREAMLIT ?= $(UV) run streamlit

# Phase 8: Streamlit (calls FastAPI; default API URL overridable)
ST_APP ?= streamlit_app/app.py
ST_PORT ?= 8501
ST_HOST ?= 127.0.0.1

COMPOSE ?= docker compose
COMPOSE_ENV ?= --env-file .env

TICKERS_VN100 ?= etl/tickers_vn100.txt

# Backfill defaults (override on CLI: `make backfill-vn100 END=2026-04-19`)
START ?= 2010-01-01
END ?= $(shell $(PY) -c "import datetime as d; print(d.date.today().isoformat())")
CHUNK_DAYS ?= 365
CONCURRENCY ?= 4
RPM ?= 55

# Phase 3: generate-windows / detect-sr
WINDOWS_OUT ?= data/windows
WIN_START ?=
WIN_END ?=
WIN_DATE_FLAGS := $(if $(strip $(WIN_START)),--start $(WIN_START),) $(if $(strip $(WIN_END)),--end $(WIN_END),)
WINDOW_SIZE ?= 30
HORIZON ?= 5
STRIDE ?= 1
UP_THRESHOLD ?= 0.02
DOWN_THRESHOLD ?= -0.02
TRAIN_RATIO ?= 0.8
SR_ORDER ?= 5

# Phase 4: encoder + embeddings
ML_ENCODER_OUT ?= ml/model_store/cnn_encoder.pt
OHLCV_TSV ?= tests/fixtures/ohlcv_small.tsv
# DB-scale training defaults (VN100 / full history): retrieval-first SupCon profile
ML_ENCODER_EPOCHS ?= 60
ML_ENCODER_BATCH ?= 512
ML_LR ?= 0.0003
ML_LOSS ?= supcon
ML_SUPCON_TEMP ?= 0.2
ML_SUPCON_CE ?= 0.2
ML_ES_PATIENCE ?= 15
# Small TSV smoke training (fixture): keep fast defaults separate from DB training
ML_TSV_TRAIN_EPOCHS ?= 8
ML_DEVICE ?= cpu
ML_EMBED_BATCH ?= 512

# Phase 6: DB benchmarks (require DATABASE_URL; HNSW scripts DROP/CREATE idx_embedding_hnsw)
BENCH_K ?= 20
BENCH_N_QUERIES ?= 30
BENCH_SEED ?= 42
BENCH_EF_SEARCH ?= 100

.PHONY: help
help:
	@echo "Makefile shortcuts for common CLI workflows."
	@echo ""
	@echo "Windows note:"
	@echo "  This repo's Makefile requires GNU Make (often not installed by default)."
	@echo "  Options: install GNU Make via MSYS2/chocolatey, or run Git Bash and use: make <target>"
	@echo ""
	@echo "Common targets:"
	@echo "  make sync                 - uv sync --dev"
	@echo "  make lint                 - ruff check"
	@echo "  make typecheck            - mypy"
	@echo "  make test                 - pytest"
	@echo "  make check                - lint + typecheck + tests"
	@echo "  make db-up                - docker compose up -d --build"
	@echo "  make db-down              - docker compose down"
	@echo "  make migrate              - alembic upgrade head"
	@echo "  make seed                 - seed small fixture dataset"
	@echo "  make api                  - run FastAPI (reload)"
	@echo "  make streamlit            - run Streamlit dashboard (Phase 8; needs API + ST_PORT/ST_HOST)"
	@echo "  make etl-backfill-vn100   - backfill full VN100 list to END (defaults to today)"
	@echo "  make etl-incremental-vn100 - incremental update to END"
	@echo "  make windows-ps-backfill-vn100 - PowerShell one-liner (no line-continuation pitfalls)"
	@echo ""
	@echo "Phase 3 (DB + disk windows export):"
	@echo "  make phase3-vn100         - generate-windows then detect-sr (VN100 list)"
	@echo "  make etl-generate-windows-vn100 - export train/test npz+csv under WINDOWS_OUT"
	@echo "  make etl-generate-windows-symbol SYMBOL=VCB - same for one symbol"
	@echo "  make etl-detect-sr-vn100  - pivot S/R -> support_resistance_zones"
	@echo "  make etl-detect-sr-symbol SYMBOL=VCB"
	@echo "  make etl-purge-inactive-sr-vn100 - DELETE is_active=FALSE for VN100 list"
	@echo "  make etl-purge-inactive-sr-all - DELETE all inactive S/R rows (whole table)"
	@echo ""
	@echo "Phase 4 (encoder weights -> pattern_embeddings):"
	@echo "  make ml-train-encoder-synthetic - smoke train (synthetic OHLCV)"
	@echo "  make ml-train-encoder       - train from OHLCV_TSV (default: small fixture)"
	@echo "  make ml-train-encoder-db    - train from stock_ohlcv (VN100 list, DATABASE_URL)"
	@echo "  make ml-train-encoder-db-symbol SYMBOL=VCB - same, single symbol"
	@echo "  make ml-embed-symbol SYMBOL=VCB - embed one symbol (truncate + insert)"
	@echo "  make ml-embed-vn100         - loop all tickers in TICKERS_VN100 (sequential)"
	@echo ""
	@echo "Phase 6 (DB benchmarks — dev DB only for HNSW DROP/CREATE; see benchmark/README.md):"
	@echo "  make bench-smoke            - pytest tests/test_benchmark_smoke.py"
	@echo "  make bench-hnsw-vs-seqscan  - HNSW vs exact KNN + EXPLAIN (vars: BENCH_K, BENCH_N_QUERIES, ...)"
	@echo "  make bench-hnsw-vs-seqscan-dry - print one EXPLAIN only (no index drop)"
	@echo "  make bench-hybrid           - global KNN vs symbol-filtered KNN"
	@echo "  make bench-indb-appside   - compute_rac_context vs app-side aggregation"
	@echo "  make bench-hnsw-sweep-quick - small HNSW param grid"
	@echo "  make bench-hnsw-sweep     - full HNSW grid (slow)"
	@echo "  make bench-chunk            - Timescale chunk interval bench tables + range EXPLAIN"
	@echo "  make bench-chunk-teardown   - DROP bench_stock_ohlcv_* hypertables"
	@echo ""
	@echo "Examples:"
	@echo "  make etl-backfill-vn100 END=2026-04-19"
	@echo "  make etl-incremental-vn100 END=2026-04-19 CONCURRENCY=2 RPM=55"
	@echo "  make etl-generate-windows-vn100 WIN_START=2015-01-01 WIN_END=2025-12-31"
	@echo "  make ml-train-encoder ML_TSV_TRAIN_EPOCHS=12 OHLCV_TSV=path/to/export.tsv"
	@echo "  make ml-train-encoder-db WIN_START=2015-01-01 WIN_END=2025-12-31"
	@echo "  make ml-embed-vn100 ML_EMBED_BATCH=256"
	@echo "  make bench-hnsw-vs-seqscan BENCH_K=20 BENCH_N_QUERIES=50 BENCH_SEED=1"

.PHONY: sync
sync:
	$(UV) sync --dev

.PHONY: lint
lint:
	$(UV) run ruff check .

.PHONY: typecheck
typecheck:
	$(UV) run mypy .

.PHONY: test
test:
	$(UV) run pytest -q

.PHONY: check
check: lint typecheck test

.PHONY: db-up
db-up:
	$(COMPOSE) $(COMPOSE_ENV) up -d --build

.PHONY: db-down
db-down:
	$(COMPOSE) $(COMPOSE_ENV) down

.PHONY: migrate
migrate:
	$(ALEMBIC) upgrade head

.PHONY: seed
seed:
	$(PY) -m db.seed_small_dataset

.PHONY: api
api:
	$(UVICORN) api.main:app --reload --host 0.0.0.0 --port 8000

.PHONY: streamlit
streamlit:
	$(STREAMLIT) run $(ST_APP) --server.address $(ST_HOST) --server.port $(ST_PORT)

.PHONY: etl-backfill-vn100
etl-backfill-vn100:
	$(PY) -m etl.pipeline backfill \
		--symbols-file $(TICKERS_VN100) \
		--start $(START) \
		--end $(END) \
		--chunk-days $(CHUNK_DAYS) \
		--concurrency $(CONCURRENCY) \
		--requests-per-minute $(RPM)

.PHONY: etl-incremental-vn100
etl-incremental-vn100:
	$(PY) -m etl.pipeline incremental \
		--symbols-file $(TICKERS_VN100) \
		--end $(END) \
		--chunk-days $(CHUNK_DAYS) \
		--concurrency $(CONCURRENCY) \
		--requests-per-minute $(RPM)

# PowerShell-friendly copy/paste (useful when users accidentally paste bash continuations into PS)
.PHONY: windows-ps-backfill-vn100
windows-ps-backfill-vn100:
	@echo "Copy/paste in PowerShell:"
	@echo '$(UV) run python -m etl.pipeline backfill `'
	@echo '  --symbols-file $(TICKERS_VN100) `'
	@echo '  --start $(START) `'
	@echo '  --end $(END) `'
	@echo '  --chunk-days $(CHUNK_DAYS) `'
	@echo '  --concurrency $(CONCURRENCY) `'
	@echo '  --requests-per-minute $(RPM)'

.PHONY: etl-generate-windows-vn100
etl-generate-windows-vn100:
	$(PY) -m etl.pipeline generate-windows $(WIN_DATE_FLAGS) \
		--symbols-file $(TICKERS_VN100) \
		--output-dir $(WINDOWS_OUT) \
		--window-size $(WINDOW_SIZE) \
		--horizon $(HORIZON) \
		--stride $(STRIDE) \
		--up-threshold $(UP_THRESHOLD) \
		--down-threshold $(DOWN_THRESHOLD) \
		--train-ratio $(TRAIN_RATIO)

.PHONY: etl-generate-windows-symbol
etl-generate-windows-symbol: _require-symbol
	$(PY) -m etl.pipeline generate-windows $(WIN_DATE_FLAGS) \
		--symbols $(SYMBOL) \
		--output-dir $(WINDOWS_OUT) \
		--window-size $(WINDOW_SIZE) \
		--horizon $(HORIZON) \
		--stride $(STRIDE) \
		--up-threshold $(UP_THRESHOLD) \
		--down-threshold $(DOWN_THRESHOLD) \
		--train-ratio $(TRAIN_RATIO)

.PHONY: etl-detect-sr-vn100
etl-detect-sr-vn100:
	$(PY) -m etl.pipeline detect-sr \
		--symbols-file $(TICKERS_VN100) \
		--order $(SR_ORDER)

.PHONY: etl-detect-sr-symbol
etl-detect-sr-symbol: _require-symbol
	$(PY) -m etl.pipeline detect-sr \
		--symbols $(SYMBOL) \
		--order $(SR_ORDER)

.PHONY: etl-purge-inactive-sr-vn100
etl-purge-inactive-sr-vn100:
	$(PY) -m etl.pipeline purge-inactive-sr --symbols-file $(TICKERS_VN100)

.PHONY: etl-purge-inactive-sr-all
etl-purge-inactive-sr-all:
	$(PY) -m etl.pipeline purge-inactive-sr --all-inactive

.PHONY: phase3-vn100
phase3-vn100:
	$(MAKE) etl-generate-windows-vn100
	$(MAKE) etl-detect-sr-vn100

.PHONY: ml-train-encoder-synthetic
ml-train-encoder-synthetic:
	$(PY) -m ml.train_pipeline --synthetic --epochs 1 --batch-size 32 --out $(ML_ENCODER_OUT) --device $(ML_DEVICE)

.PHONY: ml-train-encoder
ml-train-encoder:
	$(PY) -m ml.train_pipeline \
		--ohlcv-tsv $(OHLCV_TSV) \
		--epochs $(ML_TSV_TRAIN_EPOCHS) \
		--batch-size $(ML_ENCODER_BATCH) \
		--lr $(ML_LR) \
		--loss $(ML_LOSS) \
		--supcon-temperature $(ML_SUPCON_TEMP) \
		--supcon-ce-weight $(ML_SUPCON_CE) \
		--early-stop-patience $(ML_ES_PATIENCE) \
		--out $(ML_ENCODER_OUT) \
		--device $(ML_DEVICE)

.PHONY: ml-train-encoder-db
ml-train-encoder-db:
	$(PY) -m ml.train_pipeline --from-db \
		$(WIN_DATE_FLAGS) \
		--symbols-file $(TICKERS_VN100) \
		--epochs $(ML_ENCODER_EPOCHS) \
		--batch-size $(ML_ENCODER_BATCH) \
		--lr $(ML_LR) \
		--loss $(ML_LOSS) \
		--supcon-temperature $(ML_SUPCON_TEMP) \
		--supcon-ce-weight $(ML_SUPCON_CE) \
		--early-stop-patience $(ML_ES_PATIENCE) \
		--out $(ML_ENCODER_OUT) \
		--device $(ML_DEVICE)

.PHONY: ml-train-encoder-db-symbol
ml-train-encoder-db-symbol: _require-symbol
	$(PY) -m ml.train_pipeline --from-db \
		$(WIN_DATE_FLAGS) \
		--symbols $(SYMBOL) \
		--epochs $(ML_ENCODER_EPOCHS) \
		--batch-size $(ML_ENCODER_BATCH) \
		--lr $(ML_LR) \
		--loss $(ML_LOSS) \
		--supcon-temperature $(ML_SUPCON_TEMP) \
		--supcon-ce-weight $(ML_SUPCON_CE) \
		--early-stop-patience $(ML_ES_PATIENCE) \
		--out $(ML_ENCODER_OUT) \
		--device $(ML_DEVICE)

.PHONY: ml-embed-symbol
ml-embed-symbol: _require-symbol
	$(PY) -m ml.embedding_generator \
		--symbol $(SYMBOL) \
		--truncate-symbol \
		--model $(ML_ENCODER_OUT) \
		--batch-size $(ML_EMBED_BATCH) \
		--device $(ML_DEVICE)

.PHONY: ml-embed-vn100
ml-embed-vn100:
	set -e; grep -vE '^[[:space:]]*#|^[[:space:]]*$$' "$(TICKERS_VN100)" | tr -d '\015' | while IFS= read -r sym; do \
		[ -z "$$sym" ] && continue; \
		echo "== embedding $$sym =="; \
		$(PY) -m ml.embedding_generator \
			--symbol "$$sym" \
			--truncate-symbol \
			--model $(ML_ENCODER_OUT) \
			--batch-size $(ML_EMBED_BATCH) \
			--device $(ML_DEVICE) || exit 1; \
	done

# PowerShell-friendly VN100 embedding loop (Windows)
.PHONY: windows-ps-embed-vn100
windows-ps-embed-vn100:
	@echo "Runs embedding loop in PowerShell (Windows)."
	@powershell -NoProfile -ExecutionPolicy Bypass -Command "$$ErrorActionPreference='Stop'; $$tickers = Get-Content '$(TICKERS_VN100)' | ForEach-Object { $$_.Trim() } | Where-Object { $$_.Length -gt 0 -and -not $$_.StartsWith('#') }; foreach ($$sym in $$tickers) { Write-Host ('== embedding ' + $$sym + ' =='); $(UV) run python -m ml.embedding_generator --symbol $$sym --truncate-symbol --model '$(ML_ENCODER_OUT)' --batch-size $(ML_EMBED_BATCH) --device $(ML_DEVICE) }"

.PHONY: _require-symbol
_require-symbol:
ifeq ($(strip $(SYMBOL)),)
	$(error SYMBOL is required (example: make ml-embed-symbol SYMBOL=VCB))
endif

# Phase 4.5: encoder diagnostics (P@k theo lớp + PCA 2D)
DIAG_K ?= 20
DIAG_N_QUERIES ?= 2000
DIAG_LEAKAGE_DAYS ?= 45
DIAG_PCA_PER_LABEL ?= 3000
DIAG_OUT ?= ml/diagnostics

.PHONY: ml-diag-encoder
ml-diag-encoder:
	$(PY) -m ml.encoder_diagnostics \
		--symbols-file $(TICKERS_VN100) \
		--encoder $(ML_ENCODER_OUT) \
		$(WIN_DATE_FLAGS) \
		--train-ratio $(TRAIN_RATIO) \
		--k $(DIAG_K) \
		--n-queries $(DIAG_N_QUERIES) \
		--leakage-days $(DIAG_LEAKAGE_DAYS) \
		--pca-per-label $(DIAG_PCA_PER_LABEL) \
		--device $(ML_DEVICE) \
		--out-dir $(DIAG_OUT)

.PHONY: bench-smoke
bench-smoke:
	$(UV) run pytest -q tests/test_benchmark_smoke.py

.PHONY: bench-hnsw-vs-seqscan
bench-hnsw-vs-seqscan:
	$(PY) -m benchmark.hnsw_vs_seqscan \
		--k $(BENCH_K) \
		--n-queries $(BENCH_N_QUERIES) \
		--seed $(BENCH_SEED) \
		--ef-search $(BENCH_EF_SEARCH)

.PHONY: bench-hnsw-vs-seqscan-dry
bench-hnsw-vs-seqscan-dry:
	$(PY) -m benchmark.hnsw_vs_seqscan --dry-run

.PHONY: bench-hybrid
bench-hybrid:
	$(PY) -m benchmark.hybrid_search_bench \
		--k $(BENCH_K) \
		--n-queries $(BENCH_N_QUERIES) \
		--seed $(BENCH_SEED) \
		--ef-search $(BENCH_EF_SEARCH)

.PHONY: bench-indb-appside
bench-indb-appside:
	$(PY) -m benchmark.indb_vs_appside \
		--k $(BENCH_K) \
		--n-queries $(BENCH_N_QUERIES) \
		--seed $(BENCH_SEED) \
		--ef-search $(BENCH_EF_SEARCH)

.PHONY: bench-hnsw-sweep-quick
bench-hnsw-sweep-quick:
	$(PY) -m benchmark.hnsw_param_sweep --quick --n-queries 8 --seed $(BENCH_SEED)

.PHONY: bench-hnsw-sweep
bench-hnsw-sweep:
	$(PY) -m benchmark.hnsw_param_sweep --n-queries 12 --seed $(BENCH_SEED)

.PHONY: bench-chunk
bench-chunk:
	$(PY) -m benchmark.chunk_size_bench

.PHONY: bench-chunk-teardown
bench-chunk-teardown:
	$(PY) -m benchmark.chunk_size_bench --teardown-only
