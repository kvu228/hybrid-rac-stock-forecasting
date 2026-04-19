UV ?= uv
PY ?= $(UV) run python
ALEMBIC ?= $(UV) run alembic
UVICORN ?= $(UV) run uvicorn

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
ML_ENCODER_EPOCHS ?= 8
ML_ENCODER_BATCH ?= 256
ML_DEVICE ?= cpu
ML_EMBED_BATCH ?= 512

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
	@echo "  make ml-embed-symbol SYMBOL=VCB - embed one symbol (truncate + insert)"
	@echo "  make ml-embed-vn100         - loop all tickers in TICKERS_VN100 (sequential)"
	@echo ""
	@echo "Examples:"
	@echo "  make etl-backfill-vn100 END=2026-04-19"
	@echo "  make etl-incremental-vn100 END=2026-04-19 CONCURRENCY=2 RPM=55"
	@echo "  make etl-generate-windows-vn100 WIN_START=2015-01-01 WIN_END=2025-12-31"
	@echo "  make ml-train-encoder ML_ENCODER_EPOCHS=12 OHLCV_TSV=path/to/export.tsv"
	@echo "  make ml-embed-vn100 ML_EMBED_BATCH=256"

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
		--epochs $(ML_ENCODER_EPOCHS) \
		--batch-size $(ML_ENCODER_BATCH) \
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

.PHONY: _require-symbol
_require-symbol:
ifeq ($(strip $(SYMBOL)),)
	$(error SYMBOL is required (example: make ml-embed-symbol SYMBOL=VCB))
endif
