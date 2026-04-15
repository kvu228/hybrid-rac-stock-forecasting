"""stock_ohlcv: uniqueness on (symbol, time)

Revision ID: 0008_stock_ohlcv_unique_key
Revises: 0007_stored_procedures
Create Date: 2026-04-15
"""

from __future__ import annotations

from alembic import op

revision = "0008_stock_ohlcv_unique_key"
down_revision = "0007_stored_procedures"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Needed for idempotent ingestion (upsert/dedup) by logical key.
    op.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS ux_stock_ohlcv_symbol_time ON stock_ohlcv(symbol, time);"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ux_stock_ohlcv_symbol_time;")

