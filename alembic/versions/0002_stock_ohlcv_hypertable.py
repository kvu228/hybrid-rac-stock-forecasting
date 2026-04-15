"""stock_ohlcv hypertable + compression

Revision ID: 0002_stock_ohlcv_hypertable
Revises: 0001_extensions
Create Date: 2026-04-15
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0002_stock_ohlcv_hypertable"
down_revision = "0001_extensions"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "stock_ohlcv",
        sa.Column("time", sa.TIMESTAMP(timezone=True), nullable=False),
        sa.Column("symbol", sa.Text(), nullable=False),
        sa.Column("open", sa.Float(asdecimal=False), nullable=False),
        sa.Column("high", sa.Float(asdecimal=False), nullable=False),
        sa.Column("low", sa.Float(asdecimal=False), nullable=False),
        sa.Column("close", sa.Float(asdecimal=False), nullable=False),
        sa.Column("volume", sa.BigInteger(), nullable=False),
    )

    op.create_index(
        "idx_ohlcv_symbol_time",
        "stock_ohlcv",
        ["symbol", sa.text("time DESC")],
    )

    op.execute(
        "SELECT create_hypertable('stock_ohlcv', 'time', chunk_time_interval => INTERVAL '1 month');"
    )

    op.execute(
        """
ALTER TABLE stock_ohlcv SET (
  timescaledb.compress,
  timescaledb.compress_segmentby = 'symbol',
  timescaledb.compress_orderby = 'time DESC'
);
""".strip()
    )
    op.execute("SELECT add_compression_policy('stock_ohlcv', INTERVAL '90 days');")


def downgrade() -> None:
    op.execute("SELECT remove_compression_policy('stock_ohlcv');")
    op.drop_index("idx_ohlcv_symbol_time", table_name="stock_ohlcv")
    op.drop_table("stock_ohlcv")

