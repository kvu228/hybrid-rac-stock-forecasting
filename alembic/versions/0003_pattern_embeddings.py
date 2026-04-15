"""pattern_embeddings table (+ basic filter indexes)

Revision ID: 0003_pattern_embeddings
Revises: 0002_stock_ohlcv_hypertable
Create Date: 2026-04-15
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0003_pattern_embeddings"
down_revision = "0002_stock_ohlcv_hypertable"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
CREATE TABLE pattern_embeddings (
  id            BIGSERIAL PRIMARY KEY,
  symbol        TEXT NOT NULL,
  window_start  TIMESTAMPTZ NOT NULL,
  window_end    TIMESTAMPTZ NOT NULL,
  embedding     vector(128) NOT NULL,
  label         SMALLINT,
  future_return DOUBLE PRECISION,
  created_at    TIMESTAMPTZ DEFAULT NOW()
);
""".strip()
    )

    op.create_index("idx_embedding_symbol", "pattern_embeddings", ["symbol"])
    op.create_index("idx_embedding_label", "pattern_embeddings", ["label"])


def downgrade() -> None:
    op.drop_index("idx_embedding_label", table_name="pattern_embeddings")
    op.drop_index("idx_embedding_symbol", table_name="pattern_embeddings")
    op.drop_table("pattern_embeddings")

