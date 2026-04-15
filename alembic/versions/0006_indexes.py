"""indexes: HNSW on pattern_embeddings.embedding

Revision ID: 0006_indexes
Revises: 0005_support_resistance_zones
Create Date: 2026-04-15
"""

from __future__ import annotations

from alembic import op

revision = "0006_indexes"
down_revision = "0005_support_resistance_zones"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
CREATE INDEX idx_embedding_hnsw
  ON pattern_embeddings
  USING hnsw (embedding vector_cosine_ops)
  WITH (m = 16, ef_construction = 200);
""".strip()
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_embedding_hnsw;")

