"""rac_predictions table

Revision ID: 0004_rac_predictions
Revises: 0003_pattern_embeddings
Create Date: 2026-04-15
"""

from __future__ import annotations

from alembic import op

revision = "0004_rac_predictions"
down_revision = "0003_pattern_embeddings"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
CREATE TABLE rac_predictions (
  id                 BIGSERIAL PRIMARY KEY,
  query_embedding_id BIGINT REFERENCES pattern_embeddings(id),
  predicted_label    SMALLINT NOT NULL,
  confidence_score   DOUBLE PRECISION,
  k_neighbors        INTEGER NOT NULL,
  avg_neighbor_dist  DOUBLE PRECISION,
  neighbor_label_dist JSONB,
  neighbor_ids       BIGINT[],
  predicted_at       TIMESTAMPTZ DEFAULT NOW()
);
""".strip()
    )


def downgrade() -> None:
    op.drop_table("rac_predictions")

