"""support_resistance_zones table (+ partial index)

Revision ID: 0005_support_resistance_zones
Revises: 0004_rac_predictions
Create Date: 2026-04-15
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

revision = "0005_support_resistance_zones"
down_revision = "0004_rac_predictions"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "support_resistance_zones",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("symbol", sa.Text(), nullable=False),
        sa.Column("zone_type", sa.String(length=10), nullable=False),
        sa.Column("price_level", sa.Float(asdecimal=False), nullable=False),
        sa.Column("strength", sa.Float(asdecimal=False), nullable=True),
        sa.Column("detected_at", sa.TIMESTAMP(timezone=True), server_default=sa.text("NOW()")),
        sa.Column("is_active", sa.Boolean(), server_default=sa.text("TRUE"), nullable=False),
    )

    op.execute(
        "CREATE INDEX idx_sr_symbol_active ON support_resistance_zones(symbol) WHERE is_active = TRUE;"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_sr_symbol_active;")
    op.drop_table("support_resistance_zones")

