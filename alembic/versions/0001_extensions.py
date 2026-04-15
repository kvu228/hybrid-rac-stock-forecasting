"""extensions: timescaledb + pgvector

Revision ID: 0001_extensions
Revises: 
Create Date: 2026-04-15
"""

from __future__ import annotations

from alembic import op

revision = "0001_extensions"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS timescaledb;")
    op.execute("CREATE EXTENSION IF NOT EXISTS vector;")


def downgrade() -> None:
    op.execute("DROP EXTENSION IF EXISTS vector;")
    op.execute("DROP EXTENSION IF EXISTS timescaledb;")

