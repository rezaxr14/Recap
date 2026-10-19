"""run-level memory snapshots

Adds a frozen set of memory versions per run plus its entries. The baseline
revision creates all current metadata on a fresh database, so this migration
checks for the tables first and only creates what is missing.

Revision ID: 0003_memory_snapshots
Revises: 0002_outbox_active_dedupe
Create Date: 2026-10-19
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0003_memory_snapshots"
down_revision = "0002_outbox_active_dedupe"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = set(inspector.get_table_names())

    if "memory_snapshots" not in tables:
        op.create_table(
            "memory_snapshots",
            sa.Column("id", sa.String(length=64), primary_key=True),
            sa.Column("workspace_id", sa.String(length=64), nullable=False),
            sa.Column("reason", sa.String(length=64), nullable=False),
            sa.Column("index_generation", sa.Integer(), nullable=False),
            sa.Column("embedding_model_id", sa.String(length=128), nullable=False),
            sa.Column("entry_count", sa.Integer(), nullable=False),
            sa.Column("metadata_json", sa.JSON(), nullable=False),
            sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        )
        op.create_index(
            "ix_memory_snapshots_workspace_id",
            "memory_snapshots",
            ["workspace_id"],
        )

    if "memory_snapshot_entries" not in tables:
        op.create_table(
            "memory_snapshot_entries",
            sa.Column(
                "snapshot_id",
                sa.String(length=64),
                sa.ForeignKey("memory_snapshots.id", ondelete="CASCADE"),
                primary_key=True,
            ),
            sa.Column("memory_id", sa.String(length=64), primary_key=True),
            sa.Column("memory_version", sa.Integer(), nullable=False),
            sa.Column("content_hash", sa.String(length=80), nullable=False),
            sa.Column("status", sa.String(length=32), nullable=False),
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    tables = set(inspector.get_table_names())
    if "memory_snapshot_entries" in tables:
        op.drop_table("memory_snapshot_entries")
    if "memory_snapshots" in tables:
        op.drop_index("ix_memory_snapshots_workspace_id", table_name="memory_snapshots")
        op.drop_table("memory_snapshots")
