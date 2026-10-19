"""memory chunks

Adds the Postgres source of truth for deterministic chunks. The baseline
revision creates all metadata on a fresh database, so this migration checks for
the table first and only creates what is missing.

Revision ID: 0004_memory_chunks
Revises: 0003_memory_snapshots
Create Date: 2026-10-19
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0004_memory_chunks"
down_revision = "0003_memory_snapshots"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "memory_chunks" in set(inspector.get_table_names()):
        return
    op.create_table(
        "memory_chunks",
        sa.Column("chunk_id", sa.String(length=64), primary_key=True),
        sa.Column("workspace_id", sa.String(length=64), nullable=False),
        sa.Column("memory_id", sa.String(length=64), nullable=False),
        sa.Column("memory_version", sa.Integer(), nullable=False),
        sa.Column("run_id", sa.String(length=64), nullable=True),
        sa.Column("chunk_type", sa.String(length=32), nullable=False),
        sa.Column("seq_start", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("seq_end", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("event_ids", sa.JSON(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("text_hash", sa.String(length=80), nullable=False),
        sa.Column("token_estimate", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("truncated", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("chunker_version", sa.String(length=32), nullable=False),
        sa.Column("redaction_rule_version", sa.String(length=32), nullable=False),
        sa.Column("failure_code", sa.String(length=64), nullable=False, server_default=""),
        sa.Column("tool_names", sa.JSON(), nullable=False),
        sa.Column("tool_versions", sa.JSON(), nullable=False),
        sa.Column("environment_id", sa.String(length=128), nullable=False, server_default=""),
        sa.Column(
            "environment_fingerprint", sa.String(length=128), nullable=False, server_default=""
        ),
        sa.Column("task_families", sa.JSON(), nullable=False),
        sa.Column("exact_ids", sa.JSON(), nullable=False),
        sa.Column(
            "active_for_retrieval", sa.Boolean(), nullable=False, server_default=sa.true()
        ),
        sa.Column("index_status", sa.String(length=32), nullable=False, server_default="pending"),
        sa.Column("indexed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_memory_chunks_workspace_id", "memory_chunks", ["workspace_id"])
    op.create_index("ix_memory_chunks_memory_id", "memory_chunks", ["memory_id"])
    op.create_index("ix_memory_chunks_chunk_type", "memory_chunks", ["chunk_type"])
    op.create_index("ix_memory_chunks_memory", "memory_chunks", ["memory_id", "memory_version"])
    op.create_index("ix_memory_chunks_run", "memory_chunks", ["run_id"])


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "memory_chunks" in set(inspector.get_table_names()):
        for name in (
            "ix_memory_chunks_run",
            "ix_memory_chunks_memory",
            "ix_memory_chunks_chunk_type",
            "ix_memory_chunks_memory_id",
            "ix_memory_chunks_workspace_id",
        ):
            op.drop_index(name, table_name="memory_chunks")
        op.drop_table("memory_chunks")
