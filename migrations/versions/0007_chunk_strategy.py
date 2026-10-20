"""memory chunk strategy label

Lets the event-aware and naive_fixed ablations coexist in one chunk index so
they can be compared on the same tasks with the same embedder.

Revision ID: 0007_chunk_strategy
Revises: 0006_chunk_anchor_seq
Create Date: 2026-10-20
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0007_chunk_strategy"
down_revision = "0006_chunk_anchor_seq"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "memory_chunks" not in set(inspector.get_table_names()):
        return
    columns = {column["name"] for column in inspector.get_columns("memory_chunks")}
    if "chunk_strategy" not in columns:
        op.add_column(
            "memory_chunks",
            sa.Column(
                "chunk_strategy",
                sa.String(length=32),
                nullable=False,
                server_default="event_aware",
            ),
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "memory_chunks" not in set(inspector.get_table_names()):
        return
    columns = {column["name"] for column in inspector.get_columns("memory_chunks")}
    if "chunk_strategy" in columns:
        op.drop_column("memory_chunks", "chunk_strategy")
