"""memory chunk failure anchor

``seq_start`` is the first event actually present in a failure-window chunk,
which may be preceding context. The failure position is tracked separately so
the window stays traceable to the failing event.

Revision ID: 0006_chunk_anchor_seq
Revises: 0005_chunk_embedding_identity
Create Date: 2026-10-20
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0006_chunk_anchor_seq"
down_revision = "0005_chunk_embedding_identity"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "memory_chunks" not in set(inspector.get_table_names()):
        return
    columns = {column["name"] for column in inspector.get_columns("memory_chunks")}
    if "anchor_seq" not in columns:
        op.add_column(
            "memory_chunks",
            sa.Column("anchor_seq", sa.Integer(), nullable=False, server_default="0"),
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "memory_chunks" not in set(inspector.get_table_names()):
        return
    columns = {column["name"] for column in inspector.get_columns("memory_chunks")}
    if "anchor_seq" in columns:
        op.drop_column("memory_chunks", "anchor_seq")
