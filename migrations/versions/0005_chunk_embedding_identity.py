"""memory chunk embedding identity

Records which embedding fingerprint/generation produced a chunk vector so that a
model or prefix change is detectable without guessing, and so reconcile can skip
or redo embedding per chunk.

Revision ID: 0005_chunk_embedding_identity
Revises: 0004_memory_chunks
Create Date: 2026-10-20
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0005_chunk_embedding_identity"
down_revision = "0004_memory_chunks"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "memory_chunks" not in set(inspector.get_table_names()):
        return
    columns = {column["name"] for column in inspector.get_columns("memory_chunks")}
    if "embedding_fingerprint" not in columns:
        op.add_column(
            "memory_chunks",
            sa.Column(
                "embedding_fingerprint",
                sa.String(length=128),
                nullable=False,
                server_default="",
            ),
        )
    if "indexed_generation" not in columns:
        op.add_column(
            "memory_chunks",
            sa.Column(
                "indexed_generation",
                sa.Integer(),
                nullable=False,
                server_default="0",
            ),
        )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "memory_chunks" not in set(inspector.get_table_names()):
        return
    columns = {column["name"] for column in inspector.get_columns("memory_chunks")}
    if "indexed_generation" in columns:
        op.drop_column("memory_chunks", "indexed_generation")
    if "embedding_fingerprint" in columns:
        op.drop_column("memory_chunks", "embedding_fingerprint")
