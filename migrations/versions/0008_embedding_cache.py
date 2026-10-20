"""content-addressed embedding cache

Optional optimization only: deleting this table's rows re-embeds on the next
rebuild. Keys include the full embedding fingerprint and a namespaced input
hash, so vectors never cross models, dimensions, prefixes, or query/document
roles.

Revision ID: 0008_embedding_cache
Revises: 0007_chunk_strategy
Create Date: 2026-10-20
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0008_embedding_cache"
down_revision = "0007_chunk_strategy"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "embedding_cache" in set(inspector.get_table_names()):
        return
    op.create_table(
        "embedding_cache",
        sa.Column("embedding_fingerprint", sa.String(length=128), primary_key=True),
        sa.Column("text_hash", sa.String(length=80), primary_key=True),
        sa.Column("kind", sa.String(length=16), nullable=False, server_default="document"),
        sa.Column("dimension", sa.Integer(), nullable=False),
        sa.Column("provider", sa.String(length=64), nullable=False, server_default=""),
        sa.Column("model_id", sa.String(length=128), nullable=False, server_default=""),
        sa.Column("vector_json", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_accessed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("hit_count", sa.Integer(), nullable=False, server_default="0"),
    )
    op.create_index(
        "ix_embedding_cache_last_accessed",
        "embedding_cache",
        ["last_accessed_at"],
        unique=False,
    )


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if "embedding_cache" not in set(inspector.get_table_names()):
        return
    op.drop_index("ix_embedding_cache_last_accessed", table_name="embedding_cache")
    op.drop_table("embedding_cache")
