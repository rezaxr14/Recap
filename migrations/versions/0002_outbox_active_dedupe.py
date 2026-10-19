"""outbox dedupe is unique only for active rows

The original schema made ``dedupe_key`` globally unique, which permanently
blocked re-indexing a memory after its first outbox row completed. A partial
unique index scoped to pending/processing rows preserves deduplication while
allowing a later reindex.

This migration is written defensively: the baseline revision creates the schema
from current metadata, so on a fresh database the target state may already
exist. It inspects the live schema and only performs the missing step.

Revision ID: 0002_outbox_active_dedupe
Revises: 0001_initial
Create Date: 2026-10-19
"""

from __future__ import annotations

from alembic import op
from sqlalchemy import inspect

revision = "0002_outbox_active_dedupe"
down_revision = "0001_initial"
branch_labels = None
depends_on = None

TABLE = "index_outbox"
OLD_CONSTRAINT = "uq_outbox_dedupe"
NEW_INDEX = "uq_outbox_dedupe_active"


def upgrade() -> None:
    inspector = inspect(op.get_bind())
    constraints = {item["name"] for item in inspector.get_unique_constraints(TABLE)}
    indexes = {item["name"] for item in inspector.get_indexes(TABLE)}
    if OLD_CONSTRAINT in constraints:
        op.drop_constraint(OLD_CONSTRAINT, TABLE, type_="unique")
    if NEW_INDEX not in indexes:
        op.create_index(
            NEW_INDEX,
            TABLE,
            ["dedupe_key"],
            unique=True,
            postgresql_where="state IN ('pending', 'processing')",
            sqlite_where="state IN ('pending', 'processing')",
        )


def downgrade() -> None:
    inspector = inspect(op.get_bind())
    indexes = {item["name"] for item in inspector.get_indexes(TABLE)}
    constraints = {item["name"] for item in inspector.get_unique_constraints(TABLE)}
    if NEW_INDEX in indexes:
        op.drop_index(NEW_INDEX, table_name=TABLE)
    if OLD_CONSTRAINT not in constraints:
        op.create_unique_constraint(OLD_CONSTRAINT, TABLE, ["dedupe_key"])
