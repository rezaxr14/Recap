"""SQLAlchemy ORM models — authoritative durable state.

Elasticsearch is a rebuildable search index; these tables are the source of
truth for runs, memories, evidence, validation, and lifecycle audit.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class WorkspaceRow(Base):
    __tablename__ = "workspaces"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class EnvironmentRow(Base):
    __tablename__ = "environments"

    id: Mapped[str] = mapped_column(String(128), primary_key=True)
    workspace_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    environment_type: Mapped[str] = mapped_column(String(64), nullable=False)
    version: Mapped[str] = mapped_column(String(64), nullable=False)
    fingerprint: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class RunRow(Base):
    __tablename__ = "runs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    workspace_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    agent_id: Mapped[str] = mapped_column(String(128), nullable=False)
    task_id: Mapped[str] = mapped_column(String(256), nullable=False)
    task_family: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    environment_id: Mapped[str] = mapped_column(String(128), nullable=False)
    environment_fingerprint: Mapped[str] = mapped_column(String(128), nullable=False)
    toolset_fingerprint: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    model_provider: Mapped[str] = mapped_column(String(64), nullable=False, default="fake")
    model_name: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    model_config_fingerprint: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    failure_type: Mapped[str | None] = mapped_column(String(128), nullable=True)
    success_criteria: Mapped[str] = mapped_column(Text, nullable=False, default="")
    tool_calls_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    retry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    input_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    estimated_cost: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    memory_snapshot_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    config_snapshot: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    events: Mapped[list[RunEventRow]] = relationship(
        back_populates="run", cascade="all, delete-orphan", order_by="RunEventRow.sequence_number"
    )


class RunEventRow(Base):
    __tablename__ = "run_events"
    __table_args__ = (
        UniqueConstraint("run_id", "sequence_number", name="uq_run_event_sequence"),
        Index("ix_run_events_run_seq", "run_id", "sequence_number"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.id", ondelete="CASCADE"), nullable=False)
    sequence_number: Mapped[int] = mapped_column(Integer, nullable=False)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    payload_schema_version: Mapped[str] = mapped_column(String(32), nullable=False)
    tool_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    tool_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    correlation_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    payload_redacted: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    payload_hash: Mapped[str] = mapped_column(String(80), nullable=False)
    redaction_applied: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    redaction_rule_version: Mapped[str] = mapped_column(String(32), nullable=False)

    run: Mapped[RunRow] = relationship(back_populates="events")


class MemorySnapshotRow(Base):
    """Frozen set of memory versions available to one run.

    The snapshot is created when a run starts and is monotonic: retrieval may
    only see entries in this set. Live lifecycle checks can still remove a
    memory (fail closed), but newly promoted memories never appear mid-run.
    """

    __tablename__ = "memory_snapshots"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    workspace_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    reason: Mapped[str] = mapped_column(String(64), nullable=False, default="run_start")
    index_generation: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    embedding_model_id: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    entry_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    entries: Mapped[list[MemorySnapshotEntryRow]] = relationship(
        back_populates="snapshot", cascade="all, delete-orphan", lazy="selectin"
    )


class MemorySnapshotEntryRow(Base):
    __tablename__ = "memory_snapshot_entries"

    snapshot_id: Mapped[str] = mapped_column(
        ForeignKey("memory_snapshots.id", ondelete="CASCADE"), primary_key=True
    )
    memory_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    memory_version: Mapped[int] = mapped_column(Integer, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(80), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)

    snapshot: Mapped[MemorySnapshotRow] = relationship(back_populates="entries")


class MemoryRow(Base):
    __tablename__ = "memories"
    __table_args__ = (
        UniqueConstraint("workspace_id", "content_hash", name="uq_memory_workspace_content"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    workspace_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    family_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    memory_version: Mapped[int] = mapped_column(Integer, nullable=False)
    supersedes_memory_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    schema_version: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    status_reason: Mapped[str] = mapped_column(Text, nullable=False, default="")
    created_from: Mapped[str] = mapped_column(String(32), nullable=False)
    risk_level: Mapped[str] = mapped_column(String(16), nullable=False)
    access_scope: Mapped[str] = mapped_column(String(32), nullable=False, default="workspace")
    content_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(80), nullable=False)
    environment_id: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    environment_fingerprint: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    embedding_model_id: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    embedding_model_version: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    embedding_dimension: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    retrieval_document_id: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    index_generation: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    indexed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    index_status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    evidence_run_ids: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    evidence_trace_ids: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    validation_record_ids: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    stats_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    activated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_validated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )


class OutcomeRow(Base):
    __tablename__ = "outcomes"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    memory_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    memory_version: Mapped[int] = mapped_column(Integer, nullable=False)
    workspace_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    run_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    outcome: Mapped[str] = mapped_column(String(32), nullable=False)
    applied: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    selected: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    environment_id: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    environment_fingerprint: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    task_family: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    severity: Mapped[str] = mapped_column(String(32), nullable=False, default="normal")
    notes: Mapped[str] = mapped_column(Text, nullable=False, default="")
    evidence_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class RetrievalDecisionRow(Base):
    __tablename__ = "retrieval_decisions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    workspace_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    request_id: Mapped[str] = mapped_column(String(64), nullable=False)
    run_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    mode: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    query_redacted: Mapped[str] = mapped_column(Text, nullable=False)
    policy_version: Mapped[str] = mapped_column(String(32), nullable=False)
    decision_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    total_latency_ms: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )


class RepairCandidateRow(Base):
    __tablename__ = "repair_candidates"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    workspace_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    source_run_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    validation_status: Mapped[str] = mapped_column(String(32), nullable=False)
    risk_level: Mapped[str] = mapped_column(String(16), nullable=False)
    content_hash: Mapped[str] = mapped_column(String(80), nullable=False, index=True)
    environment_id: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    environment_fingerprint: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    candidate_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    promoted_memory_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class ValidationRow(Base):
    __tablename__ = "validation_results"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    candidate_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    validator_name: Mapped[str] = mapped_column(String(128), nullable=False)
    attempt: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    environment_id: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    environment_fingerprint: Mapped[str] = mapped_column(String(128), nullable=False, default="")
    candidate_hash: Mapped[str] = mapped_column(String(80), nullable=False, default="")
    result_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class CompatibilityRow(Base):
    __tablename__ = "compatibility"
    __table_args__ = (
        UniqueConstraint(
            "memory_id", "memory_version", "environment_fingerprint", name="uq_compat_key"
        ),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    memory_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    memory_version: Mapped[int] = mapped_column(Integer, nullable=False)
    workspace_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    environment_id: Mapped[str] = mapped_column(String(128), nullable=False)
    environment_fingerprint: Mapped[str] = mapped_column(String(128), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    reasons_json: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    evidence_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class RevalidationJobRow(Base):
    __tablename__ = "revalidation_jobs"
    __table_args__ = (UniqueConstraint("dedupe_key", name="uq_revalidation_dedupe"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    memory_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    memory_version: Mapped[int] = mapped_column(Integer, nullable=False)
    workspace_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    reason: Mapped[str] = mapped_column(String(32), nullable=False)
    target_environment_id: Mapped[str] = mapped_column(String(128), nullable=False)
    target_environment_fingerprint: Mapped[str] = mapped_column(
        String(128), nullable=False, default=""
    )
    validation_plan: Mapped[str] = mapped_column(String(64), nullable=False)
    priority: Mapped[int] = mapped_column(Integer, nullable=False, default=5)
    state: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=3)
    dedupe_key: Mapped[str] = mapped_column(String(200), nullable=False)
    result_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    error: Mapped[str] = mapped_column(Text, nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class AuditEventRow(Base):
    __tablename__ = "audit_events"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    workspace_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    entity_type: Mapped[str] = mapped_column(String(32), nullable=False)
    entity_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    action: Mapped[str] = mapped_column(String(64), nullable=False)
    reason: Mapped[str] = mapped_column(Text, nullable=False, default="")
    actor: Mapped[str] = mapped_column(String(64), nullable=False, default="system")
    before_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    after_json: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class OutboxRow(Base):
    """Transactional outbox for DB -> Elasticsearch synchronization.

    The dedupe key is unique only among *active* rows (pending/processing). A
    completed row must not block a later reindex of the same memory version.
    """

    __tablename__ = "index_outbox"
    __table_args__ = (
        Index(
            "uq_outbox_dedupe_active",
            "dedupe_key",
            unique=True,
            postgresql_where=text("state IN ('pending', 'processing')"),
            sqlite_where=text("state IN ('pending', 'processing')"),
        ),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    workspace_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    entity_type: Mapped[str] = mapped_column(String(32), nullable=False)
    entity_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    operation: Mapped[str] = mapped_column(String(16), nullable=False)
    dedupe_key: Mapped[str] = mapped_column(String(200), nullable=False)
    state: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=8)
    last_error: Mapped[str] = mapped_column(Text, nullable=False, default="")
    next_attempt_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
