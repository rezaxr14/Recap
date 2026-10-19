"""Run and event domain models.

Events are append-only and ordered by sequence number. Payloads stored here are
already redacted at the persistence boundary.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator

from recap.domain.enums import EventType, RunStatus

EVENT_PAYLOAD_SCHEMA_VERSION = "event-v1"


def utc_now() -> datetime:
    return datetime.now(UTC)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex}"


class RunEvent(BaseModel):
    id: str = Field(default_factory=lambda: new_id("evt"))
    run_id: str
    sequence_number: int = Field(ge=1)
    event_type: EventType
    timestamp: datetime = Field(default_factory=utc_now)
    payload_schema_version: str = EVENT_PAYLOAD_SCHEMA_VERSION
    tool_name: str | None = None
    tool_version: str | None = None
    correlation_id: str | None = None
    payload_redacted: dict[str, Any] = Field(default_factory=dict)
    payload_hash: str = ""
    redaction_applied: bool = False
    redaction_rule_version: str = "redaction-v1"

    @field_validator("payload_redacted")
    @classmethod
    def _bounded_payload(cls, value: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise ValueError("payload must be an object")
        return value


class RunEventCreate(BaseModel):
    """Input event; id optional (server assigns if absent)."""

    id: str | None = None
    sequence_number: int = Field(ge=1)
    event_type: EventType
    timestamp: datetime | None = None
    tool_name: str | None = None
    tool_version: str | None = None
    correlation_id: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class AgentRun(BaseModel):
    id: str = Field(default_factory=lambda: new_id("run"))
    workspace_id: str
    agent_id: str
    task_id: str
    task_family: str
    environment_id: str
    environment_fingerprint: str
    toolset_fingerprint: str = ""
    model_provider: str = "fake"
    model_name: str = "fake-deterministic"
    model_config_fingerprint: str = ""
    status: RunStatus = RunStatus.RUNNING
    failure_type: str | None = None
    success_criteria: str = ""
    tool_calls_count: int = 0
    retry_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost: float = 0.0
    duration_ms: int | None = None
    memory_snapshot_id: str | None = None
    config_snapshot: dict[str, Any] = Field(default_factory=dict)
    started_at: datetime = Field(default_factory=utc_now)
    finished_at: datetime | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class RunCreate(BaseModel):
    workspace_id: str = Field(min_length=1, max_length=128)
    agent_id: str = Field(min_length=1, max_length=128)
    task_id: str = Field(min_length=1, max_length=256)
    task_family: str = Field(min_length=1, max_length=128)
    environment_id: str = Field(min_length=1, max_length=128)
    environment_fingerprint: str = ""
    toolset_fingerprint: str = ""
    model_provider: str = "fake"
    model_name: str = "fake-deterministic"
    model_config_fingerprint: str = ""
    success_criteria: str = ""
    config_snapshot: dict[str, Any] = Field(default_factory=dict)
    memory_snapshot_id: str | None = None


class RunFinish(BaseModel):
    status: RunStatus
    failure_type: str | None = None
    duration_ms: int | None = Field(default=None, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    estimated_cost: float = Field(default=0.0, ge=0.0)
    tool_calls_count: int | None = Field(default=None, ge=0)
    retry_count: int | None = Field(default=None, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("status")
    @classmethod
    def _terminal(cls, value: RunStatus) -> RunStatus:
        if value is RunStatus.RUNNING:
            raise ValueError("finish requires a terminal status")
        return value


class RunSummary(BaseModel):
    run: AgentRun
    event_count: int
    last_sequence: int
    outcome: dict[str, Any] = Field(default_factory=dict)
