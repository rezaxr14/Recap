"""Outcome evidence, compatibility, and revalidation job models."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from recap.domain.enums import (
    CompatibilityStatus,
    OutcomeStatus,
    RevalidationReason,
    RevalidationState,
)
from recap.domain.runs import new_id, utc_now


class OutcomeEvidence(BaseModel):
    id: str = Field(default_factory=lambda: new_id("out"))
    memory_id: str
    memory_version: int = 1
    run_id: str | None = None
    workspace_id: str
    outcome: OutcomeStatus
    applied: bool = False
    selected: bool = True
    environment_id: str = ""
    environment_fingerprint: str = ""
    task_family: str = ""
    severity: str = "normal"
    notes: str = ""
    evidence: dict[str, Any] = Field(default_factory=dict)
    recorded_at: datetime = Field(default_factory=utc_now)


class CompatibilityResult(BaseModel):
    memory_id: str
    memory_version: int = 1
    environment_id: str
    environment_fingerprint: str = ""
    status: CompatibilityStatus
    reasons: list[str] = Field(default_factory=list)
    evidence: dict[str, Any] = Field(default_factory=dict)


class RevalidationJob(BaseModel):
    id: str = Field(default_factory=lambda: new_id("job"))
    memory_id: str
    memory_version: int
    workspace_id: str
    reason: RevalidationReason
    target_environment_id: str
    target_environment_fingerprint: str = ""
    validation_plan: str = "mock_sandbox"
    priority: int = Field(default=5, ge=1, le=10)
    state: RevalidationState = RevalidationState.QUEUED
    attempts: int = 0
    max_attempts: int = 3
    dedupe_key: str = ""
    result: dict[str, Any] = Field(default_factory=dict)
    error: str = ""
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class AuditEvent(BaseModel):
    id: str = Field(default_factory=lambda: new_id("aud"))
    workspace_id: str
    entity_type: str
    entity_id: str
    action: str
    reason: str = ""
    actor: str = "system"
    before: dict[str, Any] = Field(default_factory=dict)
    after: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utc_now)
