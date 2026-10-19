"""Memory management endpoints."""

from __future__ import annotations

from typing import Any, cast

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from recap.domain.enums import MemoryStatus, OutcomeStatus, RiskLevel
from recap.domain.memories import MemoryContent
from recap.errors import NotFoundError

router = APIRouter(tags=["memories"])


class MemoryCreateRequest(BaseModel):
    workspace_id: str = Field(min_length=1, max_length=128)
    content: MemoryContent
    environment_id: str = ""
    environment_fingerprint: str = ""
    evidence_run_ids: list[str] = Field(default_factory=list)
    risk_level: str = "low"


class TransitionRequest(BaseModel):
    target_status: MemoryStatus
    reason: str = Field(min_length=1, max_length=500)


class OutcomeRequest(BaseModel):
    outcome: OutcomeStatus
    applied: bool = False
    run_id: str | None = None
    environment_id: str = ""
    environment_fingerprint: str = ""
    task_family: str = ""
    severity: str = "normal"
    notes: str = ""


@router.post("/memories", status_code=201)
def create_memory(payload: MemoryCreateRequest, request: Request) -> dict[str, Any]:
    container = request.app.state.get_container()
    with container.services() as services:
        from recap.domain.enums import CreatedFrom, MemoryStatus
        from recap.domain.memories import ExperienceMemory

        memory = ExperienceMemory(
            workspace_id=payload.workspace_id,
            content=payload.content,
            status=MemoryStatus.DRAFT,
            status_reason="created manually",
            created_from=CreatedFrom.MANUAL,
            risk_level=RiskLevel(payload.risk_level),
            environment_id=payload.environment_id,
            environment_fingerprint=payload.environment_fingerprint,
            evidence_run_ids=payload.evidence_run_ids,
            embedding_model_id=services.settings.embeddings.model_id,
            embedding_model_version=services.settings.embeddings.model_version,
            embedding_dimension=services.settings.embeddings.dimension,
            index_generation=services.settings.elasticsearch.index_generation,
        )
        created = services.memories.create_memory(memory)
        return cast(dict[str, Any], created.model_dump(mode="json"))


@router.get("/memories")
def list_memories(
    request: Request,
    workspace_id: str,
    status: MemoryStatus | None = None,
    limit: int = 50,
) -> dict[str, Any]:
    container = request.app.state.get_container()
    with container.services() as services:
        memories = services.memories.list_memories(workspace_id, status=status, limit=limit)
        return {"memories": [memory.model_dump(mode="json") for memory in memories]}


@router.get("/memories/{memory_id}")
def get_memory(memory_id: str, request: Request, workspace_id: str) -> dict[str, Any]:
    container = request.app.state.get_container()
    with container.services() as services:
        memory = services.memories.get_memory(memory_id, workspace_id)
        if memory is None:
            raise NotFoundError(f"memory {memory_id} not found")
        versions = services.memories.list_versions(memory.family_id, workspace_id)
        return {
            "memory": memory.model_dump(mode="json"),
            "versions": [item.model_dump(mode="json") for item in versions],
            "compatibility": [
                item.model_dump(mode="json")
                for item in services.compatibility.list_for_memory(memory_id)
            ],
        }


@router.post("/memories/{memory_id}/transition")
def transition(
    memory_id: str, payload: TransitionRequest, request: Request, workspace_id: str
) -> dict[str, Any]:
    container = request.app.state.get_container()
    with container.services() as services:
        memory = services.memories.get_memory(memory_id, workspace_id)
        if memory is None:
            raise NotFoundError(f"memory {memory_id} not found")
        updated = memory.transition(payload.target_status, payload.reason)
        services.memories.update_status(updated)
        services.index_sync(updated)
        from recap.domain.feedback import AuditEvent

        services.audit.append(
            AuditEvent(
                workspace_id=workspace_id,
                entity_type="memory",
                entity_id=memory_id,
                action=f"transition:{memory.status}->{payload.target_status}",
                reason=payload.reason,
            )
        )
        return cast(dict[str, Any], updated.model_dump(mode="json"))


@router.post("/memories/{memory_id}/observe-outcome")
def observe_outcome(
    memory_id: str, payload: OutcomeRequest, request: Request, workspace_id: str
) -> dict[str, Any]:
    container = request.app.state.get_container()
    with container.services() as services:
        memory = services.memories.get_memory(memory_id, workspace_id)
        if memory is None:
            raise NotFoundError(f"memory {memory_id} not found")
        record = services.memory_shift.record_outcome(
            memory,
            outcome=payload.outcome,
            applied=payload.applied,
            run_id=payload.run_id,
            environment_id=payload.environment_id,
            environment_fingerprint=payload.environment_fingerprint,
            task_family=payload.task_family,
            severity=payload.severity,
            notes=payload.notes,
        )
        return cast(dict[str, Any], record.model_dump(mode="json"))


@router.get("/memories/{memory_id}/compatibility")
def compatibility(memory_id: str, request: Request, workspace_id: str) -> dict[str, Any]:
    container = request.app.state.get_container()
    with container.services() as services:
        results = services.memory_shift.compatibility_report(memory_id, workspace_id)
        return {"compatibility": [item.model_dump(mode="json") for item in results]}
