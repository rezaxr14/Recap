"""RepairCompiler endpoints."""

from __future__ import annotations

from typing import Any, cast

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from recap.domain.environments import EnvironmentFingerprint
from recap.errors import NotFoundError

router = APIRouter(tags=["repairs"])


class CompileRequest(BaseModel):
    run_id: str
    environment: EnvironmentFingerprint
    workspace_id: str


class ValidateRequest(BaseModel):
    environment: EnvironmentFingerprint
    validator_name: str = "mock_sandbox"
    attempt: int = Field(default=1, ge=1, le=10)


class PromoteRequest(BaseModel):
    reviewer: str = "system"


@router.post("/repairs/compile")
def compile_repair(payload: CompileRequest, request: Request) -> dict[str, Any]:
    container = request.app.state.get_container()
    with container.services() as services:
        candidates = services.repair_compiler.compile(
            payload.run_id, environment=payload.environment
        )
        return {"candidates": [candidate.model_dump(mode="json") for candidate in candidates]}


@router.get("/repairs/{candidate_id}")
def get_candidate(candidate_id: str, request: Request) -> dict[str, Any]:
    container = request.app.state.get_container()
    with container.services() as services:
        candidate = services.repairs.get(candidate_id)
        if candidate is None:
            raise NotFoundError(f"candidate {candidate_id} not found")
        validations = services.repairs.list_validations(candidate_id)
        return {
            "candidate": candidate.model_dump(mode="json"),
            "validations": [result.model_dump(mode="json") for result in validations],
        }


@router.post("/repairs/{candidate_id}/validate")
def validate_candidate(
    candidate_id: str, payload: ValidateRequest, request: Request
) -> dict[str, Any]:
    container = request.app.state.get_container()
    with container.services() as services:
        result = services.repair_compiler.validate(
            candidate_id,
            environment=payload.environment,
            validator_name=payload.validator_name,
            attempt=payload.attempt,
        )
        return cast(dict[str, Any], result.model_dump(mode="json"))


@router.get("/repairs/{candidate_id}/validation")
def validation_history(candidate_id: str, request: Request) -> dict[str, Any]:
    container = request.app.state.get_container()
    with container.services() as services:
        return {
            "validations": [
                result.model_dump(mode="json")
                for result in services.repairs.list_validations(candidate_id)
            ]
        }


@router.post("/repairs/{candidate_id}/promote")
def promote_candidate(
    candidate_id: str, payload: PromoteRequest, request: Request
) -> dict[str, Any]:
    container = request.app.state.get_container()
    with container.services() as services:
        candidate = services.repairs.get(candidate_id)
        if candidate is None:
            raise NotFoundError(f"candidate {candidate_id} not found")
        memory = services.promotion.promote(candidate, reviewer=payload.reviewer)
        return cast(dict[str, Any], memory.model_dump(mode="json"))
