"""Environment fingerprint endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel

from recap.domain.environments import EnvironmentFingerprint

router = APIRouter(tags=["environments"])


class FingerprintRequest(BaseModel):
    workspace_id: str
    environment: EnvironmentFingerprint


@router.post("/environments/fingerprints")
def register_fingerprint(payload: FingerprintRequest, request: Request) -> dict[str, Any]:
    container = request.app.state.get_container()
    with container.services() as services:
        services.environments.upsert(payload.workspace_id, payload.environment)
        return {
            "environment_id": payload.environment.environment_id,
            "fingerprint": payload.environment.fingerprint,
            "canonical": payload.environment.canonical_payload(),
        }
