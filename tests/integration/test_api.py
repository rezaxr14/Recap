"""API tests against the real container (Postgres + Elasticsearch).

The HTTP layer is exercised through ``TestClient`` with a real container, so
serialization, status codes, and error mapping are all validated for real.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from tests.support import context_for, make_memory, seed_promoted_memory

from recap.api.app import create_app
from recap.domain.enums import RetrievalMode

pytestmark = pytest.mark.integration


@pytest.fixture()
def client(container):  # type: ignore[no-untyped-def]
    return TestClient(create_app(container))


def test_health_endpoint_reports_real_dependencies(client) -> None:
    response = client.get("/health/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["database"] == "ok"
    assert body["elasticsearch"]["status"] in {"green", "yellow"}

    live = client.get("/health/live")
    assert live.status_code == 200
    assert live.json() == {"status": "ok"}


def test_create_run_and_append_events(client, workspace_id: str) -> None:
    created = client.post(
        "/api/v1/runs",
        json={
            "workspace_id": workspace_id,
            "agent_id": "api-agent",
            "task_id": "api-task",
            "task_family": "api_pagination",
            "environment_id": "mock-api-v1",
            "environment_fingerprint": "sha256:api",
            "toolset_fingerprint": "sha256:api",
            "model_provider": "deterministic",
            "model_name": "api-test",
            "success_criteria": "collect all records",
        },
    )
    assert created.status_code == 201, created.text
    run_id = created.json()["id"]

    event = client.post(
        f"/api/v1/runs/{run_id}/events",
        json={
            "sequence_number": 1,
            "event_type": "run_started",
            "payload": {"task": "collect"},
        },
    )
    assert event.status_code == 201, event.text

    # Same sequence + payload is an idempotent no-op, not an error.
    replay = client.post(
        f"/api/v1/runs/{run_id}/events",
        json={
            "sequence_number": 1,
            "event_type": "run_started",
            "payload": {"task": "collect"},
        },
    )
    assert replay.status_code == 201
    assert replay.json()["event_id"] == event.json()["event_id"]

    # Conflicting payload at the same sequence is a 409.
    conflict = client.post(
        f"/api/v1/runs/{run_id}/events",
        json={
            "sequence_number": 1,
            "event_type": "run_started",
            "payload": {"task": "something else"},
        },
    )
    assert conflict.status_code == 409
    assert conflict.json()["code"] == "conflict"

    summary = client.get(f"/api/v1/runs/{run_id}")
    assert summary.status_code == 200
    assert summary.json()["event_count"] == 1

    finished = client.post(
        f"/api/v1/runs/{run_id}/finish",
        json={"status": "succeeded", "tool_calls_count": 3},
    )
    assert finished.status_code == 200
    assert finished.json()["status"] == "succeeded"

    # Events may not be appended after the run is finished.
    late = client.post(
        f"/api/v1/runs/{run_id}/events",
        json={"sequence_number": 2, "event_type": "task_observation", "payload": {}},
    )
    assert late.status_code == 409


def test_retrieval_endpoint_returns_decision(client, container, workspace_id: str) -> None:
    memory = make_memory(
        container,
        workspace_id=workspace_id,
        title="API cursor recovery",
        summary="Restart pagination from page one.",
        failure_signature="ValidationError|INVALID_CURSOR|list_records",
        failure_code="INVALID_CURSOR",
        environment_version="1",
    )
    container.es_client.indices.refresh(index=container.index_manager.write_index)
    context = context_for(workspace_id=workspace_id, environment_version="1")
    response = client.post(
        "/api/v1/retrieval/search",
        json={
            "context": context.model_dump(mode="json"),
            "mode": RetrievalMode.OUTCOME_AWARE.value,
            "final_k": 3,
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "selected"
    assert [item["memory_id"] for item in body["selected_memories"]] == [memory.id]

    decision_id = body["id"]
    fetched = client.get(
        f"/api/v1/retrieval/decisions/{decision_id}", params={"workspace_id": workspace_id}
    )
    assert fetched.status_code == 200
    assert fetched.json()["id"] == decision_id

    missing = client.get(
        "/api/v1/retrieval/decisions/dec_missing", params={"workspace_id": workspace_id}
    )
    assert missing.status_code == 404
    assert missing.json()["code"] == "not_found"


def test_repair_endpoints_run_the_real_gate(client, container, workspace_id: str) -> None:
    from recap.environments.mock_api import PaginatedApiEnvironment
    from recap.environments.run_recording import record_reference_run

    environment = PaginatedApiEnvironment(version="1")
    fingerprint = environment.fingerprint()
    run_id, _ = record_reference_run(
        container, environment, workspace_id=workspace_id, task_id="api-repair"
    )
    compiled = client.post(
        "/api/v1/repairs/compile",
        json={
            "run_id": run_id,
            "workspace_id": workspace_id,
            "environment": fingerprint.model_dump(mode="json"),
        },
    )
    assert compiled.status_code == 200, compiled.text
    candidates = compiled.json()["candidates"]
    assert len(candidates) == 1
    candidate_id = candidates[0]["id"]

    for validator in ("static", "mock_sandbox"):
        validated = client.post(
            f"/api/v1/repairs/{candidate_id}/validate",
            json={
                "environment": fingerprint.model_dump(mode="json"),
                "validator_name": validator,
            },
        )
        assert validated.status_code == 200, validated.text
        assert validated.json()["status"] == "passed"

    promoted = client.post(f"/api/v1/repairs/{candidate_id}/promote", json={})
    assert promoted.status_code == 200, promoted.text
    assert promoted.json()["status"] == "active"

    # Promoting again must fail: the candidate is no longer promotable.
    again = client.post(f"/api/v1/repairs/{candidate_id}/promote", json={})
    assert again.status_code in {409, 422}


def test_unknown_memory_returns_structured_404(client, workspace_id: str) -> None:
    response = client.get("/api/v1/memories/mem_missing", params={"workspace_id": workspace_id})
    assert response.status_code == 404
    body = response.json()
    assert body["code"] == "not_found"
    assert "message" in body


def test_validation_errors_are_structured(client) -> None:
    response = client.post("/api/v1/retrieval/search", json={"context": {}})
    assert response.status_code == 422
    assert "detail" in response.json()


def test_seed_promoted_memory_is_retrievable(client, container, workspace_id: str) -> None:
    _candidate, memory = seed_promoted_memory(
        container, workspace_id=workspace_id, environment_version="1"
    )
    container.es_client.indices.refresh(index=container.index_manager.write_index)
    context = context_for(workspace_id=workspace_id, environment_version="1")
    response = client.post(
        "/api/v1/retrieval/search",
        json={"context": context.model_dump(mode="json"), "mode": "outcome_aware"},
    )
    assert response.status_code == 200
    assert [item["memory_id"] for item in response.json()["selected_memories"]] == [memory.id]
