"""Ask endpoint against the real container (Postgres + Elasticsearch)."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient
from tests.support import make_memory

from recap.api.app import create_app
from recap.domain.enums import MemoryStatus
from recap.environments.mock_api import PaginatedApiEnvironment
from recap.environments.run_recording import record_reference_run

pytestmark = pytest.mark.integration


@pytest.fixture()
def client(container):  # type: ignore[no-untyped-def]
    return TestClient(create_app(container))


class StructuredFake:
    provider_type = "fake-ask"
    model = "fake-ask-v1"

    def __init__(self, result: dict[str, Any]) -> None:
        self.result = result

    def generate_structured(self, **_: Any) -> dict[str, Any]:
        return self.result


def _seed_cursor_memory(container, workspace_id: str, *, status=MemoryStatus.ACTIVE):
    memory = make_memory(
        container,
        workspace_id=workspace_id,
        title="Cursor recovery",
        summary="Restart pagination from page one after INVALID_CURSOR.",
        failure_signature="ValidationError|INVALID_CURSOR|list_records",
        failure_code="INVALID_CURSOR",
        environment_version="2",
        status=status,
        index=status is MemoryStatus.ACTIVE,
    )
    if status is MemoryStatus.ACTIVE:
        container.es_client.indices.refresh(index=container.index_manager.write_index)
    return memory


def test_ask_returns_a_verified_memory_citation(client, container, workspace_id: str) -> None:
    memory = _seed_cursor_memory(container, workspace_id)

    response = client.post(
        "/api/v1/ask",
        json={
            "workspace_id": workspace_id,
            "question": "Which memory fixes INVALID_CURSOR?",
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "answered"
    assert body["answerer"] == "extractive"
    assert body["verification"]["valid"] >= 1
    assert body["verification"]["invalid"] == 0
    citation = body["citations"][0]
    assert citation["type"] == "memory"
    assert citation["id"] == memory.id
    assert citation["quote"]


def test_ask_cites_a_named_run(client, container, workspace_id: str) -> None:
    environment = PaginatedApiEnvironment(version="1")
    run_id, _ = record_reference_run(
        container,
        environment,
        workspace_id=workspace_id,
        task_id="ask-named-run",
    )

    response = client.post(
        "/api/v1/ask",
        json={"workspace_id": workspace_id, "question": f"Why did run {run_id} fail?"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "answered"
    cited = {(item["type"], item["id"]) for item in body["citations"]}
    assert ("run", run_id) in cited
    assert body["verification"]["invalid"] == 0


def test_ask_sees_quarantined_memories_outside_the_operational_index(
    client, container, workspace_id: str
) -> None:
    memory = _seed_cursor_memory(container, workspace_id, status=MemoryStatus.QUARANTINED)

    response = client.post(
        "/api/v1/ask",
        json={"workspace_id": workspace_id, "question": "Which memories are quarantined?"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "answered"
    assert any(item["id"] == memory.id for item in body["citations"])


def test_ask_refuses_when_no_recorded_evidence_matches(client, workspace_id: str) -> None:
    response = client.post(
        "/api/v1/ask",
        json={
            "workspace_id": workspace_id,
            "question": "What is the airspeed velocity of an unladen swallow?",
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "insufficient_evidence"
    assert body["citations"] == []


def test_llm_mode_without_a_provider_fails_closed(client, container, workspace_id: str) -> None:
    _seed_cursor_memory(container, workspace_id)

    response = client.post(
        "/api/v1/ask",
        json={
            "workspace_id": workspace_id,
            "question": "Which memory fixes INVALID_CURSOR?",
            "mode": "llm",
        },
    )
    assert response.status_code == 500, response.text
    assert response.json()["code"] == "configuration_error"


def test_llm_mode_downgrades_a_fabricated_quote(client, container, workspace_id: str) -> None:
    memory = _seed_cursor_memory(container, workspace_id)
    container.llm = StructuredFake(
        {
            "answer": "The cursor repair restarts at page one.",
            "citations": [
                {
                    "source_id": f"memory:{memory.id}",
                    "quote": "Restart from the final page.",
                }
            ],
            "confidence": 0.9,
        }
    )

    response = client.post(
        "/api/v1/ask",
        json={
            "workspace_id": workspace_id,
            "question": "Which memory fixes INVALID_CURSOR?",
            "mode": "llm",
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "insufficient_evidence"
    assert body["answerer"] == "llm"
    assert body["verification"]["invalid"] == 1
    assert body["verification"]["rejected"][0]["reason"] == "quote_not_in_source"


def test_llm_mode_accepts_a_verbatim_citation(client, container, workspace_id: str) -> None:
    memory = _seed_cursor_memory(container, workspace_id)
    container.llm = StructuredFake(
        {
            "answer": "Restart pagination from page one.",
            "citations": [
                {
                    "source_id": f"memory:{memory.id}",
                    "quote": "Restart pagination from page one after INVALID_CURSOR.",
                }
            ],
            "confidence": 0.8,
        }
    )

    response = client.post(
        "/api/v1/ask",
        json={
            "workspace_id": workspace_id,
            "question": "Which memory fixes INVALID_CURSOR?",
            "mode": "llm",
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "answered"
    assert body["answerer"] == "llm"
    assert body["provider"] == "fake-ask"
    assert body["model"] == "fake-ask-v1"
    assert body["verification"]["valid"] == 1
    assert body["verification"]["invalid"] == 0
