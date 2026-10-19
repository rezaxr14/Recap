from __future__ import annotations

import pytest

from recap.domain.enums import ValidationStatus
from recap.domain.predicates import Predicate
from recap.environments.mock_api import MockApiError, PaginatedApiEnvironment


def test_v1_paginates_all_records() -> None:
    environment = PaginatedApiEnvironment(version="1")
    collected = []
    token = None
    while True:
        parameters = {"tenant": "acme"}
        if token:
            parameters["cursor"] = token
        result = environment.list_records(**parameters)
        assert result.ok
        collected.extend(result.payload["items"])
        token = result.payload.get("cursor")
        if not result.payload["has_more"]:
            break
    assert len(collected) == len(environment.records)


def test_invalid_cursor_is_rejected_with_stable_code() -> None:
    environment = PaginatedApiEnvironment(version="1")
    result = environment.list_records(tenant="acme", cursor="start")
    assert result.ok is False
    assert result.error is not None
    assert result.error["error_code"] == "INVALID_CURSOR"


def test_expired_cursor_fails_then_fresh_cursor_succeeds() -> None:
    environment = PaginatedApiEnvironment(version="1")
    first = environment.list_records(tenant="acme")
    cursor = first.payload["cursor"]
    environment.expire_cursor(cursor)
    expired = environment.list_records(tenant="acme", cursor=cursor)
    assert expired.ok is False
    fresh = environment.list_records(tenant="acme")
    assert fresh.ok is True


def test_v2_rejects_legacy_cursor_parameter() -> None:
    environment = PaginatedApiEnvironment(version="2")
    result = environment.list_records(tenant="acme", cursor="abc")
    assert result.ok is False
    assert result.error is not None
    assert result.error["error_code"] == "UNKNOWN_PARAMETER"


def test_v2_uses_page_token() -> None:
    environment = PaginatedApiEnvironment(version="2")
    first = environment.list_records(tenant="acme")
    assert first.ok
    assert "page_token" in first.payload
    second = environment.list_records(tenant="acme", page_token=first.payload["page_token"])
    assert second.ok


def test_schema_fingerprints_differ_between_versions() -> None:
    v1 = PaginatedApiEnvironment(version="1").fingerprint()
    v2 = PaginatedApiEnvironment(version="2").fingerprint()
    assert v1.fingerprint != v2.fingerprint
    assert v1.tool_schema_hashes != v2.tool_schema_hashes


def test_transient_failure_can_be_injected() -> None:
    environment = PaginatedApiEnvironment(version="1")
    environment.fail_next_calls(1)
    failed = environment.list_records(tenant="acme")
    assert failed.ok is False
    assert failed.error is not None
    assert failed.error["error_code"] == "RATE_LIMITED"
    recovered = environment.list_records(tenant="acme")
    assert recovered.ok is True


def test_unknown_tenant_returns_not_found() -> None:
    environment = PaginatedApiEnvironment(version="1")
    result = environment.list_records(tenant="nobody")
    assert result.ok is False
    assert result.error is not None
    assert result.error["error_code"] == "TENANT_NOT_FOUND"


def test_reference_agent_recovers_with_memory() -> None:
    from recap.domain.retrieval import SelectedMemory
    from recap.environments.reference_agent import ReferenceAgent

    environment = PaginatedApiEnvironment(version="1")
    agent = ReferenceAgent()
    without = agent.run(environment)
    assert without.success is False

    memory = SelectedMemory(
        memory_id="mem_test",
        memory_version=1,
        title="Refresh cursor",
        summary="Restart pagination",
        strategy_summary="Refresh cursor",
        actions=[{"action_type": "refresh_pagination_cursor", "tool_name": "list_records"}],
        postconditions=[],
        features=__import__(
            "recap.domain.retrieval", fromlist=["UtilityFeatures"]
        ).UtilityFeatures(),
        compatibility_status="compatible",
        content_hash="sha256:x",
    )
    with_memory = agent.run(PaginatedApiEnvironment(version="1"), memory=memory)
    assert with_memory.success is True
