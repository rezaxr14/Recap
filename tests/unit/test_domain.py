from __future__ import annotations

import pytest

from recap.config import load_settings
from recap.domain.enums import ActionType, MemoryStatus, PredicateResult, SideEffectClass
from recap.domain.environments import (
    EnvironmentFingerprint,
    ToolDescriptor,
    build_fingerprint,
    hash_tool_schema,
)
from recap.domain.memories import ExperienceMemory, MemoryContent
from recap.domain.predicates import Predicate, evaluate_all, evaluate_predicate
from recap.domain.strategies import RepairStrategy, StrategyAction
from recap.errors import TransitionError
from recap.recorder.normalization import normalize_signature
from recap.recorder.redaction import contains_secret, redact_payload, redact_text


def _content(**overrides) -> MemoryContent:
    data = dict(
        title="Refresh expired pagination cursor",
        summary="Restart pagination to obtain a fresh cursor.",
        failure_signature="ValidationError|INVALID_CURSOR|list_records|invalid cursor",
        failure_code="INVALID_CURSOR",
        task_families=["api_pagination"],
        tool_names=["list_records"],
        tool_schema_fingerprints={"list_records": "sha256:abc"},
        environment_type="mock_api",
        preconditions=[
            Predicate(field="error_code", operator="in", value=["INVALID_CURSOR"]),
            Predicate(
                field="tool_schema_fingerprints.list_records",
                operator="eq",
                value="sha256:abc",
            ),
        ],
        strategy=RepairStrategy(
            summary="Refresh the cursor and continue.",
            actions=[
                StrategyAction(
                    action_type=ActionType.REFRESH_PAGINATION_CURSOR,
                    tool_name="list_records",
                    side_effect_class=SideEffectClass.READ_ONLY,
                )
            ],
        ),
        postconditions=[Predicate(field="records_collected", operator="gte", value=1)],
    )
    data.update(overrides)
    return MemoryContent(**data)


def _memory(**overrides) -> ExperienceMemory:
    data = dict(
        workspace_id="ws-1",
        content=_content(),
        status=MemoryStatus.ACTIVE,
        environment_id="mock-api-v1",
        environment_fingerprint="sha256:env1",
    )
    data.update(overrides)
    return ExperienceMemory(**data)


# ------------------------------------------------------------------ redaction
def test_redaction_masks_common_secrets() -> None:
    payload = {
        "authorization": "Bearer abcdefghijklmnopqrstuvwxyz",
        "nested": {"api_key": "sk-live-abcdefghijklmnop", "note": "token=supersecret123"},
        "safe": "list_records",
    }
    redacted, applied = redact_payload(payload)
    assert applied is True
    assert redacted["authorization"] == "[REDACTED]"
    assert redacted["nested"]["api_key"] == "[REDACTED]"
    assert "supersecret123" not in str(redacted)
    assert redacted["safe"] == "list_records"


def test_redaction_detects_private_key_and_url_token() -> None:
    text = "-----BEGIN RSA PRIVATE KEY-----\nMIIabc\n-----END RSA PRIVATE KEY-----"
    assert "[REDACTED_PRIVATE_KEY]" in redact_text(text)[0]
    assert contains_secret("https://api.example.com?token=abcdef123456")
    assert not contains_secret("INVALID_CURSOR")


# --------------------------------------------------------------- normalization
def test_normalization_collapses_volatile_ids() -> None:
    first = normalize_signature(
        message="cursor 123e4567-e89b-12d3-a456-426614174000 invalid at 2024-01-02T03:04:05Z",
        error_type="ValidationError",
        error_code="INVALID_CURSOR",
        tool_name="list_records",
    )
    second = normalize_signature(
        message="cursor 999e4567-e89b-12d3-a456-426614174111 invalid at 2025-05-06T07:08:09Z",
        error_type="ValidationError",
        error_code="INVALID_CURSOR",
        tool_name="list_records",
    )
    assert first == second
    assert "<uuid>" in first
    assert "invalid_cursor" in first.lower()


def test_normalization_keeps_different_errors_distinct() -> None:
    a = normalize_signature(
        message="rate limit exceeded", error_type="TransientError", error_code="RATE_LIMITED"
    )
    b = normalize_signature(
        message="cursor invalid", error_type="ValidationError", error_code="INVALID_CURSOR"
    )
    assert a != b


# ---------------------------------------------------------------- predicates
def test_predicates_unknown_is_not_satisfied() -> None:
    predicate = Predicate(field="environment_version", operator="eq", value="1")
    result, reason = evaluate_predicate(predicate, {})
    assert result is PredicateResult.UNKNOWN
    assert "unknown" in reason


def test_predicates_violation_and_satisfaction() -> None:
    predicates = [
        Predicate(field="error_code", operator="in", value=["INVALID_CURSOR"]),
        Predicate(field="attempt", operator="lte", value=3),
    ]
    satisfied = evaluate_all(predicates, {"error_code": "INVALID_CURSOR", "attempt": 1})
    assert satisfied.result is PredicateResult.SATISFIED
    violated = evaluate_all(predicates, {"error_code": "INVALID_CURSOR", "attempt": 9})
    assert violated.result is PredicateResult.VIOLATED


def test_predicate_rejects_unsafe_expression_fields() -> None:
    with pytest.raises(ValueError):
        Predicate(field="__class__", operator="exists")
    with pytest.raises(ValueError):
        Predicate(field="a", operator="eval", value="1+1")


def test_predicate_normalizes_unambiguous_model_generated_operator_aliases() -> None:
    assert Predicate(field="a", operator="equals", value=1).operator == "eq"
    assert Predicate(field="a", operator=" NOT_EQUALS ", value=1).operator == "neq"
    with pytest.raises(ValueError, match="unsupported predicate operator"):
        Predicate(field="a", operator="approximately", value=1)


# ------------------------------------------------------------------ lifecycle
def test_memory_transition_table_rejects_illegal_moves() -> None:
    memory = _memory(status=MemoryStatus.DRAFT)
    validating = memory.transition(MemoryStatus.VALIDATING, "start validation")
    active = validating.transition(MemoryStatus.ACTIVE, "validated")
    suspect = active.transition(MemoryStatus.SUSPECT, "drift detected")
    assert suspect.status is MemoryStatus.SUSPECT
    with pytest.raises(TransitionError):
        suspect.transition(MemoryStatus.DRAFT, "illegal")


def test_memory_content_hash_is_stable_and_changes_with_content() -> None:
    first = _memory()
    second = _memory()
    assert first.content_hash == second.content_hash
    changed = _memory(content=_content(summary="different summary"))
    assert changed.content_hash != first.content_hash


def test_memory_transition_requires_reason() -> None:
    with pytest.raises(TransitionError):
        _memory(status=MemoryStatus.DRAFT).transition(MemoryStatus.VALIDATING, "  ")


# ----------------------------------------------------------------- fingerprint
def test_fingerprint_is_stable_and_sensitive_to_schema() -> None:
    schema = {"name": "list_records", "parameters": {"cursor": "string"}}
    tool = ToolDescriptor(name="list_records", version="1", schema_hash=hash_tool_schema(schema))
    first = build_fingerprint(
        environment_id="mock-api-v1",
        environment_type="mock_api",
        environment_version="1",
        tools=[tool],
        api_schema=schema,
    )
    second = build_fingerprint(
        environment_id="mock-api-v1",
        environment_type="mock_api",
        environment_version="1",
        tools=[
            ToolDescriptor(name="list_records", version="1", schema_hash=hash_tool_schema(schema))
        ],
        api_schema=schema,
    )
    assert first.fingerprint == second.fingerprint
    changed = build_fingerprint(
        environment_id="mock-api-v2",
        environment_type="mock_api",
        environment_version="2",
        tools=[
            ToolDescriptor(name="list_records", version="2", schema_hash=hash_tool_schema({"x": 1}))
        ],
        api_schema={"x": 1},
    )
    assert changed.fingerprint != first.fingerprint


def test_fingerprint_excludes_unlisted_fields() -> None:
    fingerprint = EnvironmentFingerprint(environment_id="e", environment_version="1")
    payload = fingerprint.canonical_payload()
    assert "api_key" not in payload
    assert "env_vars" not in payload


# --------------------------------------------------------------------- config
def test_config_rejects_dimension_mismatch(tmp_path) -> None:
    from recap.errors import ConfigurationError

    config = tmp_path / "bad.yaml"
    config.write_text(
        "embeddings:\n  dimension: 128\nelasticsearch:\n  vector_dimension: 256\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError):
        load_settings(config)


def test_elasticsearch_isolation_env_overrides(monkeypatch) -> None:
    monkeypatch.setenv("ELASTICSEARCH_INDEX_PREFIX", "recap-unit")
    monkeypatch.setenv("ELASTICSEARCH_MEMORIES_READ_ALIAS", "recap-unit-memories-read")
    monkeypatch.setenv("ELASTICSEARCH_MEMORIES_WRITE_ALIAS", "recap-unit-memories-write")
    monkeypatch.setenv("ELASTICSEARCH_CHUNKS_READ_ALIAS", "recap-unit-chunks-read")
    monkeypatch.setenv("ELASTICSEARCH_CHUNKS_WRITE_ALIAS", "recap-unit-chunks-write")
    monkeypatch.setenv("RECAP_CHUNKS_INDEX_GENERATION", "7")

    settings = load_settings()

    assert settings.elasticsearch.index_prefix == "recap-unit"
    assert settings.elasticsearch.memories_read_alias == "recap-unit-memories-read"
    assert settings.elasticsearch.memories_write_alias == "recap-unit-memories-write"
    assert settings.elasticsearch.chunks_read_alias == "recap-unit-chunks-read"
    assert settings.elasticsearch.chunks_write_alias == "recap-unit-chunks-write"
    assert settings.elasticsearch.chunks_index_generation == 7


def test_safe_dump_redacts_credentials(tmp_path) -> None:
    config = tmp_path / "c.yaml"
    config.write_text(
        "storage:\n  postgres_url: postgresql://user:secret@host/db\n"
        "elasticsearch:\n  api_key: super-secret\n",
        encoding="utf-8",
    )
    settings = load_settings(config)
    dumped = str(settings.safe_dump())
    assert "secret" not in dumped
    assert "super-secret" not in dumped
