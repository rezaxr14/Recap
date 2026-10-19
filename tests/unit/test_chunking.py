"""Pure chunker tests: determinism, atomic units, traceability, redaction."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from recap.chunking.chunker import build_chunks
from recap.chunking.tokens import estimate_tokens
from recap.config import ChunkingSettings
from recap.domain.chunks import chunk_id_for
from recap.domain.enums import (
    ActionType,
    ChunkType,
    EventType,
    MemoryStatus,
    SideEffectClass,
    ValidationStatus,
)
from recap.domain.memories import ExperienceMemory, MemoryContent
from recap.domain.predicates import Predicate
from recap.domain.repairs import ValidationResult
from recap.domain.runs import RunEvent
from recap.domain.strategies import RepairStrategy, StrategyAction


def make_memory(*, summary: str = "Refresh the expired pagination cursor.") -> ExperienceMemory:
    content = MemoryContent(
        title="Refresh expired pagination cursor",
        summary=summary,
        failure_signature="ValidationError|INVALID_CURSOR|list_records",
        failure_code="INVALID_CURSOR",
        task_families=["api_pagination"],
        tool_names=["list_records"],
        tool_versions={"list_records": "2"},
        tool_schema_fingerprints={"list_records": "schema-v2"},
        environment_type="mock_api",
        preconditions=[Predicate(field="error_code", operator="eq", value="INVALID_CURSOR")],
        strategy=RepairStrategy(
            summary="Re-list from the first page and use the returned page_token.",
            actions=[
                StrategyAction(
                    action_type=ActionType.REFRESH_PAGINATION_CURSOR,
                    tool_name="list_records",
                    description="Call list_records without the stale cursor.",
                    side_effect_class=SideEffectClass.READ_ONLY,
                )
            ],
        ),
        postconditions=[Predicate(field="records_collected", operator="gte", value=1)],
        known_limitations=["Only applies to the v2 page_token schema."],
    )
    return ExperienceMemory(
        workspace_id="ws-chunk-test",
        content=content,
        status=MemoryStatus.ACTIVE,
        environment_id="mock-api-v2",
        environment_fingerprint="envfp-v2",
    )


def make_events(run_id: str = "run_chunk_test") -> list[RunEvent]:
    base = datetime(2026, 10, 19, 12, 0, tzinfo=UTC)
    rows = [
        (EventType.RUN_STARTED, {"task": "collect all records"}, None),
        (
            EventType.TOOL_CALL_REQUESTED,
            {"arguments": {"tenant": "acme", "cursor": "cur_stale_123", "limit": 3}},
            "list_records",
        ),
        (
            EventType.TOOL_CALL_FAILED,
            {
                "request": {"tenant": "acme", "cursor": "cur_stale_123", "limit": 3},
                "error_type": "ValidationError",
                "error_code": "INVALID_CURSOR",
                "error_message": "cursor 'cur_stale_123' is invalid or expired",
                "http_status": 400,
            },
            "list_records",
        ),
        (
            EventType.TOOL_CALL_REQUESTED,
            {"arguments": {"tenant": "acme", "limit": 3}},
            "list_records",
        ),
        (
            EventType.TOOL_CALL_SUCCEEDED,
            {
                "result": {
                    "records": [{"id": "rec-001"}, {"id": "rec-002"}],
                    "next_cursor": "cur_next_456",
                    "has_more": True,
                }
            },
            "list_records",
        ),
        (EventType.POSTCONDITION_CHECKED, {"passed": True, "detail": "2 records"}, None),
        (EventType.RUN_FINISHED, {"success": True}, None),
    ]
    return [
        RunEvent(
            id=f"evt_{index}",
            run_id=run_id,
            sequence_number=index,
            event_type=event_type,
            timestamp=base + timedelta(seconds=index),
            tool_name=tool,
            tool_version="2" if tool else None,
            payload_redacted=payload,
        )
        for index, (event_type, payload, tool) in enumerate(rows, start=1)
    ]


def test_card_only_is_deterministic_and_semantic_text_is_preserved() -> None:
    memory = make_memory()
    config = ChunkingSettings(strategy="card_only")
    first = build_chunks(memory, events=make_events(), config=config)
    second = build_chunks(memory, events=make_events(), config=config)

    assert len(first.chunks) == 1
    chunk = first.chunks[0]
    assert chunk.chunk_type is ChunkType.MEMORY_CARD
    assert chunk.run_id is None
    assert "Strategy:" in chunk.text
    assert "Refresh expired pagination cursor" in chunk.text
    assert chunk.chunk_id == second.chunks[0].chunk_id
    assert chunk.text == second.chunks[0].text
    assert chunk.text_hash.startswith("sha256:")
    assert chunk.token_estimate > 0


def test_event_aware_chunks_keep_atomic_action_pairs_and_failure_evidence() -> None:
    result = build_chunks(make_memory(), events=make_events())
    types = {chunk.chunk_type for chunk in result.chunks}
    assert {
        ChunkType.MEMORY_CARD,
        ChunkType.FAILURE_WINDOW,
        ChunkType.ACTION_OBSERVATION,
    } <= types

    failure = next(chunk for chunk in result.chunks if chunk.chunk_type is ChunkType.FAILURE_WINDOW)
    assert "INVALID_CURSOR" in failure.text
    assert "cur_stale_123" not in failure.text
    assert "<CURSOR>" in failure.text
    assert "cur_stale_123" in failure.exact_ids
    assert {"evt_2", "evt_3"} <= set(failure.event_ids)
    assert failure.seq_start >= 2
    assert failure.run_id == "run_chunk_test"

    # Every action chunk contains complete request/observation pairs, never a
    # request without its outcome.
    for chunk in result.chunks:
        if chunk.chunk_type is not ChunkType.ACTION_OBSERVATION:
            continue
        ids = set(chunk.event_ids)
        assert not ("evt_2" in ids) ^ ("evt_3" in ids)
        assert not ("evt_4" in ids) ^ ("evt_5" in ids)


def test_chunker_redacts_secrets_and_fails_closed() -> None:
    events = make_events()
    events[1].payload_redacted = {
        "arguments": {
            "tenant": "acme",
            "cursor": "cur_stale_123",
            "api_key": "sk-live-abcdefghijklmnop",
        }
    }
    result = build_chunks(make_memory(), events=events)
    for chunk in result.chunks:
        assert "sk-live-abcdefghijklmnop" not in chunk.text
        assert "[REDACTED]" in chunk.text or "api_key" not in chunk.text


def test_memory_card_redacts_secret_in_authored_summary() -> None:
    memory = make_memory(summary="Call with Bearer abcdefghijklmnopqrstuvwxyz123456.")
    result = build_chunks(memory, config=ChunkingSettings(strategy="card_only"))
    assert "abcdefghijklmnopqrstuvwxyz123456" not in result.chunks[0].text
    assert "[REDACTED]" in result.chunks[0].text


def test_naive_fixed_is_labeled_baseline() -> None:
    config = ChunkingSettings(strategy="naive_fixed", target_tokens=32, max_tokens=64)
    result = build_chunks(make_memory(), events=make_events(), config=config)
    assert any(chunk.chunk_type is ChunkType.NAIVE_FIXED for chunk in result.chunks)
    assert all(chunk.chunker_version == "1" for chunk in result.chunks)


def test_oversized_event_is_truncated_with_marker() -> None:
    events = make_events()
    events[1].payload_redacted = {"arguments": {"blob": "x " * 6000}}
    config = ChunkingSettings(target_tokens=32, max_tokens=48, max_chunks_per_memory=8)
    result = build_chunks(make_memory(), events=events, config=config)
    action = [chunk for chunk in result.chunks if chunk.chunk_type is ChunkType.ACTION_OBSERVATION]
    assert action
    assert all(chunk.token_estimate <= 500 for chunk in action)
    assert any("truncated" in chunk.text for chunk in action) or any(
        chunk.truncated for chunk in action
    )


def test_max_chunks_caps_evidence_and_reports_warning() -> None:
    config = ChunkingSettings(max_chunks_per_memory=2)
    result = build_chunks(make_memory(), events=make_events(), config=config)
    assert len(result.chunks) == 2
    assert result.warnings
    assert result.chunks[0].chunk_type is ChunkType.MEMORY_CARD


def test_validation_evidence_chunk_can_be_built() -> None:
    validation = ValidationResult(
        candidate_id="cand_1",
        validator_name="mock_sandbox",
        status=ValidationStatus.PASSED,
        observed_outcome="failure reproduced and postcondition passed",
        checks=[{"name": "reproduced", "passed": True}],
    )
    result = build_chunks(make_memory(), events=make_events(), validation=validation)
    chunk = next(item for item in result.chunks if item.chunk_type is ChunkType.VALIDATION_EVIDENCE)
    assert "mock_sandbox" in chunk.text
    assert "reproduced" in chunk.text
    assert "cand_1" in chunk.exact_ids


def test_chunk_id_is_stable_and_run_scoped() -> None:
    a = chunk_id_for(
        workspace_id="ws",
        memory_id="mem",
        memory_version=1,
        chunk_type="failure_window",
        seq_start=1,
        seq_end=2,
        chunker_version="1",
        text_hash="sha256:abc",
        run_id="run_a",
    )
    b = chunk_id_for(
        workspace_id="ws",
        memory_id="mem",
        memory_version=1,
        chunk_type="failure_window",
        seq_start=1,
        seq_end=2,
        chunker_version="1",
        text_hash="sha256:abc",
        run_id="run_b",
    )
    assert a != b
    assert len(a) == 32


def test_estimate_is_conservative_and_monotonic() -> None:
    short = estimate_tokens("one two three")
    long = estimate_tokens("one two three four five six")
    assert 0 < short < long


@pytest.mark.parametrize("strategy", ["event_aware", "card_only", "naive_fixed"])
def test_build_never_returns_unredacted_chunk(strategy: str) -> None:
    result = build_chunks(
        make_memory(),
        events=make_events(),
        config=ChunkingSettings(strategy=strategy),
    )
    for chunk in result.chunks:
        assert chunk.text.strip()
        assert chunk.token_estimate > 0
        assert chunk.redaction_rule_version == "redaction-v1"
