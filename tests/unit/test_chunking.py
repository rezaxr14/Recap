"""Pure chunker tests: determinism, atomic units, traceability, redaction."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

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
    # ``seq_start`` is the first event present (context included); the failure
    # position is the failed result event, carried separately.
    assert failure.seq_start == 1
    assert failure.anchor_seq == 3
    assert failure.run_id == "run_chunk_test"

    # Every action chunk contains complete request/observation pairs, never a
    # request without its outcome.
    for chunk in result.chunks:
        if chunk.chunk_type is not ChunkType.ACTION_OBSERVATION:
            continue
        ids = set(chunk.event_ids)
        assert not ("evt_2" in ids) ^ ("evt_3" in ids)
        assert not ("evt_4" in ids) ^ ("evt_5" in ids)


@pytest.mark.parametrize("factory_name", ["make_events", "make_long_window_events"])
def test_event_aware_chunks_preserve_traceability_invariants(factory_name: str) -> None:
    factory = globals()[factory_name]
    events = factory()
    source_seq = {event.id: event.sequence_number for event in events}
    config = ChunkingSettings(
        strategy="event_aware",
        target_tokens=24,
        max_tokens=96,
        failure_window_events=12,
    )
    result = build_chunks(make_memory(), events=events, config=config)

    pairs: dict[str, str] = {}
    for event, nxt in zip(events, events[1:], strict=False):
        if event.event_type is EventType.TOOL_CALL_REQUESTED and nxt.event_type in {
            EventType.TOOL_CALL_SUCCEEDED,
            EventType.TOOL_CALL_FAILED,
        }:
            pairs[event.id] = nxt.id

    failure_chunks = [
        chunk for chunk in result.chunks if chunk.chunk_type is ChunkType.FAILURE_WINDOW
    ]
    assert failure_chunks
    anchors = [chunk.anchor_seq for chunk in failure_chunks if chunk.anchor_seq]
    assert anchors, "the failure anchor must be present in at least one failure-window chunk"
    failure_anchor = anchors[0]
    assert any(chunk.seq_start <= failure_anchor <= chunk.seq_end for chunk in failure_chunks)

    for chunk in result.chunks:
        assert chunk.token_estimate <= config.max_tokens
        if not chunk.event_ids:
            continue
        # Every id points at the source run, and the sequence range is the
        # actual range of the events in the chunk (not a clamped anchor).
        assert set(chunk.event_ids) <= source_seq.keys()
        seqs = [source_seq[event_id] for event_id in chunk.event_ids]
        assert chunk.seq_start == min(seqs)
        assert chunk.seq_end == max(seqs)
        # Action/observation pairs are atomic: a chunk never contains one half.
        for request_id, result_id in pairs.items():
            assert (request_id in chunk.event_ids) == (result_id in chunk.event_ids)
        # Only a failure-window pack that actually contains the failure carries
        # an anchor (a nonzero seq); continuation packs of a long tail carry 0.
        assert bool(chunk.anchor_seq) == (
            chunk.chunk_type is ChunkType.FAILURE_WINDOW
            and chunk.seq_start <= failure_anchor <= chunk.seq_end
        )


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
    assert all(chunk.chunker_version == config.version for chunk in result.chunks)


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


# --------------------------------------------------------------- P1 regressions
def make_long_window_events(run_id: str = "run_long_window") -> list[RunEvent]:
    base = datetime(2026, 10, 19, 12, 0, tzinfo=UTC)
    rows: list[tuple[EventType, dict[str, object], str | None]] = [
        (EventType.RUN_STARTED, {"task": "collect all records"}, None)
    ]
    for page in range(8):
        rows.append(
            (
                EventType.TOOL_CALL_REQUESTED,
                {"arguments": {"page": page, "cursor": f"cur_{page}"}},
                "list_records",
            )
        )
        rows.append(
            (
                EventType.TOOL_CALL_SUCCEEDED,
                {
                    "result": {
                        "records": [{"id": f"rec-{page}"}],
                        "next_cursor": f"cur_{page + 1}",
                    }
                },
                "list_records",
            )
        )
    rows.extend(
        [
            (
                EventType.TOOL_CALL_REQUESTED,
                {"arguments": {"cursor": "cur_stale"}},
                "list_records",
            ),
            (
                EventType.TOOL_CALL_FAILED,
                {
                    "request": {"cursor": "cur_stale"},
                    "error_type": "ValidationError",
                    "error_code": "INVALID_CURSOR",
                    "error_message": "cursor is invalid or expired",
                },
                "list_records",
            ),
            (
                EventType.TOOL_CALL_REQUESTED,
                {"arguments": {"limit": 3}},
                "list_records",
            ),
            (
                EventType.TOOL_CALL_SUCCEEDED,
                {"result": {"records": [{"id": "rec-ok"}], "next_cursor": "n"}},
                "list_records",
            ),
            (EventType.POSTCONDITION_CHECKED, {"passed": True}, None),
            (EventType.RUN_FINISHED, {"success": True}, None),
        ]
    )
    return [
        RunEvent(
            id=f"evt_{index}",
            run_id=run_id,
            sequence_number=index,
            event_type=event_type,
            timestamp=base + timedelta(seconds=index),
            tool_name=tool,
            payload_redacted=payload,
        )
        for index, (event_type, payload, tool) in enumerate(rows, start=1)
    ]


def test_long_failure_window_keeps_anchor_and_preceding_context() -> None:
    config = ChunkingSettings(target_tokens=64, max_tokens=260, failure_window_events=12)
    result = build_chunks(make_memory(), events=make_long_window_events(), config=config)
    failure = next(chunk for chunk in result.chunks if chunk.chunk_type is ChunkType.FAILURE_WINDOW)

    assert "INVALID_CURSOR" in failure.text
    assert failure.anchor_seq > 0
    # At least one preceding unit was affordable alongside the failure tail.
    assert failure.seq_start < failure.anchor_seq
    assert failure.token_estimate <= config.max_tokens
    assert failure.truncated is False


@pytest.mark.parametrize("strategy", ["event_aware", "card_only", "naive_fixed"])
def test_token_estimate_never_exceeds_max_after_header(strategy: str) -> None:
    # The header alone is larger than this budget, so the final guard must fit
    # the *whole* text, not just the packed units.
    config = ChunkingSettings(
        strategy=strategy, target_tokens=16, max_tokens=16, token_multiplier=1.0
    )
    result = build_chunks(make_memory(), events=make_events(), config=config)
    assert result.chunks
    for chunk in result.chunks:
        assert chunk.token_estimate <= config.max_tokens, chunk.chunk_type


def test_untouched_event_aware_chunks_are_not_marked_truncated() -> None:
    result = build_chunks(make_memory(), events=make_events())
    failure = next(chunk for chunk in result.chunks if chunk.chunk_type is ChunkType.FAILURE_WINDOW)
    assert failure.truncated is False


def test_uuid_version_7_is_redacted_as_an_identifier() -> None:
    events = make_events()
    uuid7 = "01890a5d-ac96-774b-bcce-b302099a8057"
    events[1].payload_redacted = {"arguments": {"note": f"trace {uuid7}"}}
    result = build_chunks(make_memory(), events=events)
    failure = next(chunk for chunk in result.chunks if chunk.chunk_type is ChunkType.FAILURE_WINDOW)
    assert uuid7 not in failure.text
    assert "<UUID>" in failure.text
    assert uuid7 in failure.exact_ids


def test_naive_slice_is_not_marked_truncated_just_for_slicing() -> None:
    config = ChunkingSettings(strategy="naive_fixed", target_tokens=32, max_tokens=512)
    result = build_chunks(make_memory(), events=make_events(), config=config)
    naive = [chunk for chunk in result.chunks if chunk.chunk_type is ChunkType.NAIVE_FIXED]
    assert len(naive) > 1
    assert all(chunk.truncated is False for chunk in naive)


def test_unknown_chunking_setting_is_rejected() -> None:
    with pytest.raises(ValidationError):
        ChunkingSettings(max_token=64)  # type: ignore[call-arg]


FIXED_CLOCK = datetime(2026, 10, 20, 12, 0, tzinfo=UTC)


def deterministic_memory() -> ExperienceMemory:
    return make_memory().model_copy(update={"id": "mem_golden", "created_at": FIXED_CLOCK})


def test_two_builds_with_a_fixed_clock_are_identical() -> None:
    first = build_chunks(deterministic_memory(), events=make_events(), created_at=FIXED_CLOCK)
    second = build_chunks(deterministic_memory(), events=make_events(), created_at=FIXED_CLOCK)
    assert first.chunks == second.chunks


CROSS_PROCESS_SCRIPT = r"""
import json
import sys
from datetime import datetime

from recap.chunking.chunker import build_chunks
from recap.config import ChunkingSettings
from recap.domain.memories import ExperienceMemory
from recap.domain.runs import RunEvent

payload = json.load(sys.stdin)
memory = ExperienceMemory.model_validate(payload["memory"])
events = [RunEvent.model_validate(event) for event in payload["events"]]
config = ChunkingSettings.model_validate(payload["config"])
result = build_chunks(
    memory,
    events=events,
    run_id=payload["run_id"],
    config=config,
    created_at=datetime.fromisoformat(payload["created_at"]),
)
print(json.dumps([[chunk.chunk_id, chunk.text_hash] for chunk in result.chunks]))
"""


def test_build_is_stable_across_processes() -> None:
    config = ChunkingSettings()
    payload = {
        "memory": deterministic_memory().model_dump(mode="json"),
        "events": [event.model_dump(mode="json") for event in make_events()],
        "config": config.model_dump(mode="json"),
        "run_id": "run_chunk_test",
        "created_at": FIXED_CLOCK.isoformat(),
    }
    completed = subprocess.run(
        [sys.executable, "-c", CROSS_PROCESS_SCRIPT],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parents[2],
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    remote = json.loads(completed.stdout)
    local = build_chunks(
        deterministic_memory(),
        events=make_events(),
        run_id="run_chunk_test",
        config=config,
        created_at=FIXED_CLOCK,
    )
    assert remote == [[chunk.chunk_id, chunk.text_hash] for chunk in local.chunks]


def test_chunk_ids_match_the_golden_file() -> None:
    golden_path = Path(__file__).resolve().parents[1] / "golden" / "chunk_ids.json"
    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    result = build_chunks(deterministic_memory(), events=make_events(), created_at=FIXED_CLOCK)
    assert {
        "chunker_version": result.chunks[0].chunker_version,
        "chunks": [
            {
                "chunk_type": str(chunk.chunk_type),
                "chunk_id": chunk.chunk_id,
                "text_hash": chunk.text_hash,
                "seq_start": chunk.seq_start,
                "seq_end": chunk.seq_end,
                "anchor_seq": chunk.anchor_seq,
                "event_ids": chunk.event_ids,
                "token_estimate": chunk.token_estimate,
                "truncated": chunk.truncated,
                "chunk_strategy": chunk.chunk_strategy,
            }
            for chunk in result.chunks
        ],
    } == golden
