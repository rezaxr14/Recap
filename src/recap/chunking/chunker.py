"""Pure, deterministic event-aware chunking.

No I/O, no model calls, no tokenizer downloads. The chunker receives already
persisted domain objects and returns immutable chunks. Redaction is applied
again at this boundary and the build fails closed if a configured secret
pattern survives.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from recap.chunking.tokens import estimate_tokens
from recap.config import ChunkingSettings
from recap.domain.chunks import MemoryChunk, chunk_id_for
from recap.domain.enums import ChunkType, EventType
from recap.domain.memories import ExperienceMemory, build_embedding_text
from recap.domain.repairs import ValidationResult
from recap.domain.runs import RunEvent, utc_now
from recap.recorder.redaction import (
    REDACTION_RULE_VERSION,
    contains_secret,
    redact_text,
    redact_value,
)

MAX_UNIT_CHARS = 12_000
TRUNCATION_MARKER = "[...truncated {count} chars...]"

_UUID_RE = re.compile(
    r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
)
_ISO_TIMESTAMP_RE = re.compile(
    r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?\b"
)
_QUOTED_CURSOR_RE = re.compile(
    r"(?i)\b(page[_-]?token|cursor|next[_-]?cursor)\b(\s*(?:is|:|=)?\s*['\"])([^'\"]{1,300})(['\"])"
)
_CURSOR_KEYS = {"cursor", "page_token", "next_cursor", "next_page_token", "continuation_token"}
_REQUEST_ID_KEYS = {
    "request_id",
    "trace_id",
    "span_id",
    "correlation_id",
    "idempotency_key",
    "client_request_id",
    # Internal ids are random per run. Leaving them in the chunk text makes the
    # same logical memory render differently on every seed and destabilizes
    # lexical scoring; the values stay available via ``exact_ids``.
    "run_id",
    "event_id",
    "snapshot_id",
    "memory_snapshot_id",
    "candidate_id",
    "validation_id",
    "attempt_id",
}
_TIMESTAMP_KEYS = {
    "timestamp",
    "created_at",
    "updated_at",
    "expires_at",
    "started_at",
    "finished_at",
}


@dataclass(frozen=True)
class AtomicUnit:
    """An indivisible rendering of one event or one action/observation pair."""

    event_ids: tuple[str, ...]
    event_types: tuple[str, ...]
    seq_start: int
    seq_end: int
    text: str
    tool_name: str = ""
    ok: bool | None = None
    failed: bool = False
    failure_code: str = ""
    terminal: bool = False
    rows: tuple[dict[str, Any], ...] = ()
    exact_ids: tuple[str, ...] = ()
    truncated: bool = False


@dataclass(frozen=True)
class ChunkBuildResult:
    chunks: tuple[MemoryChunk, ...]
    warnings: tuple[str, ...] = ()
    truncated: bool = False


class _ExactIds:
    def __init__(self) -> None:
        self._values: list[str] = []

    def add(self, value: Any) -> None:
        if value is None:
            return
        text = str(value)
        if text and text != "[REDACTED]" and text not in self._values:
            self._values.append(text)

    def extend(self, values: Sequence[str]) -> None:
        for value in values:
            self.add(value)

    def snapshot(self) -> list[str]:
        return list(self._values[:200])


class _Renderer:
    def __init__(self) -> None:
        self.exact_ids = _ExactIds()

    def scalar(self, value: Any, *, key: str = "") -> str:
        lowered = key.lower().replace("-", "_")
        if isinstance(value, str):
            if lowered in _CURSOR_KEYS:
                self.exact_ids.add(value)
                return "<CURSOR>"
            if lowered in _REQUEST_ID_KEYS:
                self.exact_ids.add(value)
                return "<REQUEST_ID>"
            if lowered in _TIMESTAMP_KEYS:
                self.exact_ids.add(value)
                return "<TIMESTAMP>"
            text, _ = redact_text(value)
            text = _UUID_RE.sub(lambda match: self._replace_id(match.group(0)), text)
            text = _ISO_TIMESTAMP_RE.sub(lambda match: self._replace_id(match.group(0)), text)

            def quoted(match: re.Match[str]) -> str:
                self.exact_ids.add(match.group(3))
                return f"{match.group(1)}{match.group(2)}<CURSOR>{match.group(4)}"

            return _QUOTED_CURSOR_RE.sub(quoted, text)
        if isinstance(value, dict):
            return self.mapping(value)
        if isinstance(value, (list, tuple)):
            return "[" + ",".join(self.scalar(item) for item in value) + "]"
        return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)

    def _replace_id(self, value: str) -> str:
        self.exact_ids.add(value)
        return "<UUID>" if _UUID_RE.fullmatch(value) else "<TIMESTAMP>"

    def mapping(self, value: dict[str, Any]) -> str:
        parts: list[str] = []
        for key in sorted(value, key=str):
            parts.append(f'"{key}":{self.scalar(value[key], key=key)}')
        return "{" + ",".join(parts) + "}"


def _bounded(text: str, limit: int = MAX_UNIT_CHARS) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    removed = len(text) - limit
    # Keep a deterministic head and tail; never silently drop the end.
    head = limit * 2 // 3
    tail = limit - head
    return f"{text[:head]}\n{TRUNCATION_MARKER.format(count=removed)}\n{text[-tail:]}", True


def _tool_name(event: RunEvent) -> str:
    if event.tool_name:
        return event.tool_name
    payload = event.payload_redacted
    value = payload.get("tool_name") or payload.get("tool") or payload.get("action")
    return str(value) if value else ""


def _unit_from_event(event: RunEvent, renderer: _Renderer) -> AtomicUnit:
    payload, _ = redact_value(event.payload_redacted)
    tool = _tool_name(event)
    body = _renderer_value(renderer, payload)
    lines = [f"Step: {event.event_type}"]
    if tool:
        lines.append(f"Tool: {tool}")
    if body and body != "{}":
        lines.append(f"Data: {body}")
    text, truncated = _bounded("\n".join(lines))
    failure_code = ""
    if event.event_type is EventType.TOOL_CALL_FAILED:
        failure_code = str(payload.get("error_code") or payload.get("code") or "")
    return AtomicUnit(
        event_ids=(event.id,),
        event_types=(str(event.event_type),),
        seq_start=event.sequence_number,
        seq_end=event.sequence_number,
        text=text,
        tool_name=tool,
        ok=False if event.event_type is EventType.TOOL_CALL_FAILED else None,
        failed=event.event_type is EventType.TOOL_CALL_FAILED,
        failure_code=failure_code,
        terminal=event.event_type in {EventType.RUN_FINISHED, EventType.POSTCONDITION_CHECKED},
        rows=(payload,),
        exact_ids=tuple(renderer.exact_ids.snapshot()),
        truncated=truncated,
    )


def _renderer_value(renderer: _Renderer, value: Any) -> str:
    return renderer.scalar(value)


def _pair_units(events: Sequence[RunEvent]) -> list[AtomicUnit]:
    ordered = sorted(events, key=lambda event: event.sequence_number)
    units: list[AtomicUnit] = []
    index = 0
    while index < len(ordered):
        event = ordered[index]
        if event.event_type is EventType.TOOL_CALL_REQUESTED:
            nxt = ordered[index + 1] if index + 1 < len(ordered) else None
            if nxt is not None and nxt.event_type in {
                EventType.TOOL_CALL_SUCCEEDED,
                EventType.TOOL_CALL_FAILED,
            }:
                renderer = _Renderer()
                request, _ = redact_value(event.payload_redacted)
                result, _ = redact_value(nxt.payload_redacted)
                tool = _tool_name(event) or _tool_name(nxt)
                request_text = renderer.mapping(request)
                if nxt.event_type is EventType.TOOL_CALL_SUCCEEDED:
                    result_payload = result.get("result", result)
                    result_text = renderer.scalar(result_payload)
                    status = "Observation"
                    ok: bool | None = True
                    failed = False
                    failure_code = ""
                else:
                    # Failed events carry the request again plus error fields.
                    error_fields = {
                        key: value
                        for key, value in result.items()
                        if key
                        in {"error_type", "error_code", "error_message", "http_status", "detail"}
                    }
                    code = str(result.get("error_code") or result.get("code") or "")
                    message = str(result.get("error_message") or result.get("message") or "")
                    if error_fields:
                        result_text = renderer.mapping(error_fields)
                    else:
                        result_text = renderer.scalar(f"{code}: {message}".strip(": "))
                    status = "Error"
                    ok = False
                    failed = True
                    failure_code = code
                lines = [f"Action: {tool or 'unknown'}", f"Request: {request_text}"]
                lines.append(f"{status}: {result_text}")
                text, truncated = _bounded("\n".join(lines))
                units.append(
                    AtomicUnit(
                        event_ids=(event.id, nxt.id),
                        event_types=(str(event.event_type), str(nxt.event_type)),
                        seq_start=event.sequence_number,
                        seq_end=nxt.sequence_number,
                        text=text,
                        tool_name=tool,
                        ok=ok,
                        failed=failed,
                        failure_code=failure_code,
                        rows=(request, result),
                        exact_ids=tuple(renderer.exact_ids.snapshot()),
                        truncated=truncated,
                    )
                )
                index += 2
                continue
        units.append(_unit_from_event(event, _Renderer()))
        index += 1
    return units


def _header(memory: ExperienceMemory, chunk_type: ChunkType, units: Sequence[AtomicUnit]) -> str:
    tool = next((unit.tool_name for unit in units if unit.tool_name), "")
    if not tool and memory.content.tool_names:
        tool = memory.content.tool_names[0]
    error = next((unit.failure_code for unit in units if unit.failure_code), "")
    error = error or memory.content.failure_code or "none"
    environment = memory.environment_id or memory.content.environment_type or "unknown"
    return f"Tool: {tool or 'unknown'} | Error: {error} | Env: {environment} | Chunk: {chunk_type}"


def _truncate_unit(unit: AtomicUnit, max_tokens: int, multiplier: float) -> tuple[AtomicUnit, bool]:
    if estimate_tokens(unit.text, multiplier) <= max_tokens:
        return unit, False
    # Hard character budget derived from the conservative token estimate.
    # There is no loop: a pathological max_tokens must still terminate. The
    # marker makes the loss visible instead of silently dropping content.
    target = max(64, int(max_tokens * 2.2))
    while target >= 48:
        removed = len(unit.text) - target
        head = target * 2 // 3
        tail = target - head
        text = (
            f"{unit.text[:head]}\n{TRUNCATION_MARKER.format(count=max(removed, 0))}\n"
            f"{unit.text[-tail:]}"
        )
        if estimate_tokens(text, multiplier) <= max_tokens:
            break
        target = int(target * 0.7)
    return (
        AtomicUnit(
            event_ids=unit.event_ids,
            event_types=unit.event_types,
            seq_start=unit.seq_start,
            seq_end=unit.seq_end,
            text=text,
            tool_name=unit.tool_name,
            ok=unit.ok,
            failed=unit.failed,
            failure_code=unit.failure_code,
            terminal=unit.terminal,
            rows=unit.rows,
            exact_ids=unit.exact_ids,
            truncated=True,
        ),
        True,
    )


def _pack_units(
    units: Sequence[AtomicUnit],
    *,
    target_tokens: int,
    max_tokens: int,
    overlap_units: int,
    multiplier: float,
    overhead_tokens: int = 0,
) -> list[tuple[list[AtomicUnit], bool]]:
    # The contextual header is part of the chunk text, so its tokens must be
    # reserved from the packing budget or ``token_estimate`` can exceed
    # ``max_tokens`` after the header is prepended.
    budget = max(1, target_tokens - overhead_tokens)
    hard_limit = max(1, max_tokens - overhead_tokens)
    packed: list[tuple[list[AtomicUnit], bool]] = []
    index = 0
    while index < len(units):
        selected: list[AtomicUnit] = []
        chars = 0
        truncated = False
        cursor = index
        while cursor < len(units):
            unit = units[cursor]
            tokens = estimate_tokens(unit.text, multiplier)
            if not selected and tokens > hard_limit:
                unit, truncated = _truncate_unit(unit, hard_limit, multiplier)
                tokens = estimate_tokens(unit.text, multiplier)
            if selected and chars + tokens > budget:
                break
            selected.append(unit)
            chars += tokens
            cursor += 1
            if chars >= budget:
                break
        packed.append((selected, truncated or any(unit.truncated for unit in selected)))
        if cursor >= len(units):
            break
        index = max(index + 1, cursor - max(0, overlap_units))
    return packed


def _fit_text(text: str, max_tokens: int, multiplier: float) -> tuple[str, bool]:
    """Guarantee ``token_estimate <= max_tokens`` after the header is added."""
    if estimate_tokens(text, multiplier) <= max_tokens:
        return text, False
    target = max(64, int(max_tokens * 2.2))
    while target >= 8:
        removed = max(len(text) - target, 0)
        head = target * 2 // 3
        tail = target - head
        candidate = f"{text[:head]}\n{TRUNCATION_MARKER.format(count=removed)}\n{text[-tail:]}"
        if estimate_tokens(candidate, multiplier) <= max_tokens:
            return candidate, True
        target = int(target * 0.7)
    return text[:1], True


def _finalize(
    *,
    memory: ExperienceMemory,
    chunk_type: ChunkType,
    seq_start: int,
    seq_end: int,
    event_ids: Sequence[str],
    body: str,
    run_id: str | None,
    header: str,
    exact_ids: Sequence[str],
    config: ChunkingSettings,
    truncated: bool,
    failure_code: str | None = None,
    anchor_seq: int = 0,
    created_at: datetime | None = None,
) -> MemoryChunk:
    text = f"{header}\n\n{body}".strip()
    redacted, _ = redact_text(text)
    if contains_secret(redacted):
        raise ValueError("chunk text still contains a configured secret pattern after redaction")
    redacted, fit_truncated = _fit_text(redacted, config.max_tokens, config.token_multiplier)
    truncated = truncated or fit_truncated
    text_hash = f"sha256:{hashlib.sha256(redacted.encode('utf-8')).hexdigest()}"
    chunk_id = chunk_id_for(
        workspace_id=memory.workspace_id,
        memory_id=memory.id,
        memory_version=memory.memory_version,
        chunk_type=str(chunk_type),
        seq_start=seq_start,
        seq_end=seq_end,
        chunker_version=config.version,
        text_hash=text_hash,
        run_id=run_id,
        chunk_strategy=config.strategy,
    )
    return MemoryChunk(
        chunk_id=chunk_id,
        workspace_id=memory.workspace_id,
        memory_id=memory.id,
        memory_version=memory.memory_version,
        run_id=run_id,
        chunk_type=chunk_type,
        seq_start=seq_start,
        seq_end=seq_end,
        anchor_seq=anchor_seq,
        event_ids=list(event_ids),
        text=redacted,
        text_hash=text_hash,
        token_estimate=estimate_tokens(redacted, config.token_multiplier),
        truncated=truncated,
        chunker_version=config.version,
        chunk_strategy=config.strategy,
        redaction_rule_version=REDACTION_RULE_VERSION,
        failure_code=memory.content.failure_code if failure_code is None else failure_code,
        tool_names=list(memory.content.tool_names),
        tool_versions=[f"{key}:{value}" for key, value in memory.content.tool_versions.items()],
        environment_id=memory.environment_id,
        environment_fingerprint=memory.environment_fingerprint,
        task_families=list(memory.content.task_families),
        exact_ids=list(dict.fromkeys(exact_ids))[:200],
        created_at=created_at or utc_now(),
    )


def _card_chunk(
    memory: ExperienceMemory,
    config: ChunkingSettings,
    created_at: datetime,
) -> MemoryChunk:
    body = build_embedding_text(memory)
    return _finalize(
        memory=memory,
        chunk_type=ChunkType.MEMORY_CARD,
        seq_start=0,
        seq_end=0,
        event_ids=(),
        body=body,
        run_id=None,
        header=_header(memory, ChunkType.MEMORY_CARD, ()),
        exact_ids=(),
        config=config,
        truncated=False,
        created_at=created_at,
    )


def _failure_window(
    units: Sequence[AtomicUnit],
    config: ChunkingSettings,
    *,
    header_tokens: int = 0,
) -> list[tuple[list[AtomicUnit], bool]]:
    """Keep the failure-to-outcome window and as much preceding context as fits.

    The failure anchor is mandatory. Context is added nearest-first only when it
    fits alongside the whole tail; if the tail alone exceeds the budget it is
    packed on its own (never dropping the anchor).
    """

    failure_index = next((index for index, unit in enumerate(units) if unit.failed), None)
    if failure_index is None:
        return []
    end = len(units)
    for index in range(failure_index + 1, len(units)):
        if units[index].ok is True or units[index].terminal:
            end = index + 1
            break
    end = min(end, failure_index + config.failure_window_events)
    context_start = max(0, failure_index - config.failure_window_events)
    context = list(units[context_start:failure_index])
    tail = list(units[failure_index:end])
    budget = max(1, config.max_tokens - header_tokens)

    def tokens(seq: Sequence[AtomicUnit]) -> int:
        return estimate_tokens("\n\n".join(unit.text for unit in seq), config.token_multiplier)

    combined = context + tail
    if tokens(combined) <= budget:
        return [(combined, any(unit.truncated for unit in combined))]

    tail_tokens = tokens(tail)
    if context and tail_tokens < budget:
        kept: list[AtomicUnit] = []
        used = tail_tokens
        for unit in reversed(context):
            unit_tokens = estimate_tokens(unit.text, config.token_multiplier)
            if used + unit_tokens > budget:
                break
            kept.insert(0, unit)
            used += unit_tokens
        if kept:
            return _pack_units(
                kept + tail,
                target_tokens=budget,
                max_tokens=budget,
                overlap_units=0,
                multiplier=config.token_multiplier,
            )
    return _pack_units(
        tail,
        target_tokens=budget,
        max_tokens=budget,
        overlap_units=0,
        multiplier=config.token_multiplier,
    )


def _validation_chunk(
    memory: ExperienceMemory,
    validation: ValidationResult,
    run_id: str | None,
    config: ChunkingSettings,
    created_at: datetime,
) -> MemoryChunk:
    checks = json.dumps(validation.checks, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    lines = [
        f"Validation: {validation.status}",
        f"Validator: {validation.validator_name} v{validation.validator_version}",
    ]
    if validation.observed_outcome:
        lines.append(f"Observed outcome: {validation.observed_outcome}")
    if checks and checks != "[]":
        lines.append(f"Checks: {checks}")
    if validation.detail:
        lines.append(f"Detail: {validation.detail}")
    if validation.error:
        lines.append(f"Error: {validation.error}")
    body = "\n".join(lines)
    return _finalize(
        memory=memory,
        chunk_type=ChunkType.VALIDATION_EVIDENCE,
        seq_start=0,
        seq_end=0,
        event_ids=(),
        body=body,
        run_id=run_id,
        header=_header(memory, ChunkType.VALIDATION_EVIDENCE, ()),
        exact_ids=(validation.id, validation.candidate_id),
        config=config,
        truncated=False,
        created_at=created_at,
    )


def _naive_chunks(
    memory: ExperienceMemory,
    units: Sequence[AtomicUnit],
    run_id: str | None,
    config: ChunkingSettings,
    created_at: datetime,
) -> list[MemoryChunk]:
    """Fixed-size baseline that intentionally ignores event boundaries."""

    if not units:
        return []
    rendered: list[tuple[AtomicUnit, str]] = [(unit, unit.text) for unit in units]
    joined = "\n\n".join(text for _, text in rendered)
    window_chars = max(200, int(config.target_tokens / config.token_multiplier * 4.2))
    overlap_chars = max(0, config.overlap_units) * 200
    chunks: list[MemoryChunk] = []
    start = 0
    while start < len(joined):
        end = min(len(joined), start + window_chars)
        text = joined[start:end]
        included = [
            unit
            for unit, unit_text in rendered
            if text.find(unit_text[:80]) >= 0 or unit_text[:80] in text
        ]
        event_ids = [event_id for unit in included for event_id in unit.event_ids]
        seqs = [unit.seq_start for unit in included] + [unit.seq_end for unit in included]
        chunks.append(
            _finalize(
                memory=memory,
                chunk_type=ChunkType.NAIVE_FIXED,
                seq_start=min(seqs) if seqs else 0,
                seq_end=max(seqs) if seqs else 0,
                event_ids=event_ids,
                body=text,
                run_id=run_id,
                header=_header(memory, ChunkType.NAIVE_FIXED, included),
                exact_ids=(),
                config=config,
                # A character slice is *supposed* to cut events; units whose
                # own rendering was bounded carry that forward. Set
                # ``truncated`` unconditionally and it stops meaning anything.
                truncated=any(unit.truncated for unit in included),
                created_at=created_at,
            )
        )
        if end >= len(joined):
            break
        start = max(start + 1, end - overlap_chars)
    return chunks


def build_chunks(
    memory: ExperienceMemory,
    *,
    events: Sequence[RunEvent] = (),
    validation: ValidationResult | None = None,
    run_id: str | None = None,
    config: ChunkingSettings | None = None,
    created_at: datetime | None = None,
) -> ChunkBuildResult:
    """Build chunks for one memory and (optionally) one backing run.

    The function is pure. Events must belong to one run; the caller decides
    which evidence run to materialize. ``card_only`` is the fair card baseline.
    ``naive_fixed`` exists only for evaluation and is labeled in the chunk type.

    ``created_at`` is accepted so callers (and tests) can pass a fixed clock;
    otherwise a single timestamp is used for every chunk in the build.
    """

    settings = config or ChunkingSettings()
    if run_id is None and events:
        run_id = events[0].run_id
    built_at = created_at or utc_now()
    included = set(settings.include_types)
    warnings: list[str] = []
    chunks: list[MemoryChunk] = []

    if settings.strategy == "card_only" or "memory_card" in included:
        chunks.append(_card_chunk(memory, settings, built_at))
    if settings.strategy == "card_only":
        return ChunkBuildResult(chunks=tuple(chunks))

    units = _pair_units(events)
    if settings.strategy == "naive_fixed":
        naive = _naive_chunks(memory, units, run_id, settings, built_at)
        if len(naive) > settings.max_chunks_per_memory:
            warnings.append(
                f"naive_fixed chunks truncated from {len(naive)} to "
                f"{settings.max_chunks_per_memory}"
            )
            naive = naive[: settings.max_chunks_per_memory]
        return ChunkBuildResult(
            chunks=tuple(chunks + naive),
            warnings=tuple(warnings),
            truncated=bool(warnings),
        )

    typed: list[MemoryChunk] = []
    # A failed tool call is paired with its request, so ``seq_end`` is the
    # failed event itself; ``seq_start`` would point at the request instead.
    failure_anchor = next((unit.seq_end for unit in units if unit.failed), 0)
    if "failure_window" in included:
        failure_header = _header(memory, ChunkType.FAILURE_WINDOW, units)
        for packed, truncated in _failure_window(
            units,
            settings,
            header_tokens=estimate_tokens(failure_header, settings.token_multiplier),
        ):
            if not packed:
                continue
            body = "\n\n".join(unit.text for unit in packed)
            event_ids = [event_id for unit in packed for event_id in unit.event_ids]
            exact = [value for unit in packed for value in unit.exact_ids]
            seq_start = min(unit.seq_start for unit in packed)
            seq_end = max(unit.seq_end for unit in packed)
            typed.append(
                _finalize(
                    memory=memory,
                    chunk_type=ChunkType.FAILURE_WINDOW,
                    # ``seq_start`` is the first event actually present in the
                    # text (context can precede the failure). ``anchor_seq``
                    # carries the failure position separately, but only on the
                    # pack that actually contains it: a long tail can span
                    # several continuation chunks, and those have no failure
                    # event of their own.
                    seq_start=seq_start,
                    seq_end=seq_end,
                    anchor_seq=(failure_anchor if seq_start <= failure_anchor <= seq_end else 0),
                    event_ids=event_ids,
                    body=body,
                    run_id=run_id,
                    header=failure_header,
                    exact_ids=exact,
                    config=settings,
                    truncated=truncated,
                    failure_code=next(
                        (unit.failure_code for unit in packed if unit.failure_code), None
                    ),
                    created_at=built_at,
                )
            )
    if validation is not None and "validation_evidence" in included:
        typed.append(_validation_chunk(memory, validation, run_id, settings, built_at))
    if "action_observation" in included:
        action_units = [unit for unit in units if unit.tool_name]
        action_header = _header(memory, ChunkType.ACTION_OBSERVATION, action_units)
        for packed, truncated in _pack_units(
            action_units,
            target_tokens=settings.target_tokens,
            max_tokens=settings.max_tokens,
            overlap_units=settings.overlap_units,
            multiplier=settings.token_multiplier,
            overhead_tokens=estimate_tokens(action_header, settings.token_multiplier),
        ):
            if not packed:
                continue
            body = "\n\n".join(unit.text for unit in packed)
            typed.append(
                _finalize(
                    memory=memory,
                    chunk_type=ChunkType.ACTION_OBSERVATION,
                    seq_start=min(unit.seq_start for unit in packed),
                    seq_end=max(unit.seq_end for unit in packed),
                    event_ids=[event_id for unit in packed for event_id in unit.event_ids],
                    body=body,
                    run_id=run_id,
                    header=action_header,
                    exact_ids=[value for unit in packed for value in unit.exact_ids],
                    config=settings,
                    truncated=truncated,
                    failure_code=next(
                        (unit.failure_code for unit in packed if unit.failure_code), None
                    ),
                    created_at=built_at,
                )
            )

    priority = {
        ChunkType.FAILURE_WINDOW: 0,
        ChunkType.VALIDATION_EVIDENCE: 1,
        ChunkType.ACTION_OBSERVATION: 2,
    }
    typed.sort(key=lambda chunk: (priority.get(chunk.chunk_type, 9), chunk.seq_start))
    remaining = settings.max_chunks_per_memory - len(chunks)
    if remaining <= 0:
        warnings.append("no room for evidence chunks after the memory card")
        typed = []
    elif len(typed) > remaining:
        warnings.append(f"evidence chunks truncated from {len(typed)} to {remaining}")
        typed = typed[:remaining]
    chunks.extend(typed)
    return ChunkBuildResult(
        chunks=tuple(chunks),
        warnings=tuple(warnings),
        truncated=any(chunk.truncated for chunk in chunks),
    )
