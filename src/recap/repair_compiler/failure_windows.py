"""Deterministic failure-window extraction.

The first pass is rule-based and never assumes the last tool call caused the
failure. It locates the terminal error, the nearest failing tool pair, and a
bounded context window.
"""

from __future__ import annotations

from recap.domain.enums import EventType
from recap.domain.repairs import FailureWindow
from recap.domain.runs import RunEvent
from recap.recorder.normalization import normalize_signature


def _as_dict(event: RunEvent) -> dict[str, object]:
    return event.payload_redacted or {}


def extract_failure_window(
    *,
    run_id: str,
    events: list[RunEvent],
    max_events: int = 20,
) -> FailureWindow | None:
    if not events:
        return None
    failures = [
        event
        for event in events
        if event.event_type in {EventType.TOOL_CALL_FAILED, EventType.POSTCONDITION_CHECKED}
        and (
            event.event_type is EventType.TOOL_CALL_FAILED
            or not bool(_as_dict(event).get("passed", True))
        )
    ]
    if not failures:
        return None
    # Prefer the earliest distinct failing signature to avoid anchoring on a retry.
    anchor = failures[0]
    payload = _as_dict(anchor)
    message = str(
        payload.get("error_message") or payload.get("message") or payload.get("detail") or ""
    )
    error_type = str(payload.get("error_type") or payload.get("exception") or "")
    error_code = str(payload.get("error_code") or payload.get("code") or "")
    tool_name = anchor.tool_name or ""
    signature = normalize_signature(
        message=message, error_type=error_type, error_code=error_code, tool_name=tool_name
    )

    anchor_index = events.index(anchor)
    start_index = max(0, anchor_index - max_events // 2)
    end_index = min(len(events) - 1, anchor_index + max_events // 2)
    window = events[start_index : end_index + 1]
    retry_count = (
        sum(
            1
            for event in events
            if event.event_type is EventType.TOOL_CALL_FAILED
            and normalize_signature(
                message=str(
                    _as_dict(event).get("error_message") or _as_dict(event).get("message") or ""
                ),
                error_type=str(_as_dict(event).get("error_type") or ""),
                error_code=str(_as_dict(event).get("error_code") or ""),
                tool_name=event.tool_name or "",
            )
            == signature
        )
        - 1
    )

    category = _categorize(error_code=error_code, error_type=error_type, message=message)
    notes: list[str] = []
    if not any(e.event_type is EventType.TASK_OBSERVATION for e in window):
        notes.append("no task observation inside failure window")
    if anchor_index == len(events) - 1:
        notes.append("failure event is last event; cause not assumed")
    return FailureWindow(
        run_id=run_id,
        start_sequence=window[0].sequence_number,
        end_sequence=window[-1].sequence_number,
        start_event_id=window[0].id,
        end_event_id=window[-1].id,
        failure_category=category,
        normalized_signature=signature,
        raw_signature=message[:1000],
        error_code=error_code,
        error_type=error_type,
        tool_name=tool_name,
        event_ids=[event.id for event in window],
        retry_count=max(retry_count, 0),
        incomplete=False,
        notes=notes,
    )


def _categorize(*, error_code: str, error_type: str, message: str) -> str:
    combined = f"{error_code} {error_type} {message}".lower()
    if "invalid_cursor" in combined or ("cursor" in combined and "invalid" in combined):
        return "invalid_pagination_cursor"
    if "rate_limit" in combined or "rate limit" in combined or "429" in combined:
        return "transient_rate_limit"
    if "unknown_parameter" in combined or "unexpected" in combined and "param" in combined:
        return "tool_parameter_mismatch"
    if "timeout" in combined or "timed out" in combined:
        return "transient_timeout"
    if "token_expired" in combined or "token expired" in combined:
        return "authentication_expired"
    if "idempotency" in combined or "request_id is required" in combined:
        return "missing_idempotency_key"
    if "missing_response_field" in combined or "returned under 'data'" in combined:
        return "response_shape_changed"
    if "missing_required_parameter" in combined or "required for this api version" in combined:
        return "missing_required_parameter"
    if "invalid_enum" in combined or "sort must be one of" in combined:
        return "enum_value_mismatch"
    if "invalid_date_format" in combined or "iso-8601" in combined or "yyyy/mm/dd" in combined:
        return "timestamp_format_mismatch"
    if "permission" in combined or "forbidden" in combined or "403" in combined:
        return "authorization_denied"
    if "not_found" in combined or "404" in combined:
        return "resource_not_found"
    if "schema" in combined or "validation" in combined:
        return "validation_error"
    return "unclassified_failure"
