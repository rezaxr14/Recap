"""RunRecorder: validate, redact, and persist runs/events.

Event ingestion is idempotent by event id and by (run_id, sequence_number).
Sequence numbers must be strictly increasing; duplicates are safe no-ops only
when the payload matches.
"""

from __future__ import annotations

import hashlib
from typing import Any, Sequence

from recap.domain.environments import canonical_json
from recap.domain.runs import AgentRun, RunEvent, RunEventCreate, new_id, utc_now
from recap.errors import ConflictError, NotFoundError, ValidationError
from recap.logging import get_logger
from recap.recorder.normalization import normalize_signature
from recap.recorder.redaction import redact_payload

logger = get_logger(__name__)

MAX_SEQUENCE_GAP = 10_000


def payload_hash(payload: dict[str, Any]) -> str:
    return f"sha256:{hashlib.sha256(canonical_json(payload).encode()).hexdigest()}"


class RunRecorder:
    """Domain service for run recording. Storage-agnostic.

    The repository must provide:
      - create_run(run) -> AgentRun
      - get_run(run_id) -> AgentRun | None
      - update_run(run) -> AgentRun
      - append_event(event) -> bool (False if duplicate idempotent no-op)
      - list_events(run_id) -> list[RunEvent]
    """

    def __init__(self, repository: Any) -> None:
        self.repository = repository

    def start_run(self, run: AgentRun) -> AgentRun:
        if not run.workspace_id:
            raise ValidationError("workspace_id is required")
        return self.repository.create_run(run)

    def append_event(self, run: AgentRun, event: RunEventCreate) -> RunEvent:
        if run.status.value not in {"running"}:
            raise ConflictError(
                "cannot append events to a finished run", details={"status": run.status.value}
            )
        existing = self.repository.list_events(run.id)
        last_sequence = existing[-1].sequence_number if existing else 0
        if event.sequence_number <= last_sequence:
            duplicate = next(
                (
                    item
                    for item in existing
                    if item.sequence_number == event.sequence_number
                    or (event.id is not None and item.id == event.id)
                ),
                None,
            )
            if duplicate is not None:
                redacted, _ = redact_payload(event.payload)
                if payload_hash(redacted) == duplicate.payload_hash:
                    return duplicate
                raise ConflictError(
                    "sequence number already used with a different payload",
                    details={"sequence_number": event.sequence_number},
                )
            raise ValidationError(
                "sequence numbers must be strictly increasing",
                details={"sequence_number": event.sequence_number, "last_sequence": last_sequence},
            )
        if event.sequence_number - last_sequence > MAX_SEQUENCE_GAP:
            raise ValidationError("sequence number jump too large")

        redacted, applied = redact_payload(event.payload)
        from recap.domain.runs import new_id

        record = RunEvent(
            id=event.id or new_id("evt"),
            run_id=run.id,
            sequence_number=event.sequence_number,
            event_type=event.event_type,
            timestamp=event.timestamp or __import__("recap.domain.runs", fromlist=["utc_now"]).utc_now(),
            tool_name=event.tool_name,
            tool_version=event.tool_version,
            correlation_id=event.correlation_id,
            payload_redacted=redacted,
            payload_hash=payload_hash(redacted),
            redaction_applied=applied,
        )
        inserted = self.repository.append_event(record)
        if not inserted:
            # Idempotent replay of an already-persisted event id.
            return record
        return record

    def finish_run(self, run: AgentRun, **fields: Any) -> AgentRun:
        from recap.domain.runs import utc_now

        finished = run.model_copy(deep=True)
        for key, value in fields.items():
            if value is not None and hasattr(finished, key):
                setattr(finished, key, value)
        finished.finished_at = utc_now()
        finished.updated_at = utc_now()
        return self.repository.update_run(finished)

    def build_failure_context(
        self, run: AgentRun, events: Sequence[RunEvent]
    ) -> dict[str, Any]:
        """Derive a redacted failure context used by retrieval and repair."""
        failed = [event for event in events if event.event_type.value == "tool_call_failed"]
        last = failed[-1] if failed else None
        message = ""
        error_type = ""
        error_code = ""
        tool_name = ""
        if last is not None:
            payload = last.payload_redacted
            message = str(payload.get("error_message") or payload.get("message") or "")
            error_type = str(payload.get("error_type") or payload.get("exception") or "")
            error_code = str(payload.get("error_code") or payload.get("code") or "")
            tool_name = last.tool_name or ""
        signature = normalize_signature(
            message=message, error_type=error_type, error_code=error_code, tool_name=tool_name
        )
        return {
            "run_id": run.id,
            "task_id": run.task_id,
            "task_family": run.task_family,
            "environment_id": run.environment_id,
            "environment_fingerprint": run.environment_fingerprint,
            "error_type": error_type,
            "error_code": error_code,
            "error_message": message,
            "tool_name": tool_name,
            "failure_signature": signature,
            "failure_type": run.failure_type or "",
        }

    def get_run(self, run_id: str) -> AgentRun:
        run = self.repository.get_run(run_id)
        if run is None:
            raise NotFoundError(f"run {run_id} not found")
        return run
