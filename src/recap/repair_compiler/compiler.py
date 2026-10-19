"""RepairCompiler orchestration.

Deterministic failure window -> diagnosis hypothesis -> typed candidate ->
validation -> promotion. The compiler may conclude ``no_repair_candidate``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from recap.config import Settings
from recap.domain.enums import RepairCandidateStatus, ValidationStatus
from recap.domain.environments import EnvironmentFingerprint
from recap.domain.repairs import RepairCandidate
from recap.domain.runs import AgentRun
from recap.errors import NotFoundError, ValidationError
from recap.logging import get_logger
from recap.repair_compiler.candidate_generation import (
    CandidateGenerator,
    DeterministicGenerator,
    build_candidate,
    validate_draft,
)
from recap.repair_compiler.failure_windows import extract_failure_window
from recap.repair_compiler.validation import ValidatorRegistry

logger = get_logger(__name__)


class RepairCompiler:
    def __init__(
        self,
        *,
        settings: Settings,
        run_repository: Any,
        repair_repository: Any,
        validators: ValidatorRegistry,
        generator: CandidateGenerator | None = None,
    ) -> None:
        self.settings = settings
        self.runs = run_repository
        self.repairs = repair_repository
        self.validators = validators
        self.generator = generator or DeterministicGenerator()

    def compile(
        self,
        run_id: str,
        *,
        environment: EnvironmentFingerprint,
    ) -> list[RepairCandidate]:
        run = self.runs.get_run(run_id)
        if run is None:
            raise NotFoundError(f"run {run_id} not found")
        events = self.runs.list_events(run_id)
        window = extract_failure_window(
            run_id=run_id,
            events=events,
            max_events=self.settings.repair_compiler.max_failure_window_events,
        )
        if window is None:
            logger.info("no failure window found", extra={"run_id": run_id})
            return []
        environment_info = {
            "environment_id": environment.environment_id,
            "environment_version": environment.environment_version,
            "tool_schema_hash": next(iter(environment.tool_schema_hashes.values()), ""),
            "tool_schema_hashes": environment.tool_schema_hashes,
        }
        drafts = self.generator.generate(window=window, environment=environment_info)
        allowed_event_ids = {event.id for event in events}
        candidates: list[RepairCandidate] = []
        for draft in drafts:
            try:
                validate_draft(
                    draft,
                    window=window,
                    settings=self.settings.repair_compiler,
                    allowed_event_ids=allowed_event_ids,
                )
            except ValidationError as exc:
                logger.warning("rejected candidate draft: %s", exc.message)
                continue
            candidate = build_candidate(
                draft,
                window=window,
                workspace_id=run.workspace_id,
                environment_id=environment.environment_id,
                environment_fingerprint=environment.fingerprint,
                generated_by=getattr(self.generator, "name", "unknown"),
                task_family=run.task_family,
            )
            existing = self.repairs.find_by_hash(candidate.content_hash, run.workspace_id)
            if existing is not None:
                candidates.append(existing)
                continue
            candidates.append(self.repairs.create(candidate))
        return candidates

    def validate(
        self,
        candidate_id: str,
        *,
        environment: EnvironmentFingerprint,
        validator_name: str = "static",
        attempt: int = 1,
    ) -> Any:
        candidate = self.repairs.get(candidate_id)
        if candidate is None:
            raise NotFoundError(f"candidate {candidate_id} not found")
        if candidate.status is RepairCandidateStatus.PROMOTED:
            raise ValidationError("promoted candidates are immutable")
        validator = self.validators.get(validator_name)
        candidate.status = RepairCandidateStatus.VALIDATING
        candidate.updated_at = datetime.now(UTC)
        self.repairs.update(candidate)
        result = validator.validate(candidate, environment=environment, attempt=attempt)
        self.repairs.add_validation(result)
        candidate.validation_status = result.status
        candidate.status = (
            RepairCandidateStatus.VALIDATED
            if result.status is ValidationStatus.PASSED
            else RepairCandidateStatus.REJECTED
        )
        candidate.updated_at = datetime.now(UTC)
        self.repairs.update(candidate)
        return result
