"""MemoryShift: compatibility, reliability, drift, and revalidation.

Simple and explainable first: fingerprint comparison, contextual outcome
statistics, minimum-sample safeguards, and explicit reasons.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from recap.config import Settings
from recap.domain.enums import (
    CompatibilityStatus,
    MemoryStatus,
    OutcomeStatus,
    RevalidationReason,
    RevalidationState,
)
from recap.domain.feedback import (
    AuditEvent,
    CompatibilityResult,
    OutcomeEvidence,
    RevalidationJob,
)
from recap.domain.memories import ExperienceMemory
from recap.domain.runs import new_id
from recap.errors import NotFoundError
from recap.logging import get_logger

logger = get_logger(__name__)


class MemoryShiftService:
    def __init__(
        self,
        *,
        settings: Settings,
        memory_repository: Any,
        outcome_repository: Any,
        compatibility_repository: Any,
        revalidation_repository: Any,
        audit_repository: Any,
        index_sync: Any | None = None,
    ) -> None:
        self.settings = settings
        self.memories = memory_repository
        self.outcomes = outcome_repository
        self.compatibility = compatibility_repository
        self.revalidation = revalidation_repository
        self.audit = audit_repository
        self.index_sync = index_sync

    # ------------------------------------------------------- compatibility
    def evaluate_compatibility(
        self,
        memory: ExperienceMemory,
        *,
        environment_id: str,
        environment_fingerprint: str,
        tool_schema_fingerprints: dict[str, str] | None = None,
        tool_versions: dict[str, str] | None = None,
    ) -> CompatibilityResult:
        reasons: list[str] = []
        status = CompatibilityStatus.COMPATIBLE

        if not memory.environment_fingerprint or not environment_fingerprint:
            status = CompatibilityStatus.UNKNOWN
            reasons.append("missing_environment_fingerprint")
        elif memory.environment_fingerprint == environment_fingerprint:
            status = CompatibilityStatus.COMPATIBLE
            reasons.append("environment_fingerprint_match")
        else:
            schema_conflict = False
            unknown_schema = False
            declared = memory.content.tool_schema_fingerprints or {}
            observed = tool_schema_fingerprints or {}
            for tool, expected in declared.items():
                actual = observed.get(tool)
                if actual is None:
                    unknown_schema = True
                elif actual != expected:
                    schema_conflict = True
                    reasons.append(f"tool_schema_mismatch:{tool}")
            if schema_conflict:
                status = CompatibilityStatus.INCOMPATIBLE
            else:
                status = CompatibilityStatus.SUSPECT
                reasons.append("environment_fingerprint_changed")
                if unknown_schema:
                    reasons.append("tool_schema_unverified")

        if memory.status is MemoryStatus.QUARANTINED:
            status = CompatibilityStatus.INCOMPATIBLE
            reasons.append("memory_quarantined")
        elif memory.status is MemoryStatus.RETIRED:
            status = CompatibilityStatus.INCOMPATIBLE
            reasons.append("memory_retired")

        result = CompatibilityResult(
            memory_id=memory.id,
            memory_version=memory.memory_version,
            environment_id=environment_id,
            environment_fingerprint=environment_fingerprint,
            status=status,
            reasons=reasons,
            evidence={
                "workspace_id": memory.workspace_id,
                "memory_fingerprint": memory.environment_fingerprint,
                "observed_fingerprint": environment_fingerprint,
                "tool_schema_fingerprints": tool_schema_fingerprints or {},
                "tool_versions": tool_versions or {},
            },
        )
        self.compatibility.upsert(result)
        return result

    # ------------------------------------------------------------ outcomes
    def record_outcome(
        self,
        memory: ExperienceMemory,
        *,
        outcome: OutcomeStatus,
        applied: bool,
        run_id: str | None = None,
        environment_id: str = "",
        environment_fingerprint: str = "",
        task_family: str = "",
        severity: str = "normal",
        notes: str = "",
        evidence: dict[str, Any] | None = None,
    ) -> OutcomeEvidence:
        if environment_fingerprint and memory.environment_fingerprint and (
            environment_fingerprint != memory.environment_fingerprint
        ):
            self.audit.append(
                AuditEvent(
                    workspace_id=memory.workspace_id,
                    entity_type="memory",
                    entity_id=memory.id,
                    action="outcome_recorded_in_foreign_context",
                    reason="outcome environment differs from memory origin",
                    after={"outcome": str(outcome)},
                )
            )
        record = OutcomeEvidence(
            memory_id=memory.id,
            memory_version=memory.memory_version,
            run_id=run_id,
            workspace_id=memory.workspace_id,
            outcome=outcome,
            applied=applied,
            environment_id=environment_id,
            environment_fingerprint=environment_fingerprint,
            task_family=task_family,
            severity=severity,
            notes=notes,
            evidence=evidence or {},
        )
        self.outcomes.append(record)

        # Only verified outcomes update reliability counters.
        stats = memory.stats
        if outcome is OutcomeStatus.VERIFIED_SUCCESS:
            stats.verified_successes += 1
            stats.consecutive_failures = 0
        elif outcome is OutcomeStatus.VERIFIED_FAILURE:
            stats.verified_failures += 1
            stats.consecutive_failures += 1
        elif outcome is OutcomeStatus.PARTIAL:
            stats.partial += 1
        else:
            stats.unknown += 1
        if applied:
            stats.applied += 1
        stats.last_outcome_at = datetime.now(UTC)
        self.memories.update_stats(memory)

        self._maybe_degrade(memory, outcome=outcome, severity=severity)
        return record

    def _maybe_degrade(
        self, memory: ExperienceMemory, *, outcome: OutcomeStatus, severity: str
    ) -> None:
        if memory.status is not MemoryStatus.ACTIVE:
            return
        stats = memory.stats
        observations = stats.verified_successes + stats.verified_failures
        mean, _ = stats.posterior(
            self.settings.reliability.beta_prior_alpha,
            self.settings.reliability.beta_prior_beta,
        )
        config = self.settings.reliability
        reason: str | None = None
        target = MemoryStatus.SUSPECT
        if (
            severity == "high"
            and outcome is OutcomeStatus.VERIFIED_FAILURE
            and self.settings.reliability.quarantine_posterior_mean_threshold >= mean
        ):
            target = MemoryStatus.QUARANTINED
            reason = "high_severity_verified_failure"
        elif (
            observations >= config.minimum_verified_outcomes_for_alert
            and mean < config.suspect_posterior_mean_threshold
        ):
            reason = f"posterior_mean_below_threshold:{mean:.3f}"
        elif stats.consecutive_failures >= config.consecutive_failures_for_suspect:
            reason = f"consecutive_verified_failures:{stats.consecutive_failures}"
        if reason is None:
            return
        transitioned = memory.transition(target, reason)
        self.memories.update_status(transitioned)
        if self.index_sync is not None:
            self.index_sync(transitioned)
        self.audit.append(
            AuditEvent(
                workspace_id=memory.workspace_id,
                entity_type="memory",
                entity_id=memory.id,
                action=f"transition:{memory.status}->{target}",
                reason=reason,
                before={"status": str(memory.status)},
                after={"status": str(target)},
            )
        )
        if self.settings.memory_shift.revalidation_enabled:
            self.enqueue_revalidation(
                transitioned,
                reason=RevalidationReason.RELIABILITY_DROP,
                environment_id=memory.environment_id,
                environment_fingerprint=memory.environment_fingerprint,
            )

    # -------------------------------------------------------- revalidation
    def enqueue_revalidation(
        self,
        memory: ExperienceMemory,
        *,
        reason: RevalidationReason,
        environment_id: str,
        environment_fingerprint: str = "",
        validation_plan: str = "mock_sandbox",
        priority: int = 5,
    ) -> RevalidationJob:
        dedupe_key = f"{memory.id}:v{memory.memory_version}:{environment_fingerprint or environment_id}:{validation_plan}"
        job = RevalidationJob(
            memory_id=memory.id,
            memory_version=memory.memory_version,
            workspace_id=memory.workspace_id,
            reason=reason,
            target_environment_id=environment_id,
            target_environment_fingerprint=environment_fingerprint,
            validation_plan=validation_plan,
            priority=priority,
            dedupe_key=dedupe_key,
        )
        return self.revalidation.enqueue(job)

    def run_revalidation_job(self, job_id: str, validator: Any) -> RevalidationJob:
        job = self.revalidation.get(job_id)
        if job is None:
            raise NotFoundError(f"revalidation job {job_id} not found")
        job.state = RevalidationState.RUNNING
        job.attempts += 1
        self.revalidation.update(job)
        try:
            memory = self.memories.get_memory(job.memory_id, job.workspace_id)
            if memory is None:
                raise NotFoundError(f"memory {job.memory_id} not found")
            result = validator.revalidate(memory, environment_id=job.target_environment_id)
            job.result = result
            job.state = (
                RevalidationState.PASSED
                if result.get("status") == "passed"
                else RevalidationState.FAILED
            )
            if job.state is RevalidationState.PASSED and memory.status is MemoryStatus.SUSPECT:
                restored = memory.transition(MemoryStatus.ACTIVE, "revalidation passed")
                self.memories.update_status(restored)
                if self.index_sync is not None:
                    self.index_sync(restored)
        except Exception as exc:
            job.state = RevalidationState.INCONCLUSIVE
            job.error = str(exc)
        self.revalidation.update(job)
        return job

    def compatibility_report(self, memory_id: str, workspace_id: str) -> list[CompatibilityResult]:
        memory = self.memories.get_memory(memory_id, workspace_id)
        if memory is None:
            raise NotFoundError(f"memory {memory_id} not found")
        return self.compatibility.list_for_memory(memory_id)
