"""Promotion policy and lifecycle for compiled repairs.

A candidate becomes an active memory only through this single service, which
enforces every promotion requirement from plan section 8.2 / 11.8.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from recap.config import Settings
from recap.domain.enums import (
    CreatedFrom,
    MemoryStatus,
    RepairCandidateStatus,
    ValidationStatus,
)
from recap.domain.memories import ExperienceMemory, MemoryContent
from recap.domain.repairs import PromotionDecision, RepairCandidate
from recap.errors import PromotionRejected
from recap.logging import get_logger
from recap.recorder.redaction import contains_secret

logger = get_logger(__name__)


class PromotionService:
    def __init__(
        self,
        *,
        settings: Settings,
        memory_repository: Any,
        repair_repository: Any,
        audit_repository: Any,
        index_sync: Any,
    ) -> None:
        self.settings = settings
        self.memories = memory_repository
        self.repairs = repair_repository
        self.audit = audit_repository
        self.index_sync = index_sync

    def evaluate(self, candidate: RepairCandidate) -> PromotionDecision:
        checks: list[dict[str, Any]] = []

        def check(name: str, passed: bool, detail: str = "") -> None:
            checks.append({"check": name, "passed": passed, "detail": detail})

        validations = self.repairs.list_validations(candidate.id)
        passed_validation = any(v.status is ValidationStatus.PASSED for v in validations)
        check("validation_passed", passed_validation, f"{len(validations)} validation record(s)")
        check(
            "evidence_present",
            bool(candidate.diagnosis.evidence_event_ids),
            "evidence event ids required",
        )
        check("preconditions_present", bool(candidate.preconditions))
        check("postconditions_present", bool(candidate.postconditions))
        check("environment_recorded", bool(candidate.environment_fingerprint))
        check(
            "no_secret_leakage", not contains_secret(candidate.model_dump_json()), "candidate payload"
        )
        check(
            "no_destructive_actions",
            all(a.side_effect_class.value != "destructive" for a in candidate.strategy.actions),
        )
        duplicate = self.memories.find_active_by_content_hash(
            candidate.workspace_id, candidate.content_hash or candidate.compute_hash()
        )
        check(
            "no_duplicate_active_memory",
            duplicate is None,
            f"existing memory {duplicate.id}" if duplicate else "no duplicate",
        )
        if self.settings.repair_compiler.require_validation_for_promotion:
            check("validation_required", passed_validation)
        approved = all(item["passed"] for item in checks)
        failed = [item["check"] for item in checks if not item["passed"]]
        return PromotionDecision(
            approved=approved,
            reason="all promotion checks passed" if approved else "failed checks: " + ", ".join(failed),
            checks=checks,
        )

    def promote(
        self,
        candidate: RepairCandidate,
        *,
        reviewer: str = "system",
        force_review_approval: bool = False,
    ) -> ExperienceMemory:
        decision = self.evaluate(candidate)
        if not decision.approved:
            raise PromotionRejected(
                "candidate does not satisfy promotion policy",
                details={"reason": decision.reason, "checks": decision.checks},
            )
        content = self._build_content(candidate)
        now = datetime.now(UTC)
        memory = ExperienceMemory(
            workspace_id=candidate.workspace_id,
            content=content,
            status=MemoryStatus.ACTIVE,
            status_reason="promoted from validated repair candidate",
            created_from=CreatedFrom.REPAIR_COMPILER,
            risk_level=candidate.risk_level,
            evidence_run_ids=[candidate.source_run_id],
            validation_record_ids=[v.id for v in self.repairs.list_validations(candidate.id)],
            environment_id=candidate.environment_id,
            environment_fingerprint=candidate.environment_fingerprint,
            embedding_model_id=self.settings.embeddings.model_id,
            embedding_model_version=self.settings.embeddings.model_version,
            embedding_dimension=self.settings.embeddings.dimension,
            index_generation=self.settings.elasticsearch.index_generation,
            activated_at=now,
            last_validated_at=now,
        )
        created = self.memories.create_memory(memory, index_status="pending")
        candidate.status = RepairCandidateStatus.PROMOTED
        candidate.promotion_decision = "approved"
        candidate.promotion_reason = decision.reason
        candidate.promoted_memory_id = created.id
        candidate.updated_at = now
        self.repairs.update(candidate)
        self.index_sync(created)
        from recap.domain.feedback import AuditEvent

        self.audit.append(
            AuditEvent(
                workspace_id=candidate.workspace_id,
                entity_type="memory",
                entity_id=created.id,
                action="promoted",
                reason=decision.reason,
                actor=reviewer,
                after={"candidate_id": candidate.id, "status": str(created.status)},
            )
        )
        return created

    def _build_content(self, candidate: RepairCandidate) -> MemoryContent:
        window = candidate.failure_window
        return MemoryContent(
            title=f"Recovery: {window.failure_category.replace('_', ' ')}",
            summary=candidate.strategy.summary,
            failure_signature=window.normalized_signature,
            failure_code=window.error_code,
            task_families=[candidate.task_family] if candidate.task_family else [],
            tool_names=[window.tool_name] if window.tool_name else [],
            tool_versions={},
            tool_schema_fingerprints=self._schema_fingerprints(candidate),
            environment_constraints={},
            environment_type="mock_api",
            preconditions=candidate.preconditions,
            strategy=candidate.strategy,
            postconditions=candidate.postconditions,
            known_limitations=candidate.strategy.limitations,
            counterexamples=[],
        )

    def _schema_fingerprints(self, candidate: RepairCandidate) -> dict[str, str]:
        fingerprints: dict[str, str] = {}
        for predicate in candidate.preconditions:
            prefix = "tool_schema_fingerprints."
            if predicate.field.startswith(prefix) and isinstance(predicate.value, str):
                fingerprints[predicate.field[len(prefix) :]] = predicate.value
        return fingerprints
