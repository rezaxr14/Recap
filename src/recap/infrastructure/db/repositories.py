"""SQLAlchemy repository implementations.

These translate between domain models and ORM rows. All reads/writes are scoped
by workspace where the domain requires it.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Sequence

from sqlalchemy import delete, func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from recap.domain.enums import (
    CompatibilityStatus,
    MemoryStatus,
    OutcomeStatus,
    RepairCandidateStatus,
    RevalidationState,
    RunStatus,
    ValidationStatus,
)
from recap.domain.feedback import (
    AuditEvent,
    CompatibilityResult,
    OutcomeEvidence,
    RevalidationJob,
)
from recap.domain.memories import ExperienceMemory, MemoryContent, OutcomeStats
from recap.domain.repairs import RepairCandidate, ValidationResult
from recap.domain.retrieval import RetrievalDecision
from recap.domain.runs import AgentRun, RunEvent
from recap.errors import ConflictError, NotFoundError
from recap.infrastructure.db.models import (
    AuditEventRow,
    CompatibilityRow,
    MemoryRow,
    OutcomeRow,
    OutboxRow,
    RepairCandidateRow,
    RetrievalDecisionRow,
    RevalidationJobRow,
    RunEventRow,
    RunRow,
    ValidationRow,
    WorkspaceRow,
)
from recap.domain.runs import new_id, utc_now


# --------------------------------------------------------------------- mappers
def _run_to_domain(row: RunRow) -> AgentRun:
    return AgentRun(
        id=row.id,
        workspace_id=row.workspace_id,
        agent_id=row.agent_id,
        task_id=row.task_id,
        task_family=row.task_family,
        environment_id=row.environment_id,
        environment_fingerprint=row.environment_fingerprint,
        toolset_fingerprint=row.toolset_fingerprint,
        model_provider=row.model_provider,
        model_name=row.model_name,
        model_config_fingerprint=row.model_config_fingerprint,
        status=RunStatus(row.status),
        failure_type=row.failure_type,
        success_criteria=row.success_criteria,
        tool_calls_count=row.tool_calls_count,
        retry_count=row.retry_count,
        input_tokens=row.input_tokens,
        output_tokens=row.output_tokens,
        estimated_cost=row.estimated_cost,
        duration_ms=row.duration_ms,
        memory_snapshot_id=row.memory_snapshot_id,
        config_snapshot=row.config_snapshot or {},
        started_at=row.started_at,
        finished_at=row.finished_at,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


def _event_to_domain(row: RunEventRow) -> RunEvent:
    return RunEvent(
        id=row.id,
        run_id=row.run_id,
        sequence_number=row.sequence_number,
        event_type=row.event_type,
        timestamp=row.timestamp,
        payload_schema_version=row.payload_schema_version,
        tool_name=row.tool_name,
        tool_version=row.tool_version,
        correlation_id=row.correlation_id,
        payload_redacted=row.payload_redacted or {},
        payload_hash=row.payload_hash,
        redaction_applied=row.redaction_applied,
        redaction_rule_version=row.redaction_rule_version,
    )


def _memory_to_domain(row: MemoryRow) -> ExperienceMemory:
    return ExperienceMemory(
        id=row.id,
        workspace_id=row.workspace_id,
        schema_version=row.schema_version,
        memory_version=row.memory_version,
        supersedes_memory_id=row.supersedes_memory_id,
        family_id=row.family_id,
        content=MemoryContent.model_validate(row.content_json),
        status=MemoryStatus(row.status),
        status_reason=row.status_reason,
        created_from=row.created_from,
        risk_level=row.risk_level,
        access_scope=row.access_scope,
        evidence_trace_ids=row.evidence_trace_ids or [],
        evidence_run_ids=row.evidence_run_ids or [],
        validation_record_ids=row.validation_record_ids or [],
        environment_fingerprint=row.environment_fingerprint,
        environment_id=row.environment_id,
        embedding_model_id=row.embedding_model_id,
        embedding_model_version=row.embedding_model_version,
        embedding_dimension=row.embedding_dimension,
        retrieval_document_id=row.retrieval_document_id,
        index_generation=row.index_generation,
        content_hash=row.content_hash,
        stats=OutcomeStats.model_validate(row.stats_json or {}),
        created_at=row.created_at,
        updated_at=row.updated_at,
        activated_at=row.activated_at,
        last_validated_at=row.last_validated_at,
    )


def _memory_to_row(memory: ExperienceMemory, *, index_status: str = "pending") -> MemoryRow:
    return MemoryRow(
        id=memory.id,
        workspace_id=memory.workspace_id,
        family_id=memory.family_id,
        memory_version=memory.memory_version,
        supersedes_memory_id=memory.supersedes_memory_id,
        schema_version=memory.schema_version,
        status=str(memory.status),
        status_reason=memory.status_reason,
        created_from=str(memory.created_from),
        risk_level=str(memory.risk_level),
        access_scope=memory.access_scope,
        content_json=memory.content.model_dump(mode="json"),
        content_hash=memory.content_hash,
        environment_id=memory.environment_id,
        environment_fingerprint=memory.environment_fingerprint,
        embedding_model_id=memory.embedding_model_id,
        embedding_model_version=memory.embedding_model_version,
        embedding_dimension=memory.embedding_dimension,
        retrieval_document_id=memory.retrieval_document_id,
        index_generation=memory.index_generation,
        indexed_at=memory.last_validated_at if index_status == "indexed" else None,
        index_status=index_status,
        evidence_run_ids=memory.evidence_run_ids,
        evidence_trace_ids=memory.evidence_trace_ids,
        validation_record_ids=memory.validation_record_ids,
        stats_json=memory.stats.model_dump(mode="json"),
        created_at=memory.created_at,
        updated_at=memory.updated_at,
        activated_at=memory.activated_at,
        last_validated_at=memory.last_validated_at,
    )


# ------------------------------------------------------------------ run store
class SqlRunRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create_run(self, run: AgentRun) -> AgentRun:
        row = RunRow(
            id=run.id,
            workspace_id=run.workspace_id,
            agent_id=run.agent_id,
            task_id=run.task_id,
            task_family=run.task_family,
            environment_id=run.environment_id,
            environment_fingerprint=run.environment_fingerprint,
            toolset_fingerprint=run.toolset_fingerprint,
            model_provider=run.model_provider,
            model_name=run.model_name,
            model_config_fingerprint=run.model_config_fingerprint,
            status=str(run.status),
            failure_type=run.failure_type,
            success_criteria=run.success_criteria,
            tool_calls_count=run.tool_calls_count,
            retry_count=run.retry_count,
            input_tokens=run.input_tokens,
            output_tokens=run.output_tokens,
            estimated_cost=run.estimated_cost,
            duration_ms=run.duration_ms,
            memory_snapshot_id=run.memory_snapshot_id,
            config_snapshot=run.config_snapshot,
            started_at=run.started_at,
            finished_at=run.finished_at,
            created_at=run.created_at,
            updated_at=run.updated_at,
        )
        self.session.add(row)
        self.session.flush()
        return _run_to_domain(row)

    def get_run(self, run_id: str) -> AgentRun | None:
        row = self.session.get(RunRow, run_id)
        return _run_to_domain(row) if row else None

    def update_run(self, run: AgentRun) -> AgentRun:
        row = self.session.get(RunRow, run.id)
        if row is None:
            raise NotFoundError(f"run {run.id} not found")
        row.status = str(run.status)
        row.failure_type = run.failure_type
        row.tool_calls_count = run.tool_calls_count
        row.retry_count = run.retry_count
        row.input_tokens = run.input_tokens
        row.output_tokens = run.output_tokens
        row.estimated_cost = run.estimated_cost
        row.duration_ms = run.duration_ms
        row.finished_at = run.finished_at
        row.updated_at = run.updated_at
        row.memory_snapshot_id = run.memory_snapshot_id
        row.config_snapshot = run.config_snapshot
        self.session.flush()
        return _run_to_domain(row)

    def append_event(self, event: RunEvent) -> bool:
        existing = self.session.get(RunEventRow, event.id)
        if existing is not None:
            return False
        try:
            with self.session.begin_nested():
                self.session.add(
                    RunEventRow(
                        id=event.id,
                        run_id=event.run_id,
                        sequence_number=event.sequence_number,
                        event_type=str(event.event_type),
                        timestamp=event.timestamp,
                        payload_schema_version=event.payload_schema_version,
                        tool_name=event.tool_name,
                        tool_version=event.tool_version,
                        correlation_id=event.correlation_id,
                        payload_redacted=event.payload_redacted,
                        payload_hash=event.payload_hash,
                        redaction_applied=event.redaction_applied,
                        redaction_rule_version=event.redaction_rule_version,
                    )
                )
        except IntegrityError as exc:
            raise ConflictError(
                "event sequence conflict", details={"sequence_number": event.sequence_number}
            ) from exc
        self.session.flush()
        return True

    def list_events(self, run_id: str) -> list[RunEvent]:
        rows = self.session.scalars(
            select(RunEventRow)
            .where(RunEventRow.run_id == run_id)
            .order_by(RunEventRow.sequence_number)
        ).all()
        return [_event_to_domain(row) for row in rows]

    def list_runs(self, workspace_id: str, limit: int = 50) -> list[AgentRun]:
        rows = self.session.scalars(
            select(RunRow)
            .where(RunRow.workspace_id == workspace_id)
            .order_by(RunRow.created_at.desc())
            .limit(limit)
        ).all()
        return [_run_to_domain(row) for row in rows]


# --------------------------------------------------------------- memory store
class SqlMemoryRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create_memory(
        self, memory: ExperienceMemory, *, index_status: str = "pending"
    ) -> ExperienceMemory:
        existing = self.session.scalar(
            select(MemoryRow).where(
                MemoryRow.workspace_id == memory.workspace_id,
                MemoryRow.content_hash == memory.content_hash,
            )
        )
        if existing is not None:
            return _memory_to_domain(existing)
        row = _memory_to_row(memory, index_status=index_status)
        self.session.add(row)
        try:
            self.session.flush()
        except IntegrityError as exc:
            raise ConflictError("duplicate memory content hash") from exc
        return _memory_to_domain(row)

    def get_memory(
        self, memory_id: str, workspace_id: str | None = None
    ) -> ExperienceMemory | None:
        row = self.session.get(MemoryRow, memory_id)
        if row is None:
            return None
        if workspace_id is not None and row.workspace_id != workspace_id:
            return None
        return _memory_to_domain(row)

    def list_memories(
        self,
        workspace_id: str,
        *,
        status: MemoryStatus | None = None,
        task_family: str | None = None,
        limit: int = 100,
    ) -> list[ExperienceMemory]:
        stmt = select(MemoryRow).where(MemoryRow.workspace_id == workspace_id)
        if status is not None:
            stmt = stmt.where(MemoryRow.status == str(status))
        if task_family is not None:
            stmt = stmt.where(
                func.json_extract_path_text(MemoryRow.content_json, "task_families")
                .contains(task_family)
            )
        stmt = stmt.order_by(MemoryRow.created_at.desc()).limit(limit)
        return [_memory_to_domain(row) for row in self.session.scalars(stmt).all()]

    def list_versions(self, family_id: str, workspace_id: str) -> list[ExperienceMemory]:
        rows = self.session.scalars(
            select(MemoryRow)
            .where(MemoryRow.workspace_id == workspace_id, MemoryRow.family_id == family_id)
            .order_by(MemoryRow.memory_version)
        ).all()
        return [_memory_to_domain(row) for row in rows]

    def update_status(
        self, memory: ExperienceMemory, *, index_status: str | None = None
    ) -> ExperienceMemory:
        row = self.session.get(MemoryRow, memory.id)
        if row is None:
            raise NotFoundError(f"memory {memory.id} not found")
        row.status = str(memory.status)
        row.status_reason = memory.status_reason
        row.updated_at = memory.updated_at
        row.activated_at = memory.activated_at
        row.last_validated_at = memory.last_validated_at
        row.stats_json = memory.stats.model_dump(mode="json")
        if index_status is not None:
            row.index_status = index_status
        self.session.flush()
        return _memory_to_domain(row)

    def update_stats(self, memory: ExperienceMemory) -> None:
        row = self.session.get(MemoryRow, memory.id)
        if row is None:
            raise NotFoundError(f"memory {memory.id} not found")
        row.stats_json = memory.stats.model_dump(mode="json")
        row.updated_at = utc_now()
        self.session.flush()

    def mark_indexed(self, memory_id: str, generation: int, *, success: bool) -> None:
        row = self.session.get(MemoryRow, memory_id)
        if row is None:
            raise NotFoundError(f"memory {memory_id} not found")
        row.index_status = "indexed" if success else "failed"
        if success:
            row.index_generation = generation
            row.indexed_at = utc_now()
        self.session.flush()

    def find_active_by_content_hash(
        self, workspace_id: str, content_hash: str
    ) -> ExperienceMemory | None:
        row = self.session.scalar(
            select(MemoryRow).where(
                MemoryRow.workspace_id == workspace_id,
                MemoryRow.content_hash == content_hash,
                MemoryRow.status.in_([str(MemoryStatus.ACTIVE), str(MemoryStatus.SUSPECT)]),
            )
        )
        return _memory_to_domain(row) if row else None

    def delete_memory(self, memory_id: str, workspace_id: str) -> bool:
        row = self.session.get(MemoryRow, memory_id)
        if row is None or row.workspace_id != workspace_id:
            return False
        self.session.delete(row)
        self.session.flush()
        return True


# ------------------------------------------------------------- outcome store
class SqlOutcomeRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def append(self, evidence: OutcomeEvidence) -> OutcomeEvidence:
        row = OutcomeRow(
            id=evidence.id,
            memory_id=evidence.memory_id,
            memory_version=evidence.memory_version,
            workspace_id=evidence.workspace_id,
            run_id=evidence.run_id,
            outcome=str(evidence.outcome),
            applied=evidence.applied,
            selected=evidence.selected,
            environment_id=evidence.environment_id,
            environment_fingerprint=evidence.environment_fingerprint,
            task_family=evidence.task_family,
            severity=evidence.severity,
            notes=evidence.notes,
            evidence_json=evidence.evidence,
            recorded_at=evidence.recorded_at,
        )
        self.session.add(row)
        self.session.flush()
        return evidence

    def list_for_memory(self, memory_id: str, limit: int = 200) -> list[OutcomeEvidence]:
        rows = self.session.scalars(
            select(OutcomeRow)
            .where(OutcomeRow.memory_id == memory_id)
            .order_by(OutcomeRow.recorded_at.desc())
            .limit(limit)
        ).all()
        return [
            OutcomeEvidence(
                id=row.id,
                memory_id=row.memory_id,
                memory_version=row.memory_version,
                run_id=row.run_id,
                workspace_id=row.workspace_id,
                outcome=OutcomeStatus(row.outcome),
                applied=row.applied,
                selected=row.selected,
                environment_id=row.environment_id,
                environment_fingerprint=row.environment_fingerprint,
                task_family=row.task_family,
                severity=row.severity,
                notes=row.notes,
                evidence=row.evidence_json or {},
                recorded_at=row.recorded_at,
            )
            for row in rows
        ]

    def context_stats(
        self, memory_id: str, environment_fingerprint: str
    ) -> dict[str, int]:
        rows = self.session.scalars(
            select(OutcomeRow).where(
                OutcomeRow.memory_id == memory_id,
                OutcomeRow.environment_fingerprint == environment_fingerprint,
            )
        ).all()
        stats = {"successes": 0, "failures": 0, "partial": 0, "unknown": 0, "applied": 0}
        for row in rows:
            if row.outcome == str(OutcomeStatus.VERIFIED_SUCCESS):
                stats["successes"] += 1
            elif row.outcome == str(OutcomeStatus.VERIFIED_FAILURE):
                stats["failures"] += 1
            elif row.outcome == str(OutcomeStatus.PARTIAL):
                stats["partial"] += 1
            else:
                stats["unknown"] += 1
            if row.applied:
                stats["applied"] += 1
        return stats


# ------------------------------------------------------- decision/audit store
class SqlDecisionRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def save(self, decision: RetrievalDecision) -> None:
        self.session.merge(
            RetrievalDecisionRow(
                id=decision.id,
                workspace_id=decision.workspace_id,
                request_id=decision.request_id,
                run_id=decision.run_id,
                mode=str(decision.mode),
                status=str(decision.status),
                query_redacted=decision.query,
                policy_version=decision.policy_version,
                decision_json=decision.model_dump(mode="json"),
                total_latency_ms=decision.total_latency_ms,
                created_at=decision.timestamp,
            )
        )
        self.session.flush()

    def get(self, decision_id: str, workspace_id: str) -> RetrievalDecision | None:
        row = self.session.get(RetrievalDecisionRow, decision_id)
        if row is None or row.workspace_id != workspace_id:
            return None
        return RetrievalDecision.model_validate(row.decision_json)

    def list_for_run(self, run_id: str) -> list[RetrievalDecision]:
        rows = self.session.scalars(
            select(RetrievalDecisionRow)
            .where(RetrievalDecisionRow.run_id == run_id)
            .order_by(RetrievalDecisionRow.created_at)
        ).all()
        return [RetrievalDecision.model_validate(row.decision_json) for row in rows]


class SqlAuditRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def append(self, event: AuditEvent) -> None:
        self.session.add(
            AuditEventRow(
                id=event.id,
                workspace_id=event.workspace_id,
                entity_type=event.entity_type,
                entity_id=event.entity_id,
                action=event.action,
                reason=event.reason,
                actor=event.actor,
                before_json=event.before,
                after_json=event.after,
                created_at=event.created_at,
            )
        )
        self.session.flush()

    def list_for_entity(self, entity_id: str, limit: int = 100) -> list[AuditEvent]:
        rows = self.session.scalars(
            select(AuditEventRow)
            .where(AuditEventRow.entity_id == entity_id)
            .order_by(AuditEventRow.created_at.desc())
            .limit(limit)
        ).all()
        return [
            AuditEvent(
                id=row.id,
                workspace_id=row.workspace_id,
                entity_type=row.entity_type,
                entity_id=row.entity_id,
                action=row.action,
                reason=row.reason,
                actor=row.actor,
                before=row.before_json or {},
                after=row.after_json or {},
                created_at=row.created_at,
            )
            for row in rows
        ]


# ------------------------------------------------------ repair/validation store
def _candidate_to_domain(row: RepairCandidateRow) -> RepairCandidate:
    return RepairCandidate.model_validate(row.candidate_json)


class SqlRepairRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create(self, candidate: RepairCandidate) -> RepairCandidate:
        row = RepairCandidateRow(
            id=candidate.id,
            workspace_id=candidate.workspace_id,
            source_run_id=candidate.source_run_id,
            status=str(candidate.status),
            validation_status=str(candidate.validation_status),
            risk_level=str(candidate.risk_level),
            content_hash=candidate.content_hash or candidate.compute_hash(),
            environment_id=candidate.environment_id,
            environment_fingerprint=candidate.environment_fingerprint,
            candidate_json=candidate.model_dump(mode="json"),
            promoted_memory_id=candidate.promoted_memory_id,
            created_at=candidate.created_at,
            updated_at=candidate.updated_at,
        )
        self.session.add(row)
        self.session.flush()
        return _candidate_to_domain(row)

    def get(self, candidate_id: str) -> RepairCandidate | None:
        row = self.session.get(RepairCandidateRow, candidate_id)
        return _candidate_to_domain(row) if row else None

    def update(self, candidate: RepairCandidate) -> RepairCandidate:
        row = self.session.get(RepairCandidateRow, candidate.id)
        if row is None:
            raise NotFoundError(f"candidate {candidate.id} not found")
        row.status = str(candidate.status)
        row.validation_status = str(candidate.validation_status)
        row.candidate_json = candidate.model_dump(mode="json")
        row.promoted_memory_id = candidate.promoted_memory_id
        row.updated_at = utc_now()
        self.session.flush()
        return _candidate_to_domain(row)

    def find_by_hash(self, content_hash: str, workspace_id: str) -> RepairCandidate | None:
        row = self.session.scalar(
            select(RepairCandidateRow).where(
                RepairCandidateRow.content_hash == content_hash,
                RepairCandidateRow.workspace_id == workspace_id,
            )
        )
        return _candidate_to_domain(row) if row else None

    def add_validation(self, result: ValidationResult) -> ValidationResult:
        row = ValidationRow(
            id=result.id,
            candidate_id=result.candidate_id,
            status=str(result.status),
            validator_name=result.validator_name,
            attempt=result.attempt,
            environment_id=result.environment_id,
            environment_fingerprint=result.environment_fingerprint,
            candidate_hash=result.candidate_hash,
            result_json=result.model_dump(mode="json"),
            started_at=result.started_at,
            finished_at=result.finished_at,
        )
        self.session.add(row)
        self.session.flush()
        return result

    def list_validations(self, candidate_id: str) -> list[ValidationResult]:
        rows = self.session.scalars(
            select(ValidationRow)
            .where(ValidationRow.candidate_id == candidate_id)
            .order_by(ValidationRow.started_at)
        ).all()
        return [ValidationResult.model_validate(row.result_json) for row in rows]


# ------------------------------------------------- compatibility/revalidation
class SqlCompatibilityRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def upsert(self, result: CompatibilityResult) -> CompatibilityResult:
        row = self.session.scalar(
            select(CompatibilityRow).where(
                CompatibilityRow.memory_id == result.memory_id,
                CompatibilityRow.memory_version == result.memory_version,
                CompatibilityRow.environment_fingerprint == result.environment_fingerprint,
            )
        )
        if row is None:
            row = CompatibilityRow(
                id=new_id("cmp"),
                memory_id=result.memory_id,
                memory_version=result.memory_version,
                workspace_id=result.evidence.get("workspace_id", ""),
                environment_id=result.environment_id,
                environment_fingerprint=result.environment_fingerprint,
                status=str(result.status),
                reasons_json=result.reasons,
                evidence_json=result.evidence,
                updated_at=utc_now(),
            )
            self.session.add(row)
        else:
            row.status = str(result.status)
            row.reasons_json = result.reasons
            row.evidence_json = result.evidence
            row.updated_at = utc_now()
        self.session.flush()
        return result

    def get(
        self, memory_id: str, environment_fingerprint: str
    ) -> CompatibilityResult | None:
        row = self.session.scalar(
            select(CompatibilityRow).where(
                CompatibilityRow.memory_id == memory_id,
                CompatibilityRow.environment_fingerprint == environment_fingerprint,
            )
        )
        if row is None:
            return None
        return CompatibilityResult(
            memory_id=row.memory_id,
            memory_version=row.memory_version,
            environment_id=row.environment_id,
            environment_fingerprint=row.environment_fingerprint,
            status=CompatibilityStatus(row.status),
            reasons=row.reasons_json or [],
            evidence=row.evidence_json or {},
        )

    def list_for_memory(self, memory_id: str) -> list[CompatibilityResult]:
        rows = self.session.scalars(
            select(CompatibilityRow)
            .where(CompatibilityRow.memory_id == memory_id)
            .order_by(CompatibilityRow.updated_at.desc())
        ).all()
        return [
            CompatibilityResult(
                memory_id=row.memory_id,
                memory_version=row.memory_version,
                environment_id=row.environment_id,
                environment_fingerprint=row.environment_fingerprint,
                status=CompatibilityStatus(row.status),
                reasons=row.reasons_json or [],
                evidence=row.evidence_json or {},
            )
            for row in rows
        ]


class SqlRevalidationRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def enqueue(self, job: RevalidationJob) -> RevalidationJob:
        existing = self.session.scalar(
            select(RevalidationJobRow).where(
                RevalidationJobRow.dedupe_key == job.dedupe_key,
                RevalidationJobRow.state.in_(
                    [str(RevalidationState.QUEUED), str(RevalidationState.RUNNING)]
                ),
            )
        )
        if existing is not None:
            return _job_to_domain(existing)
        row = RevalidationJobRow(
            id=job.id,
            memory_id=job.memory_id,
            memory_version=job.memory_version,
            workspace_id=job.workspace_id,
            reason=str(job.reason),
            target_environment_id=job.target_environment_id,
            target_environment_fingerprint=job.target_environment_fingerprint,
            validation_plan=job.validation_plan,
            priority=job.priority,
            state=str(job.state),
            attempts=job.attempts,
            max_attempts=job.max_attempts,
            dedupe_key=job.dedupe_key,
            result_json=job.result,
            error=job.error,
            created_at=job.created_at,
            updated_at=job.updated_at,
        )
        self.session.add(row)
        try:
            self.session.flush()
        except IntegrityError:
            self.session.rollback()
            existing = self.session.scalar(
                select(RevalidationJobRow).where(RevalidationJobRow.dedupe_key == job.dedupe_key)
            )
            if existing is None:
                raise
            return _job_to_domain(existing)
        return _job_to_domain(row)

    def get(self, job_id: str) -> RevalidationJob | None:
        row = self.session.get(RevalidationJobRow, job_id)
        return _job_to_domain(row) if row else None

    def update(self, job: RevalidationJob) -> RevalidationJob:
        row = self.session.get(RevalidationJobRow, job.id)
        if row is None:
            raise NotFoundError(f"revalidation job {job.id} not found")
        row.state = str(job.state)
        row.attempts = job.attempts
        row.result_json = job.result
        row.error = job.error
        row.updated_at = utc_now()
        self.session.flush()
        return _job_to_domain(row)

    def list_queued(self, limit: int = 10) -> list[RevalidationJob]:
        rows = self.session.scalars(
            select(RevalidationJobRow)
            .where(RevalidationJobRow.state == str(RevalidationState.QUEUED))
            .order_by(RevalidationJobRow.priority, RevalidationJobRow.created_at)
            .limit(limit)
        ).all()
        return [_job_to_domain(row) for row in rows]


def _job_to_domain(row: RevalidationJobRow) -> RevalidationJob:
    from recap.domain.enums import RevalidationReason

    return RevalidationJob(
        id=row.id,
        memory_id=row.memory_id,
        memory_version=row.memory_version,
        workspace_id=row.workspace_id,
        reason=RevalidationReason(row.reason),
        target_environment_id=row.target_environment_id,
        target_environment_fingerprint=row.target_environment_fingerprint,
        validation_plan=row.validation_plan,
        priority=row.priority,
        state=RevalidationState(row.state),
        attempts=row.attempts,
        max_attempts=row.max_attempts,
        dedupe_key=row.dedupe_key,
        result=row.result_json or {},
        error=row.error,
        created_at=row.created_at,
        updated_at=row.updated_at,
    )


# ------------------------------------------------------------- environment
class SqlEnvironmentRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def upsert(self, workspace_id: str, fingerprint, metadata: dict[str, Any] | None = None) -> None:  # type: ignore[no-untyped-def]
        from recap.infrastructure.db.models import EnvironmentRow

        row = self.session.get(EnvironmentRow, fingerprint.environment_id)
        if row is None:
            row = EnvironmentRow(
                id=fingerprint.environment_id,
                workspace_id=workspace_id,
                environment_type=fingerprint.environment_type,
                version=fingerprint.environment_version,
                fingerprint=fingerprint.fingerprint,
                metadata_json=metadata or fingerprint.canonical_payload(),
                created_at=utc_now(),
            )
            self.session.add(row)
        else:
            row.fingerprint = fingerprint.fingerprint
            row.version = fingerprint.environment_version
            row.metadata_json = metadata or fingerprint.canonical_payload()
        self.session.flush()


# ------------------------------------------------------------------ outbox
class SqlOutboxRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def enqueue(
        self, *, workspace_id: str, entity_id: str, operation: str, dedupe_key: str
    ) -> None:
        existing = self.session.scalar(
            select(OutboxRow).where(
                OutboxRow.dedupe_key == dedupe_key,
                OutboxRow.state.in_(["pending", "processing"]),
            )
        )
        if existing is not None:
            return
        now = utc_now()
        self.session.add(
            OutboxRow(
                id=new_id("obx"),
                workspace_id=workspace_id,
                entity_type="memory",
                entity_id=entity_id,
                operation=operation,
                dedupe_key=dedupe_key,
                state="pending",
                attempts=0,
                max_attempts=8,
                last_error="",
                next_attempt_at=now,
                created_at=now,
                updated_at=now,
            )
        )
        self.session.flush()

    def claim_due(self, limit: int = 20) -> list[OutboxRow]:
        now = utc_now()
        rows = self.session.scalars(
            select(OutboxRow)
            .where(OutboxRow.state == "pending", OutboxRow.next_attempt_at <= now)
            .order_by(OutboxRow.next_attempt_at)
            .limit(limit)
            .with_for_update(skip_locked=True)
        ).all()
        for row in rows:
            row.state = "processing"
            row.attempts += 1
            row.updated_at = now
        self.session.flush()
        return list(rows)

    def mark_done(self, row: OutboxRow) -> None:
        row.state = "done"
        row.updated_at = utc_now()
        self.session.flush()

    def mark_failed(self, row: OutboxRow, error: str) -> None:
        row.last_error = error[:2000]
        row.updated_at = utc_now()
        if row.attempts >= row.max_attempts:
            row.state = "dead"
        else:
            row.state = "pending"
            delay = min(2**row.attempts, 300)
            row.next_attempt_at = utc_now() + timedelta(seconds=delay)
        self.session.flush()

    def pending_count(self) -> int:
        return int(
            self.session.scalar(
                select(func.count()).select_from(OutboxRow).where(OutboxRow.state == "pending")
            )
            or 0
        )


def ensure_workspace(session: Session, workspace_id: str, name: str | None = None) -> None:
    if session.get(WorkspaceRow, workspace_id) is None:
        session.add(
            WorkspaceRow(id=workspace_id, name=name or workspace_id, created_at=utc_now())
        )
        session.flush()
