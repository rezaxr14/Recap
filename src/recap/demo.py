"""Deterministic end-to-end demo of the three RECAP subsystems.

Flow (plan section 20.1):
 1. mock API v1 with an invalid-cursor failure
 2. reference agent run recorded with structured events
 3. RepairCompiler produces a typed candidate
 4. sandbox validation + policy-checked promotion to active memory
 5. outcome-aware retrieval finds it
 6. verified success recorded
 7. mock API v2 changes the schema
 8. old memory is rejected for v2 and marked incompatible/suspect
 9. v2 repair is compiled, validated, promoted, and retrieved
10. benchmark + generated report

Everything here is deterministic and requires no external model or API key.
"""

from __future__ import annotations

import contextlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from recap.container import Container
from recap.domain.enums import (
    OutcomeStatus,
    RetrievalMode,
    RevalidationReason,
)
from recap.domain.feedback import AuditEvent
from recap.domain.retrieval import RetrievalContext, RetrievalRequest
from recap.environments.mock_api import PaginatedApiEnvironment
from recap.environments.reference_agent import ReferenceAgent
from recap.logging import get_logger

logger = get_logger(__name__)

WORKSPACE = "demo-workspace"


def run_demo(container: Container, *, write_report: bool = True) -> dict[str, Any]:
    container.migrate()
    container.init_search()
    _reset_demo_state(container)
    report: dict[str, Any] = {
        "started_at": datetime.now(UTC).isoformat(),
        "workspace_id": WORKSPACE,
        "steps": [],
    }

    environment_v1 = PaginatedApiEnvironment(version="1")
    fingerprint_v1 = environment_v1.fingerprint()

    with container.services() as services:
        from recap.infrastructure.db.repositories import ensure_workspace

        ensure_workspace(services.session, WORKSPACE)
        services.environments.upsert(WORKSPACE, fingerprint_v1)

    run_id = _record_failed_run(container, environment_v1)
    report["steps"].append({"step": "record_failed_run", "run_id": run_id})

    with container.services() as services:
        candidates = services.repair_compiler.compile(run_id, environment=fingerprint_v1)
        if not candidates:
            raise RuntimeError("demo failed: no repair candidate produced")
        candidate = candidates[0]
        candidate_id = candidate.id
        static = services.repair_compiler.validate(
            candidate_id, environment=fingerprint_v1, validator_name="static"
        )
        sandbox = services.repair_compiler.validate(
            candidate_id, environment=fingerprint_v1, validator_name="mock_sandbox"
        )
        if sandbox.status.value != "passed":
            raise RuntimeError(f"demo failed: sandbox validation {sandbox.status}")
        refreshed = services.repairs.get(candidate_id)
        if refreshed is None:
            raise RuntimeError("demo failed: candidate disappeared before promotion")
        memory_v1 = services.promotion.promote(refreshed)
        memory_v1_id = memory_v1.id
    report["steps"].append(
        {
            "step": "compile_validate_promote",
            "candidate_id": candidate_id,
            "static_validation": static.status.value,
            "sandbox_validation": sandbox.status.value,
            "memory_id": memory_v1_id,
        }
    )

    decision_v1 = _retrieve(
        container,
        query="API rejected the pagination cursor with INVALID_CURSOR",
        fingerprint=fingerprint_v1,
        error_code="INVALID_CURSOR",
        error_message="cursor 'start' is invalid or expired",
        tool_name="list_records",
    )
    report["steps"].append(
        {
            "step": "retrieve_v1",
            "status": decision_v1.status.value,
            "selected": [m.memory_id for m in decision_v1.selected_memories],
            "top_candidates": [
                {
                    "memory_id": c.memory_id,
                    "score": round(c.final_score, 4),
                    "compatibility": c.compatibility_status.value,
                    "rejected": c.rejection_reasons,
                }
                for c in decision_v1.candidates[:5]
            ],
        }
    )
    if not decision_v1.selected_memories:
        raise RuntimeError("demo failed: v1 memory was not retrieved")

    # Apply the retrieved memory with the reference agent and verify the outcome.
    agent = ReferenceAgent()
    applied = agent.run(environment_v1, memory=decision_v1.selected_memories[0])
    report["steps"].append(
        {
            "step": "apply_memory_v1",
            "agent_success": applied.success,
            "tool_calls": applied.tool_calls,
            "retries": applied.retries,
        }
    )
    with container.services() as services:
        memory = services.memories.get_memory(memory_v1_id, WORKSPACE)
        assert memory is not None
        services.memory_shift.record_outcome(
            memory,
            outcome=OutcomeStatus.VERIFIED_SUCCESS
            if applied.success
            else OutcomeStatus.VERIFIED_FAILURE,
            applied=True,
            environment_id=fingerprint_v1.environment_id,
            environment_fingerprint=fingerprint_v1.fingerprint,
            task_family="api_pagination",
            notes="verified by environment postcondition",
        )

    # Version 2: schema changes, old memory must not be selected.
    environment_v2 = PaginatedApiEnvironment(version="2")
    fingerprint_v2 = environment_v2.fingerprint()
    with container.services() as services:
        memory = services.memories.get_memory(memory_v1_id, WORKSPACE)
        assert memory is not None
        compatibility = services.memory_shift.evaluate_compatibility(
            memory,
            environment_id=fingerprint_v2.environment_id,
            environment_fingerprint=fingerprint_v2.fingerprint,
            tool_schema_fingerprints=fingerprint_v2.tool_schema_hashes,
            tool_versions=fingerprint_v2.tool_versions,
        )
        services.environments.upsert(WORKSPACE, fingerprint_v2)
        job = services.memory_shift.enqueue_revalidation(
            memory,
            reason=RevalidationReason.VERSION_CHANGE,
            environment_id=fingerprint_v2.environment_id,
            environment_fingerprint=fingerprint_v2.fingerprint,
        )
        services.audit.append(
            AuditEvent(
                workspace_id=WORKSPACE,
                entity_type="memory",
                entity_id=memory.id,
                action="compatibility_evaluated",
                reason="; ".join(compatibility.reasons),
                after={"status": compatibility.status.value},
            )
        )
    report["steps"].append(
        {
            "step": "version_change",
            "compatibility": compatibility.status.value,
            "reasons": compatibility.reasons,
            "revalidation_job": job.id,
        }
    )

    decision_v2_stale = _retrieve(
        container,
        query="API rejected the pagination cursor with INVALID_CURSOR",
        fingerprint=fingerprint_v2,
        error_code="INVALID_CURSOR",
        error_message="cursor 'start' is invalid or expired",
        tool_name="list_records",
    )
    report["steps"].append(
        {
            "step": "retrieve_v2_stale_memory",
            "status": decision_v2_stale.status.value,
            "selected": [m.memory_id for m in decision_v2_stale.selected_memories],
            "reason": decision_v2_stale.reason,
        }
    )
    if any(m.memory_id == memory_v1_id for m in decision_v2_stale.selected_memories):
        raise RuntimeError("demo failed: stale v1 memory was selected for v2")

    # Compile and promote the v2 replacement.
    run_v2_id = _record_failed_run(container, environment_v2)
    with container.services() as services:
        candidates_v2 = services.repair_compiler.compile(run_v2_id, environment=fingerprint_v2)
        if not candidates_v2:
            raise RuntimeError("demo failed: no v2 repair candidate")
        candidate_v2_id = candidates_v2[0].id
        validation_v2 = services.repair_compiler.validate(
            candidate_v2_id, environment=fingerprint_v2, validator_name="mock_sandbox"
        )
        if validation_v2.status.value != "passed":
            raise RuntimeError(f"demo failed: v2 validation {validation_v2.status}")
        refreshed_v2 = services.repairs.get(candidate_v2_id)
        if refreshed_v2 is None:
            raise RuntimeError("demo failed: v2 candidate disappeared before promotion")
        memory_v2 = services.promotion.promote(refreshed_v2)
        memory_v2_id = memory_v2.id
    report["steps"].append(
        {
            "step": "compile_validate_promote_v2",
            "candidate_id": candidate_v2_id,
            "validation": validation_v2.status.value,
            "memory_id": memory_v2_id,
        }
    )

    decision_v2_fresh = _retrieve(
        container,
        query="API requires page_token and rejected the legacy cursor parameter",
        fingerprint=fingerprint_v2,
        error_code="UNKNOWN_PARAMETER",
        error_message="unknown parameter 'cursor'; this API version requires 'page_token'",
        tool_name="list_records",
    )
    report["steps"].append(
        {
            "step": "retrieve_v2_fresh_memory",
            "status": decision_v2_fresh.status.value,
            "selected": [m.memory_id for m in decision_v2_fresh.selected_memories],
        }
    )
    if not any(m.memory_id == memory_v2_id for m in decision_v2_fresh.selected_memories):
        raise RuntimeError("demo failed: fresh v2 memory not retrieved")

    # Benchmark
    from recap.evals.runner import run_benchmark

    benchmark = run_benchmark(container, split="dev", output_dir=Path("benchmark/reports"))
    report["benchmark"] = benchmark["summary"]

    report["summary"] = {
        "failed_run_id": run_id,
        "repair_candidate_id": candidate_id,
        "memory_v1": memory_v1_id,
        "memory_v2": memory_v2_id,
        "stale_memory_blocked": True,
        "fresh_memory_retrieved": True,
        "benchmark_report": benchmark["report_path"],
    }
    report["finished_at"] = datetime.now(UTC).isoformat()

    if write_report:
        path = Path("demo_report.md")
        path.write_text(_render_report(report), encoding="utf-8")
        json_path = Path("benchmark/reports/demo_run.json")
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        report["report_path"] = str(path)
    return report


def _reset_demo_state(container: Container) -> None:
    """Remove only this demo workspace's data so the demo is reproducible.

    Scoped to ``WORKSPACE`` and to benchmark workspaces; never a global wipe.
    """
    from sqlalchemy import delete

    from recap.infrastructure.db.models import (
        AuditEventRow,
        CompatibilityRow,
        MemoryRow,
        OutboxRow,
        OutcomeRow,
        RepairCandidateRow,
        RetrievalDecisionRow,
        RevalidationJobRow,
        RunEventRow,
        RunRow,
        ValidationRow,
    )

    with container.database.session() as session:
        run_ids = [
            row
            for row in session.scalars(
                __import__("sqlalchemy")
                .select(RunRow.id)
                .where(RunRow.workspace_id.in_([WORKSPACE, "benchmark-workspace"]))
            )
        ]
        if run_ids:
            session.execute(delete(RunEventRow).where(RunEventRow.run_id.in_(run_ids)))
        candidate_ids = [
            row
            for row in session.scalars(
                __import__("sqlalchemy")
                .select(RepairCandidateRow.id)
                .where(RepairCandidateRow.workspace_id.in_([WORKSPACE, "benchmark-workspace"]))
            )
        ]
        if candidate_ids:
            session.execute(
                delete(ValidationRow).where(ValidationRow.candidate_id.in_(candidate_ids))
            )
        for model in (
            RetrievalDecisionRow,
            RepairCandidateRow,
            RevalidationJobRow,
            CompatibilityRow,
            OutcomeRow,
            MemoryRow,
            AuditEventRow,
            OutboxRow,
            RunRow,
        ):
            session.execute(
                delete(model).where(model.workspace_id.in_([WORKSPACE, "benchmark-workspace"]))
            )
    with contextlib.suppress(Exception):
        # The index may not exist yet on a truly clean checkout.
        container.es_client.delete_by_query(
            index=container.index_manager.write_index,
            query={"terms": {"workspace_id": [WORKSPACE, "benchmark-workspace"]}},
            refresh=True,
            conflicts="proceed",
        )


def _record_failed_run(container: Container, environment: PaginatedApiEnvironment) -> str:
    from recap.environments.run_recording import record_reference_run

    run_id, _ = record_reference_run(container, environment, workspace_id=WORKSPACE)
    return run_id


def _retrieve(
    container: Container,
    *,
    query: str,
    fingerprint,
    error_code: str,
    error_message: str,
    tool_name: str,
) -> Any:
    with container.services() as services:
        request = RetrievalRequest(
            context=RetrievalContext(
                workspace_id=WORKSPACE,
                query=query,
                task_family="api_pagination",
                environment_id=fingerprint.environment_id,
                environment_fingerprint=fingerprint.fingerprint,
                environment_version=fingerprint.environment_version,
                environment_type=fingerprint.environment_type,
                tool_names=[tool_name],
                tool_versions=fingerprint.tool_versions,
                tool_schema_fingerprints=fingerprint.tool_schema_hashes,
                error_type="ValidationError",
                error_code=error_code,
                error_message=error_message,
                state={
                    "records_collected": 0,
                    "tool_schema_fingerprints": fingerprint.tool_schema_hashes,
                },
            ),
            mode=RetrievalMode.OUTCOME_AWARE,
            final_k=3,
        )
        return services.recall.retrieve(request)


def _render_report(report: dict[str, Any]) -> str:
    lines = [
        "# RECAP demo report",
        "",
        f"Generated: {report['finished_at']}",
        "",
        "> This report is generated from a deterministic mock environment and a "
        "deterministic reference agent. It is a pipeline/reproducibility demonstration, "
        "**not** evidence of real LLM agent performance.",
        "",
        "## Steps",
        "",
    ]
    for step in report["steps"]:
        lines.append(f"### {step['step']}")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(step, indent=2, default=str))
        lines.append("```")
        lines.append("")
    lines.append("## Benchmark summary")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(report.get("benchmark", {}), indent=2, default=str))
    lines.append("```")
    lines.append("")
    lines.append("## Reproduction")
    lines.append("")
    lines.append("```bash")
    lines.append("recap db migrate")
    lines.append("recap search init")
    lines.append("recap demo run")
    lines.append("```")
    return "\n".join(lines)
