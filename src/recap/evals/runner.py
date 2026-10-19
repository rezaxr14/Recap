"""Benchmark runner: fair baselines over the same tasks, real measured metrics.

Baselines compared (all deterministic, same tool budget):
  - no_memory
  - retry_only
  - bm25
  - vector
  - hybrid_rrf
  - outcome_aware (RECAP selection policy)

Frozen-memory setting: retrieval is read-only and tasks never write memories
during the evaluation, so test memories cannot leak into later test episodes.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from recap.container import Container
from recap.domain.enums import EventType, MemoryStatus, RetrievalMode, RunStatus
from recap.domain.retrieval import RetrievalContext, RetrievalRequest
from recap.environments.mock_api import PaginatedApiEnvironment
from recap.environments.reference_agent import ReferenceAgent
from recap.environments.run_recording import (
    AgentRunMetadata,
    complete_recorded_run,
    start_recorded_run,
)
from recap.evals.datasets import BenchmarkTask, default_tasks, split_hash
from recap.evals.metrics import (
    bootstrap_ci,
    mean,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)
from recap.logging import get_logger

logger = get_logger(__name__)

BASELINES: list[tuple[str, RetrievalMode | None]] = [
    ("no_memory", None),
    ("retry_only", None),
    ("bm25", RetrievalMode.BM25),
    ("vector", RetrievalMode.VECTOR),
    ("hybrid_rrf", RetrievalMode.HYBRID_RRF),
    ("outcome_aware", RetrievalMode.OUTCOME_AWARE),
    # Chunk-granule ablations on the same tasks. A chunk mode is only meaningful
    # because the chunker and chunk index are real; it is labeled in the report.
    ("bm25_chunk", RetrievalMode.BM25_CHUNK),
    ("vector_chunk", RetrievalMode.VECTOR_CHUNK),
    ("hybrid_rrf_chunk", RetrievalMode.HYBRID_RRF_CHUNK),
    ("outcome_aware_chunk", RetrievalMode.OUTCOME_AWARE_CHUNK),
]


def run_benchmark(
    container: Container,
    *,
    split: str = "dev",
    output_dir: Path | str = Path("benchmark/reports"),
    workspace_id: str = "benchmark-workspace",
) -> dict[str, Any]:
    container.migrate()
    container.init_search()
    container.init_search_chunks()
    tasks = [task for task in default_tasks() if task.split == split]
    if not tasks:
        raise ValueError(f"no tasks for split {split}")

    _seed_memories(container, workspace_id, tasks)

    per_task: list[dict[str, Any]] = []
    for baseline, mode in BASELINES:
        for task in tasks:
            per_task.append(_run_task(container, workspace_id, task, baseline=baseline, mode=mode))

    summary = _summarize(per_task)
    by_family = _summarize_by_family(per_task)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    json_path = output / f"benchmark_{split}_{stamp}.json"
    csv_path = output / f"benchmark_{split}_{stamp}.csv"
    md_path = output / f"benchmark_{split}_{stamp}.md"
    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "split": split,
        "dataset_hash": split_hash(tasks),
        "tasks": [task.to_dict() for task in tasks],
        "baselines": [name for name, _ in BASELINES],
        "per_task": per_task,
        "summary": summary,
        "by_task_family": by_family,
        "protocol": {
            "memory_setting": "frozen_memory",
            "memory_snapshot": "one per run, created before retrieval",
            "retrieval_scope": "run",
            "agent": "deterministic-reference-agent-v1",
            "provider": "deterministic",
            "environment": "mock-api",
            "metrics_source": "persisted_run_records",
            "note": ("Deterministic mock pipeline test. Not evidence of real LLM performance."),
        },
    }
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    _write_csv(csv_path, per_task)
    md_path.write_text(_render_markdown(payload), encoding="utf-8")
    return {
        "summary": summary,
        "per_task": per_task,
        "report_path": str(md_path),
        "json_path": str(json_path),
        "csv_path": str(csv_path),
    }


def _seed_memories(container: Container, workspace_id: str, tasks: list[BenchmarkTask]) -> None:
    """Create one validated memory per environment version present in the split.

    Memories are generated through the real compile -> validate -> promote path,
    so the benchmark exercises the production promotion gate.
    """
    versions = sorted({task.environment_version for task in tasks})
    with container.services() as services:
        from recap.infrastructure.db.repositories import ensure_workspace

        ensure_workspace(services.session, workspace_id)
        existing = services.memories.list_memories(
            workspace_id, status=MemoryStatus.ACTIVE, limit=100
        )
    # Seed per environment version so adding a new version to a split does not
    # silently reuse an incomplete memory set.
    seeded_versions = {
        memory.environment_id.rsplit("-v", 1)[-1] for memory in existing if memory.environment_id
    }
    versions_to_seed = [version for version in versions if version not in seeded_versions]

    for version in versions_to_seed:
        environment = PaginatedApiEnvironment(version=version)
        fingerprint = environment.fingerprint()
        from recap.environments.run_recording import record_reference_run

        run_id, _ = record_reference_run(
            container, environment, workspace_id=workspace_id, task_id=f"seed-{version}"
        )
        with container.services() as services:
            services.environments.upsert(workspace_id, fingerprint)
            candidates = services.repair_compiler.compile(run_id, environment=fingerprint)
            if not candidates:
                continue
            candidate_id = candidates[0].id
            result = services.repair_compiler.validate(
                candidate_id, environment=fingerprint, validator_name="mock_sandbox"
            )
            if result.status.value != "passed":
                logger.warning("benchmark seed validation failed for v%s", version)
                continue
            refreshed = services.repairs.get(candidate_id)
            assert refreshed is not None
            services.promotion.promote(refreshed)

    # Chunks are derived state: rebuild them for every active memory so an
    # already-seeded benchmark workspace is brought up to date idempotently.
    with container.services() as services:
        memories = services.memories.list_memories(
            workspace_id, status=MemoryStatus.ACTIVE, limit=100
        )
        for memory in memories:
            if memory.environment_id.rsplit("-v", 1)[-1] not in versions:
                continue
            chunk_report = services.chunk_service.rebuild(memory)
            if not chunk_report.indexed:
                logger.warning(
                    "benchmark chunk build failed for %s: %s", memory.id, chunk_report.error
                )


def _run_task(
    container: Container,
    workspace_id: str,
    task: BenchmarkTask,
    *,
    baseline: str,
    mode: RetrievalMode | None,
) -> dict[str, Any]:
    environment = PaginatedApiEnvironment(version=task.environment_version)
    fingerprint = environment.fingerprint()
    agent = ReferenceAgent()

    metadata = AgentRunMetadata(
        agent_id="benchmark-reference-agent",
        model_provider="deterministic",
        model_name="reference-agent-v1",
        task_family=task.task_family,
        success_criteria="all records collected",
        failure_type="benchmark_task_failed",
        config_snapshot={
            "baseline": baseline,
            "retrieval_mode": mode.value if mode is not None else None,
            "split": task.split,
            "dataset_hash": split_hash([task]),
            "run_scoped_retrieval": True,
        },
    )
    # Start the run before retrieval so its snapshot pins the candidate set for
    # the whole task. All reported task metrics below are read back from this
    # persisted run, not from an in-memory shortcut.
    run = start_recorded_run(
        container,
        environment,
        metadata=metadata,
        workspace_id=workspace_id,
        task_id=task.task_id,
    )

    selected_ids: list[str] = []
    status = "not_retrieved"
    ranked_ids: list[str] = []
    error_code = (
        "UNKNOWN_PARAMETER" if task.failure_mode == "renamed_parameter" else "INVALID_CURSOR"
    )
    error_message = (
        "unknown parameter 'cursor'; this API version requires 'page_token'"
        if task.failure_mode == "renamed_parameter"
        else "cursor is invalid or expired"
    )
    request: RetrievalRequest | None = None
    if mode is not None:
        with container.services() as services:
            request = RetrievalRequest(
                context=RetrievalContext(
                    workspace_id=workspace_id,
                    query=task.instruction,
                    task_family=task.task_family,
                    environment_id=fingerprint.environment_id,
                    environment_fingerprint=fingerprint.fingerprint,
                    environment_version=fingerprint.environment_version,
                    environment_type=fingerprint.environment_type,
                    tool_names=["list_records"],
                    tool_versions=fingerprint.tool_versions,
                    tool_schema_fingerprints=fingerprint.tool_schema_hashes,
                    error_type="ValidationError",
                    error_code=error_code,
                    error_message=error_message,
                    state={"tool_schema_fingerprints": fingerprint.tool_schema_hashes},
                ),
                mode=mode,
                final_k=3,
                run_id=run.id,
            )
            decision = services.recall.retrieve(request)
            status = decision.status.value
            ranked_ids = [candidate.memory_id for candidate in decision.candidates]
            selected_ids = [memory.memory_id for memory in decision.selected_memories]

    memory = None
    if baseline == "retry_only":
        # A naive retry with the same invalid cursor cannot succeed.
        result = agent.run(environment, memory=None)
    elif selected_ids:
        with container.services() as services:
            decision_memory = services.memories.get_memory(selected_ids[0], workspace_id)
            if decision_memory is not None:
                from recap.domain.retrieval import CandidateEvaluation
                from recap.recall.engine import _to_selected

                memory = _to_selected(
                    decision_memory,
                    CandidateEvaluation(
                        memory_id=decision_memory.id,
                        memory_version=decision_memory.memory_version,
                        title=decision_memory.content.title,
                        status=str(decision_memory.status),
                    ),
                )
        result = (
            agent.run(environment, memory=memory) if memory is not None else agent.run(environment)
        )
    else:
        result = agent.run(environment)

    run_id = complete_recorded_run(
        container,
        run,
        environment,
        result,
        metadata=metadata,
        memory=memory,
        memory_source="retrieval_selection",
    )
    relevant = _relevant_ids(container, workspace_id, task)
    incompatible_selected = _count_incompatible_selected(
        container, workspace_id, selected_ids, task, error_code=error_code
    )
    with container.services() as services:
        recorded = services.recorder.get_run(run_id)
        events = services.runs.list_events(run_id)
    used_memory_id = next(
        (
            str(event.payload_redacted["memory_id"])
            for event in events
            if event.event_type is EventType.MEMORY_APPLIED
            and event.payload_redacted.get("memory_id")
        ),
        None,
    )
    finished = next((event for event in events if event.event_type is EventType.RUN_FINISHED), None)
    error = None
    if finished is not None:
        value = finished.payload_redacted.get("error")
        error = str(value) if value else None
    success = recorded.status == RunStatus.SUCCEEDED
    return {
        "task_id": task.task_id,
        "task_family": task.task_family,
        "baseline": baseline,
        "split": task.split,
        "environment_version": task.environment_version,
        "run_id": run_id,
        "memory_snapshot_id": recorded.memory_snapshot_id,
        "retrieval_status": status,
        "selected_memory_ids": selected_ids,
        "relevant_memory_ids": sorted(relevant),
        "recall_at_3": recall_at_k(relevant, ranked_ids, 3),
        "mrr": reciprocal_rank(relevant, ranked_ids),
        "ndcg_at_3": ndcg_at_k(relevant, ranked_ids, 3),
        "task_success": success,
        "tool_calls": recorded.tool_calls_count,
        "retries": recorded.retry_count,
        "used_memory_id": used_memory_id,
        "harmful_transfer": bool(used_memory_id and not success),
        "incompatible_selected": incompatible_selected,
        "error": error,
    }


def _count_incompatible_selected(
    container: Container,
    workspace_id: str,
    selected_ids: list[str],
    task: BenchmarkTask,
    *,
    error_code: str,
) -> int:
    """Independently re-evaluate selected memories against the task context.

    This is a *safety* metric, not an outcome metric: it counts memories that a
    policy-free baseline selected even though their declared preconditions or
    schema compatibility do not hold for this environment version. It is
    computed with the same evaluators used by the outcome-aware policy, but it
    never influences selection, so it cannot flatter any baseline.
    """
    if not selected_ids:
        return 0
    from recap.recall.applicability import combine_compatibility, evaluate_applicability

    environment = PaginatedApiEnvironment(version=task.environment_version)
    fingerprint = environment.fingerprint()
    context = RetrievalContext(
        workspace_id=workspace_id,
        query=task.instruction,
        task_family=task.task_family,
        environment_id=fingerprint.environment_id,
        environment_fingerprint=fingerprint.fingerprint,
        environment_version=fingerprint.environment_version,
        environment_type=fingerprint.environment_type,
        tool_names=["list_records"],
        tool_versions=fingerprint.tool_versions,
        tool_schema_fingerprints=fingerprint.tool_schema_hashes,
        error_type="ValidationError",
        error_code=error_code,
        state={"tool_schema_fingerprints": fingerprint.tool_schema_hashes},
    )
    count = 0
    with container.services() as services:
        for memory_id in selected_ids:
            memory = services.memories.get_memory(memory_id, workspace_id)
            if memory is None:
                continue
            compatibility, _ = combine_compatibility(memory, context, None)
            applicability = evaluate_applicability(memory, context)
            schema_ok = all(
                context.tool_schema_fingerprints.get(tool) == expected
                for tool, expected in memory.content.tool_schema_fingerprints.items()
                if tool in context.tool_schema_fingerprints
            )
            if (
                compatibility.value == "incompatible"
                or applicability.result.value != "satisfied"
                or not schema_ok
            ):
                count += 1
    return count


def _relevant_ids(container: Container, workspace_id: str, task: BenchmarkTask) -> set[str]:
    with container.services() as services:
        memories = services.memories.list_memories(
            workspace_id, status=MemoryStatus.ACTIVE, limit=100
        )
    relevant = set()
    for memory in memories:
        if (
            memory.environment_fingerprint
            and task.environment_version == memory.environment_id.rsplit("-v", 1)[-1]
        ):
            relevant.add(memory.id)
    return relevant


def _summarize(per_task: list[dict[str, Any]]) -> dict[str, Any]:
    baselines = sorted({row["baseline"] for row in per_task})
    return {
        baseline: _summarize_rows([row for row in per_task if row["baseline"] == baseline])
        for baseline in baselines
    }


def _summarize_by_family(per_task: list[dict[str, Any]]) -> dict[str, Any]:
    families = sorted({row["task_family"] for row in per_task})
    return {
        family: _summarize([row for row in per_task if row["task_family"] == family])
        for family in families
    }


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successes = [1.0 if row["task_success"] else 0.0 for row in rows]
    lower, upper = bootstrap_ci(successes)
    return {
        "tasks": len(rows),
        "task_success_rate": mean(successes),
        "task_success_ci95": [lower, upper],
        "mean_tool_calls": mean([row["tool_calls"] for row in rows]),
        "mean_retries": mean([row["retries"] for row in rows]),
        "recall_at_3": mean([row["recall_at_3"] for row in rows]),
        "mrr": mean([row["mrr"] for row in rows]),
        "ndcg_at_3": mean([row["ndcg_at_3"] for row in rows]),
        "harmful_transfer_rate": mean([1.0 if row["harmful_transfer"] else 0.0 for row in rows]),
        "incompatible_selection_rate": mean(
            [1.0 if row["incompatible_selected"] else 0.0 for row in rows]
        ),
        "total_incompatible_selected": sum(row["incompatible_selected"] for row in rows),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: value for key, value in row.items()})


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        f"# RECAP benchmark — split `{payload['split']}`",
        "",
        f"Generated: {payload['generated_at']}",
        f"Dataset hash: `{payload['dataset_hash']}`",
        "",
        "> Deterministic mock environment and deterministic reference agent. These numbers "
        "measure pipeline behavior, not real LLM agent performance.",
        "",
        "## Summary",
        "",
        "| baseline | success | 95% CI | recall@3 | MRR | nDCG@3 | tool calls | harmful transfer | incompatible selection |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for baseline, stats in payload["summary"].items():
        lines.append(
            f"| {baseline} | {stats['task_success_rate']:.3f} | "
            f"[{stats['task_success_ci95'][0]:.3f}, {stats['task_success_ci95'][1]:.3f}] | "
            f"{stats['recall_at_3']:.3f} | {stats['mrr']:.3f} | {stats['ndcg_at_3']:.3f} | "
            f"{stats['mean_tool_calls']:.2f} | {stats['harmful_transfer_rate']:.3f} | "
            f"{stats['incompatible_selection_rate']:.3f} |"
        )
    lines.extend(
        [
            "",
            "`incompatible selection` is an independent safety audit: selected memories are "
            "re-evaluated after the fact against the task's schema and preconditions. It never "
            "influences selection, so it cannot flatter any baseline.",
        ]
    )
    by_family = payload.get("by_task_family") or {}
    if by_family:
        lines.extend(
            [
                "",
                "## Per-family metrics",
                "",
                "| family | baseline | success | recall@3 | MRR | harmful transfer | incompatible selection |",
                "|---|---|---|---|---|---|---|",
            ]
        )
        for family, baselines in by_family.items():
            for baseline, stats in baselines.items():
                lines.append(
                    f"| {family} | {baseline} | {stats['task_success_rate']:.3f} | "
                    f"{stats['recall_at_3']:.3f} | {stats['mrr']:.3f} | "
                    f"{stats['harmful_transfer_rate']:.3f} | "
                    f"{stats['incompatible_selection_rate']:.3f} |"
                )
    lines.extend(
        [
            "",
            "## Per-task results",
            "",
            "| task | baseline | run | success | status | error |",
            "|---|---|---|---|---|---|",
        ]
    )
    for row in payload["per_task"]:
        error = row["error"] or ""
        lines.append(
            f"| {row['task_id']} | {row['baseline']} | `{row['run_id']}` | "
            f"{row['task_success']} | {row['retrieval_status']} | {error[:60]} |"
        )
    lines.extend(
        [
            "",
            "## Reproduction",
            "",
            "```bash",
            f"recap eval run --split {payload['split']}",
            "```",
        ]
    )
    return "\n".join(lines)
