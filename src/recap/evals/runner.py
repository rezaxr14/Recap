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
from recap.domain.enums import MemoryStatus, OutcomeStatus, RetrievalMode
from recap.domain.retrieval import RetrievalContext, RetrievalRequest
from recap.environments.mock_api import PaginatedApiEnvironment
from recap.environments.reference_agent import ReferenceAgent
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
    tasks = [task for task in default_tasks() if task.split == split]
    if not tasks:
        raise ValueError(f"no tasks for split {split}")

    _seed_memories(container, workspace_id, tasks)

    per_task: list[dict[str, Any]] = []
    for baseline, mode in BASELINES:
        for task in tasks:
            per_task.append(
                _run_task(container, workspace_id, task, baseline=baseline, mode=mode)
            )

    summary = _summarize(per_task)
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
        "protocol": {
            "memory_setting": "frozen_memory",
            "agent": "deterministic-reference-agent-v1",
            "provider": "deterministic",
            "environment": "mock-api",
            "note": (
                "Deterministic mock pipeline test. Not evidence of real LLM performance."
            ),
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
    if existing:
        return

    for version in versions:
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
            services.promotion.promote(refreshed)


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

    selected_ids: list[str] = []
    status = "not_retrieved"
    ranked_ids: list[str] = []
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
                    error_code="INVALID_CURSOR",
                    error_message="cursor is invalid or expired",
                    state={"tool_schema_fingerprints": fingerprint.tool_schema_hashes},
                ),
                mode=mode,
                final_k=3,
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
                from recap.recall.engine import _to_selected

                from recap.domain.retrieval import CandidateEvaluation

                memory = _to_selected(
                    decision_memory,
                    CandidateEvaluation(
                        memory_id=decision_memory.id,
                        memory_version=decision_memory.memory_version,
                        title=decision_memory.content.title,
                        status=str(decision_memory.status),
                    ),
                )
        result = agent.run(environment, memory=memory) if memory is not None else agent.run(environment)
    else:
        result = agent.run(environment)

    relevant = _relevant_ids(container, workspace_id, task)
    return {
        "task_id": task.task_id,
        "baseline": baseline,
        "split": task.split,
        "environment_version": task.environment_version,
        "retrieval_status": status,
        "selected_memory_ids": selected_ids,
        "relevant_memory_ids": sorted(relevant),
        "recall_at_3": recall_at_k(relevant, ranked_ids, 3),
        "mrr": reciprocal_rank(relevant, ranked_ids),
        "ndcg_at_3": ndcg_at_k(relevant, ranked_ids, 3),
        "task_success": result.success,
        "tool_calls": result.tool_calls,
        "retries": result.retries,
        "used_memory_id": result.used_memory_id,
        "harmful_transfer": bool(result.used_memory_id and not result.success),
        "error": result.final_error,
    }


def _relevant_ids(container: Container, workspace_id: str, task: BenchmarkTask) -> set[str]:
    with container.services() as services:
        memories = services.memories.list_memories(
            workspace_id, status=MemoryStatus.ACTIVE, limit=100
        )
    relevant = set()
    for memory in memories:
        if memory.environment_fingerprint and task.environment_version == memory.environment_id.rsplit("-v", 1)[-1]:
            relevant.add(memory.id)
    return relevant


def _summarize(per_task: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    baselines = sorted({row["baseline"] for row in per_task})
    for baseline in baselines:
        rows = [row for row in per_task if row["baseline"] == baseline]
        successes = [1.0 if row["task_success"] else 0.0 for row in rows]
        lower, upper = bootstrap_ci(successes)
        summary[baseline] = {
            "tasks": len(rows),
            "task_success_rate": mean(successes),
            "task_success_ci95": [lower, upper],
            "mean_tool_calls": mean([row["tool_calls"] for row in rows]),
            "mean_retries": mean([row["retries"] for row in rows]),
            "recall_at_3": mean([row["recall_at_3"] for row in rows]),
            "mrr": mean([row["mrr"] for row in rows]),
            "ndcg_at_3": mean([row["ndcg_at_3"] for row in rows]),
            "harmful_transfer_rate": mean([1.0 if row["harmful_transfer"] else 0.0 for row in rows]),
        }
    return summary


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
        "| baseline | success | 95% CI | recall@3 | MRR | nDCG@3 | tool calls | harmful transfer |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for baseline, stats in payload["summary"].items():
        lines.append(
            f"| {baseline} | {stats['task_success_rate']:.3f} | "
            f"[{stats['task_success_ci95'][0]:.3f}, {stats['task_success_ci95'][1]:.3f}] | "
            f"{stats['recall_at_3']:.3f} | {stats['mrr']:.3f} | {stats['ndcg_at_3']:.3f} | "
            f"{stats['mean_tool_calls']:.2f} | {stats['harmful_transfer_rate']:.3f} |"
        )
    lines.extend(["", "## Per-task results", "", "| task | baseline | success | status | error |", "|---|---|---|---|---|"])
    for row in payload["per_task"]:
        lines.append(
            f"| {row['task_id']} | {row['baseline']} | {row['task_success']} | "
            f"{row['retrieval_status']} | {row['error'][:60]} |"
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
