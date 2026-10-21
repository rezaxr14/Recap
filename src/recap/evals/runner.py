"""Benchmark runner: fair baselines over the same tasks, real measured metrics.

Baselines compared (all deterministic, same tool budget):
  - no_memory
  - retry_only
  - bm25
  - vector
  - hybrid_rrf
  - hybrid_rerank (configured reranker; lexical by default)
  - outcome_aware (RECAP selection policy)

Frozen-memory setting: retrieval is read-only and tasks never write memories
during the evaluation, so test memories cannot leak into later test episodes.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from recap.config import ChunkingSettings
from recap.container import Container
from recap.domain.enums import EventType, MemoryStatus, OutcomeStatus, RetrievalMode, RunStatus
from recap.domain.memories import ExperienceMemory
from recap.domain.retrieval import RetrievalContext, RetrievalRequest
from recap.environments.mock_api import PaginatedApiEnvironment, family_error
from recap.environments.reference_agent import ReferenceAgent
from recap.environments.run_recording import (
    AgentRunMetadata,
    complete_recorded_run,
    start_recorded_run,
)
from recap.evals.datasets import (
    FAMILY_SPLITS,
    BenchmarkTask,
    build_benchmark_v2_tasks,
    default_tasks,
    split_hash,
)
from recap.evals.metrics import (
    bootstrap_ci,
    mean,
    ndcg_at_k,
    percentile,
    recall_at_k,
    reciprocal_rank,
)
from recap.logging import get_logger

logger = get_logger(__name__)

BASELINES: list[tuple[str, RetrievalMode | None, str]] = [
    ("no_memory", None, ""),
    ("retry_only", None, ""),
    ("bm25", RetrievalMode.BM25, ""),
    ("vector", RetrievalMode.VECTOR, ""),
    ("hybrid_rrf", RetrievalMode.HYBRID_RRF, ""),
    # A fair, deliberately weak version-aware baseline: run ordinary hybrid
    # retrieval, then discard candidates whose declared environment id differs
    # from the task. It gets the same information as outcome-aware selection
    # but none of the compatibility, applicability, or utility policy.
    ("hybrid_rrf_version_filter", RetrievalMode.HYBRID_RRF, ""),
    ("hybrid_rerank", RetrievalMode.HYBRID_RERANK, ""),
    ("outcome_aware", RetrievalMode.OUTCOME_AWARE, ""),
    # Chunk-granule ablations on the same tasks. Both chunk strategies are
    # indexed side by side, so "naive vs event-aware" differs only in the
    # chunker; the retriever and embedder are identical.
    ("bm25_naive_chunk", RetrievalMode.BM25_CHUNK, "naive_fixed"),
    ("vector_naive_chunk", RetrievalMode.VECTOR_CHUNK, "naive_fixed"),
    ("hybrid_rrf_naive_chunk", RetrievalMode.HYBRID_RRF_CHUNK, "naive_fixed"),
    ("outcome_aware_naive_chunk", RetrievalMode.OUTCOME_AWARE_CHUNK, "naive_fixed"),
    ("bm25_chunk", RetrievalMode.BM25_CHUNK, "event_aware"),
    ("vector_chunk", RetrievalMode.VECTOR_CHUNK, "event_aware"),
    ("hybrid_rrf_chunk", RetrievalMode.HYBRID_RRF_CHUNK, "event_aware"),
    ("outcome_aware_chunk", RetrievalMode.OUTCOME_AWARE_CHUNK, "event_aware"),
]


def run_benchmark(
    container: Container,
    *,
    split: str = "dev",
    suite: str = "legacy",
    tasks: list[BenchmarkTask] | None = None,
    limit: int | None = None,
    output_dir: Path | str = Path("benchmark/reports"),
    workspace_id: str | None = None,
    cleanup: bool = True,
) -> dict[str, Any]:
    container.migrate()
    container.init_search()
    container.init_search_chunks()
    # The dimension probe above is not part of benchmark cost. Reset after it
    # so the report covers only seeding and retrieval in this run.
    _reset_embedding_stats(container.embeddings)
    if suite not in {"legacy", "v2"}:
        raise ValueError("suite must be legacy or v2")
    available = (
        tasks
        if tasks is not None
        else (default_tasks() if suite == "legacy" else build_benchmark_v2_tasks())
    )
    tasks = [task for task in available if task.split == split]
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        tasks = tasks[:limit]
    if not tasks:
        raise ValueError(f"no tasks for split {split} in suite {suite}")

    dataset_hash = split_hash(tasks)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    # Never reuse a workspace across runs: stale memories or chunk documents
    # from an earlier run would compete with the freshly seeded ones and make
    # the summaries drift. An explicit workspace is reset; otherwise each run
    # gets its own name and is torn down afterwards.
    workspace = workspace_id or f"bench-{dataset_hash[:8]}-{stamp.lower()}"
    from recap.maintenance import reset_workspace

    reset_workspace(container, [workspace])
    try:
        _seed_memories(container, workspace, tasks)

        per_task: list[dict[str, Any]] = []
        for baseline, mode, chunk_strategy in BASELINES:
            for task in tasks:
                per_task.append(
                    _run_task(
                        container,
                        workspace,
                        task,
                        baseline=baseline,
                        mode=mode,
                        chunk_strategy=chunk_strategy,
                    )
                )

        summary = _summarize(per_task)
        by_family = _summarize_by_family(per_task)
        performance = _summarize_performance(per_task)
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        json_path = output / f"benchmark_{split}_{stamp}.json"
        csv_path = output / f"benchmark_{split}_{stamp}.csv"
        md_path = output / f"benchmark_{split}_{stamp}.md"
        embedding_usage = _embedding_usage(container.embeddings)
        cache_table = (
            container.embedding_cache.stats() if container.embedding_cache is not None else None
        )
        payload = {
            "generated_at": datetime.now(UTC).isoformat(),
            "split": split,
            "suite": suite,
            "dataset_hash": dataset_hash,
            "workspace_id": workspace,
            "tasks": [task.to_dict() for task in tasks],
            "baselines": [name for name, _, _ in BASELINES],
            "per_task": per_task,
            "summary": summary,
            "by_task_family": by_family,
            "performance": performance,
            "retrieval": _retrieval_provenance(container),
            "embedding": embedding_usage,
            "embedding_cache_table": cache_table,
            "protocol": {
                "memory_setting": "frozen_memory",
                "memory_snapshot": "one per run, created before retrieval",
                "retrieval_scope": "run",
                "agent": "deterministic-reference-agent-v1",
                "provider": "deterministic",
                "environment": "mock-api-failure-families",
                "suite": suite,
                "failure_families": sorted({task.failure_mode for task in tasks}),
                "metrics_source": "persisted_run_records",
                "workspace_isolation": "unique workspace per run, reset before and after",
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
            "workspace_id": workspace,
        }
    finally:
        if cleanup:
            reset_workspace(container, [workspace])


def _seed_memories(container: Container, workspace_id: str, tasks: list[BenchmarkTask]) -> None:
    """Create validated memories and an explicit bad-track-record distractor.

    Memories are generated through the real compile -> validate -> promote path,
    so the benchmark exercises the production promotion gate. The family is part
    of the environment fingerprint, which keeps wrong-family memories available
    as hard negatives without making them applicable to the task.

    One extra family from a different split is seeded and then given two
    verified failures. It remains active (below the degradation thresholds), so
    it is a tempting, compatible-looking distractor with a worse track record.
    This is a benchmark fixture, not a claim that the synthetic failures came
    from production traffic; the report labels it accordingly.
    """
    expected: dict[str, tuple[str, str]] = {}
    for task in tasks:
        environment = PaginatedApiEnvironment(
            version=task.environment_version, failure_family=task.failure_mode
        )
        expected[environment.fingerprint().environment_id] = (
            task.failure_mode,
            task.environment_version,
        )
    with container.services() as services:
        from recap.infrastructure.db.repositories import ensure_workspace

        ensure_workspace(services.session, workspace_id)
        existing = services.memories.list_memories(
            workspace_id, status=MemoryStatus.ACTIVE, limit=100
        )
    seeded = {memory.environment_id for memory in existing if memory.environment_id}

    for environment_id, (family, version) in sorted(expected.items()):
        if environment_id in seeded:
            continue
        _promote_seed_memory(container, workspace_id, family=family, version=version)

    # A bad-track-record hard negative. Use a family held out of this split so
    # lowering its reliability cannot damage any task that is actually scored.
    split = tasks[0].split
    distractor_family = next(
        (family for family, family_split in FAMILY_SPLITS.items() if family_split != split),
        "",
    )
    distractor_environment: PaginatedApiEnvironment | None = None
    if distractor_family:
        distractor_environment = PaginatedApiEnvironment(
            version="2", failure_family=distractor_family
        )
        distractor_fingerprint = distractor_environment.fingerprint()
        if distractor_fingerprint.environment_id not in seeded:
            _promote_seed_memory(container, workspace_id, family=distractor_family, version="2")
        with container.services() as services:
            memory = next(
                (
                    item
                    for item in services.memories.list_memories(
                        workspace_id, status=MemoryStatus.ACTIVE, limit=100
                    )
                    if item.environment_id == distractor_fingerprint.environment_id
                ),
                None,
            )
            if memory is not None and memory.stats.verified_failures < 2:
                for _ in range(2 - memory.stats.verified_failures):
                    services.memory_shift.record_outcome(
                        memory,
                        outcome=OutcomeStatus.VERIFIED_FAILURE,
                        applied=True,
                        environment_id=distractor_fingerprint.environment_id,
                        environment_fingerprint=distractor_fingerprint.fingerprint,
                        task_family=distractor_family,
                        notes="benchmark bad-track-record fixture",
                        evidence={"source": "benchmark_v2_fixture"},
                    )
                    refreshed = services.memories.get_memory(memory.id, workspace_id)
                    if refreshed is None:
                        break
                    memory = refreshed

    # Chunks are derived state: rebuild them for every active memory so an
    # already-seeded benchmark workspace is brought up to date idempotently.
    seed_env_ids = set(expected)
    if distractor_environment is not None:
        seed_env_ids.add(distractor_environment.fingerprint().environment_id)
    with container.services() as services:
        memories = services.memories.list_memories(
            workspace_id, status=MemoryStatus.ACTIVE, limit=100
        )
        for memory in memories:
            if memory.environment_id not in seed_env_ids:
                continue
            # Build both ablation sets into the same workspace/index. Chunk
            # ids include the strategy, so neither rebuild deletes the other.
            for strategy in ("event_aware", "naive_fixed"):
                chunk_report = services.chunk_service.rebuild(
                    memory, config=ChunkingSettings(strategy=strategy)
                )
                if not chunk_report.indexed:
                    logger.warning(
                        "benchmark %s chunk build failed for %s: %s",
                        strategy,
                        memory.id,
                        chunk_report.error,
                    )


def _promote_seed_memory(
    container: Container, workspace_id: str, *, family: str, version: str
) -> ExperienceMemory | None:
    """Run one real failure, compile a candidate, validate it, and promote it."""
    from recap.environments.run_recording import record_reference_run

    environment = PaginatedApiEnvironment(version=version, failure_family=family)
    fingerprint = environment.fingerprint()
    run_id, _ = record_reference_run(
        container,
        environment,
        workspace_id=workspace_id,
        task_id=f"seed-{family}-v{version}",
        failure_family=family,
    )
    with container.services() as services:
        services.environments.upsert(workspace_id, fingerprint)
        candidates = services.repair_compiler.compile(run_id, environment=fingerprint)
        if not candidates:
            logger.warning("benchmark seed produced no candidate for %s v%s", family, version)
            return None
        candidate_id = candidates[0].id
        result = services.repair_compiler.validate(
            candidate_id, environment=fingerprint, validator_name="mock_sandbox"
        )
        if result.status.value != "passed":
            logger.warning("benchmark seed validation failed for %s v%s", family, version)
            return None
        refreshed = services.repairs.get(candidate_id)
        assert refreshed is not None
        return services.promotion.promote(refreshed)


def _run_task(
    container: Container,
    workspace_id: str,
    task: BenchmarkTask,
    *,
    baseline: str,
    mode: RetrievalMode | None,
    chunk_strategy: str = "",
) -> dict[str, Any]:
    environment = PaginatedApiEnvironment(
        version=task.environment_version, failure_family=task.failure_mode
    )
    fingerprint = environment.fingerprint()
    agent = ReferenceAgent()
    failure = family_error(task.failure_mode)

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
            "chunk_strategy": chunk_strategy or None,
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
    error_code = failure["error_code"]
    error_message = failure["error_message"]
    error_type = failure["error_type"]
    request: RetrievalRequest | None = None
    decision = None
    if mode is not None:
        with container.services() as services:
            request = RetrievalRequest(
                context=RetrievalContext(
                    workspace_id=workspace_id,
                    query=task.instruction,
                    task_family=task.failure_mode,
                    environment_id=fingerprint.environment_id,
                    environment_fingerprint=fingerprint.fingerprint,
                    environment_version=fingerprint.environment_version,
                    environment_type=fingerprint.environment_type,
                    tool_names=["list_records"],
                    tool_versions=fingerprint.tool_versions,
                    tool_schema_fingerprints=fingerprint.tool_schema_hashes,
                    error_type=error_type,
                    error_code=error_code,
                    error_message=error_message,
                    state={"tool_schema_fingerprints": fingerprint.tool_schema_hashes},
                ),
                mode=mode,
                chunk_strategy=chunk_strategy or "event_aware",
                final_k=3,
                run_id=run.id,
            )
            decision = services.recall.retrieve(request)
            status = decision.status.value
            ranked_ids = [candidate.memory_id for candidate in decision.candidates]
            selected_ids = [memory.memory_id for memory in decision.selected_memories]
            if baseline == "hybrid_rrf_version_filter":
                allowed = {
                    memory.id
                    for memory in services.memories.list_memories(
                        workspace_id, status=MemoryStatus.ACTIVE, limit=100
                    )
                    if memory.environment_id == fingerprint.environment_id
                }
                ranked_ids = [memory_id for memory_id in ranked_ids if memory_id in allowed]
                selected_ids = [memory_id for memory_id in selected_ids if memory_id in allowed]
                status = "selected" if selected_ids else "no_safe_memory"

    memory = None
    if baseline == "retry_only":
        # A naive retry with the same pre-memory request cannot succeed.
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
    wrong_selected = [memory_id for memory_id in selected_ids if memory_id not in relevant]
    first_relevant_index = next(
        (index for index, memory_id in enumerate(selected_ids) if memory_id in relevant), None
    )
    first_wrong_index = next(
        (index for index, memory_id in enumerate(selected_ids) if memory_id not in relevant), None
    )
    hard_negative_outranked_correct = first_wrong_index is not None and (
        first_relevant_index is None or first_wrong_index < first_relevant_index
    )
    false_abstention = mode is RetrievalMode.OUTCOME_AWARE and bool(relevant) and not selected_ids
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
        "failure_family": task.failure_mode,
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
        "hard_negative_selected": bool(wrong_selected),
        "hard_negative_outranked_correct": hard_negative_outranked_correct,
        "false_abstention": false_abstention,
        "retrieval_latency_ms": (
            round(decision.total_latency_ms, 3) if decision is not None else None
        ),
        "retrieval_stage_latency_ms": (
            {key: round(value, 3) for key, value in decision.stage_latency_ms.items()}
            if decision is not None
            else {}
        ),
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

    environment = PaginatedApiEnvironment(
        version=task.environment_version, failure_family=task.failure_mode
    )
    fingerprint = environment.fingerprint()
    context = RetrievalContext(
        workspace_id=workspace_id,
        query=task.instruction,
        task_family=task.failure_mode,
        environment_id=fingerprint.environment_id,
        environment_fingerprint=fingerprint.fingerprint,
        environment_version=fingerprint.environment_version,
        environment_type=fingerprint.environment_type,
        tool_names=["list_records"],
        tool_versions=fingerprint.tool_versions,
        tool_schema_fingerprints=fingerprint.tool_schema_hashes,
        error_type=family_error(task.failure_mode)["error_type"],
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
    environment = PaginatedApiEnvironment(
        version=task.environment_version, failure_family=task.failure_mode
    )
    expected_environment = environment.fingerprint().environment_id
    with container.services() as services:
        memories = services.memories.list_memories(
            workspace_id, status=MemoryStatus.ACTIVE, limit=100
        )
    return {memory.id for memory in memories if memory.environment_id == expected_environment}


def _reset_embedding_stats(provider: Any) -> None:
    reset = getattr(provider, "reset_stats", None)
    if callable(reset):
        reset()


def _embedding_usage(provider: Any) -> dict[str, Any] | None:
    usage = getattr(provider, "embedding_usage", None)
    if not callable(usage):
        return None
    result = usage()
    return dict(result) if isinstance(result, dict) else None


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
        "hard_negative_selection_rate": mean(
            [1.0 if row["hard_negative_selected"] else 0.0 for row in rows]
        ),
        "hard_negative_outrank_rate": mean(
            [1.0 if row["hard_negative_outranked_correct"] else 0.0 for row in rows]
        ),
        "false_abstention_rate": mean([1.0 if row["false_abstention"] else 0.0 for row in rows]),
        "total_incompatible_selected": sum(row["incompatible_selected"] for row in rows),
    }


def _retrieval_provenance(container: Container) -> dict[str, Any]:
    """Configuration that changes retrieval results, recorded beside the report.

    The benchmark report already records the embedding provider. This adds the
    reranker identity so two runs with the same metrics can still be told apart
    (for example lexical vs. a local CrossEncoder).
    """
    recall = container.settings.recall
    with container.services() as services:
        reranker = getattr(services.recall, "reranker", None)
    return {
        "default_mode": recall.default_mode,
        "configured_reranker": recall.reranker,
        "reranker": getattr(reranker, "name", "none"),
        "reranker_model_path": recall.rerank_model_path,
        "rerank_max_length": recall.rerank_max_length,
        "rerank_batch_size": recall.rerank_batch_size,
        "embedding_model": getattr(container.embeddings, "model_id", ""),
        "embedding_fingerprint": getattr(container.embeddings, "embedding_fingerprint", ""),
        "vector_dimension": container.settings.elasticsearch.vector_dimension,
        "configured_index_generation": container.settings.elasticsearch.index_generation,
        "active_index": container.index_manager.active_index_name(),
    }


def _summarize_performance(per_task: list[dict[str, Any]]) -> dict[str, Any]:
    """Latency is reported separately because it is not reproducible.

    Keeping it out of ``summary`` preserves the CI reproducibility gate, which
    must compare quality/safety metrics and not machine speed.
    """
    baselines = sorted({row["baseline"] for row in per_task})
    result: dict[str, Any] = {}
    for baseline in baselines:
        rows = [row for row in per_task if row["baseline"] == baseline]
        latencies = [
            float(row["retrieval_latency_ms"])
            for row in rows
            if row.get("retrieval_latency_ms") is not None
        ]
        stages: dict[str, list[float]] = {}
        for row in rows:
            for stage, value in (row.get("retrieval_stage_latency_ms") or {}).items():
                stages.setdefault(stage, []).append(float(value))
        result[baseline] = {
            "retrieval_tasks": len(latencies),
            "retrieval_latency_p50_ms": (
                round(percentile(latencies, 50.0), 3) if latencies else None
            ),
            "retrieval_latency_p95_ms": (
                round(percentile(latencies, 95.0), 3) if latencies else None
            ),
            "stage_latency_p50_ms": {
                stage: round(percentile(values, 50.0), 3)
                for stage, values in sorted(stages.items())
            },
            "stage_latency_p95_ms": {
                stage: round(percentile(values, 95.0), 3)
                for stage, values in sorted(stages.items())
            },
        }
    return result


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
        f"# RECAP benchmark - suite `{payload.get('suite', 'legacy')}` split `{payload['split']}`",
        "",
        f"Generated: {payload['generated_at']}",
        f"Dataset hash: `{payload['dataset_hash']}`",
        "",
        "> Deterministic mock environment and deterministic reference agent. These numbers "
        "measure pipeline behavior, not real LLM agent performance.",
        "",
        "## Summary",
        "",
        "| baseline | success | 95% CI | recall@3 | MRR | nDCG@3 | tool calls | harmful transfer | incompatible selection | hard-neg selected | hard-neg outrank | false abstention |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for baseline, stats in payload["summary"].items():
        lines.append(
            f"| {baseline} | {stats['task_success_rate']:.3f} | "
            f"[{stats['task_success_ci95'][0]:.3f}, {stats['task_success_ci95'][1]:.3f}] | "
            f"{stats['recall_at_3']:.3f} | {stats['mrr']:.3f} | {stats['ndcg_at_3']:.3f} | "
            f"{stats['mean_tool_calls']:.2f} | {stats['harmful_transfer_rate']:.3f} | "
            f"{stats['incompatible_selection_rate']:.3f} | "
            f"{stats['hard_negative_selection_rate']:.3f} | "
            f"{stats['hard_negative_outrank_rate']:.3f} | "
            f"{stats['false_abstention_rate']:.3f} |"
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
                "| family | baseline | success | recall@3 | MRR | harmful transfer | incompatible selection | hard-neg outrank | false abstention |",
                "|---|---|---|---|---|---|---|---|---|",
            ]
        )
        for family, baselines in by_family.items():
            for baseline, stats in baselines.items():
                lines.append(
                    f"| {family} | {baseline} | {stats['task_success_rate']:.3f} | "
                    f"{stats['recall_at_3']:.3f} | {stats['mrr']:.3f} | "
                    f"{stats['harmful_transfer_rate']:.3f} | "
                    f"{stats['incompatible_selection_rate']:.3f} | "
                    f"{stats['hard_negative_outrank_rate']:.3f} | "
                    f"{stats['false_abstention_rate']:.3f} |"
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
    performance = payload.get("performance") or {}
    if performance:
        lines.extend(
            [
                "",
                "## Retrieval latency (non-reproducible)",
                "",
                "| baseline | measured tasks | total p50 ms | total p95 ms | rerank p50 ms | rerank p95 ms |",
                "|---|---|---|---|---|---|",
            ]
        )
        for baseline, stats in performance.items():
            p50 = stats.get("retrieval_latency_p50_ms")
            p95 = stats.get("retrieval_latency_p95_ms")
            rerank_p50 = (stats.get("stage_latency_p50_ms") or {}).get("rerank")
            rerank_p95 = (stats.get("stage_latency_p95_ms") or {}).get("rerank")
            p50_text = "n/a" if p50 is None else f"{p50:.3f}"
            p95_text = "n/a" if p95 is None else f"{p95:.3f}"
            rerank_p50_text = "n/a" if rerank_p50 is None else f"{rerank_p50:.3f}"
            rerank_p95_text = "n/a" if rerank_p95 is None else f"{rerank_p95:.3f}"
            lines.append(
                f"| {baseline} | {stats.get('retrieval_tasks', 0)} | {p50_text} | "
                f"{p95_text} | {rerank_p50_text} | {rerank_p95_text} |"
            )

    retrieval = payload.get("retrieval")
    if isinstance(retrieval, dict):
        lines.extend(
            [
                "",
                "## Retrieval configuration",
                "",
                f"- default mode: `{retrieval.get('default_mode', '')}`",
                f"- configured reranker: `{retrieval.get('configured_reranker', '')}`",
                f"- active reranker: `{retrieval.get('reranker', '')}`",
                f"- reranker model path: `{retrieval.get('reranker_model_path', '')}`",
                f"- embedding model: `{retrieval.get('embedding_model', '')}`",
                f"- vector dimension: {retrieval.get('vector_dimension', 0)}",
                f"- active index: `{retrieval.get('active_index', '')}`",
            ]
        )

    embedding = payload.get("embedding")
    if isinstance(embedding, dict):
        prompt_tokens = embedding.get("prompt_tokens")
        lines.extend(
            [
                "",
                "## Embedding usage",
                "",
                f"- provider: `{embedding.get('provider', 'unknown')}`",
                f"- model: `{embedding.get('model', '')}`",
                f"- dimension: {embedding.get('dimension', 0)}",
                f"- cache hits / misses: {embedding.get('hits', 0)} / "
                f"{embedding.get('misses', 0)} "
                f"(hit rate {embedding.get('hit_rate', 0.0):.3f})",
                f"- provider batches / texts: {embedding.get('embedded_batches', 0)} / "
                f"{embedding.get('embedded_texts', 0)}",
                f"- provider latency: {embedding.get('provider_latency_ms', 0.0):.3f} ms",
                f"- prompt tokens: {'not reported' if prompt_tokens is None else prompt_tokens}",
            ]
        )
    lines.extend(
        [
            "",
            "## Reproduction",
            "",
            "```bash",
            f"recap eval run --suite {payload.get('suite', 'legacy')} --split {payload['split']}",
            "```",
        ]
    )
    return "\n".join(lines)
