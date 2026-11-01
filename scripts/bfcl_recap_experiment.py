"""RECAP-stack experiment on real BFCL v3: does selection add value over relevance?

This is the Task 2 experiment from ``AGENT_PROMPT.md``. Unlike
``bfcl_memory_transfer.py`` (an Elasticsearch retrieval proxy), every memory
here is produced by RECAP's own stack:

    RunRecorder -> failure window -> SchemaRenameGenerator -> RepairCompiler
    -> BFCLSandboxValidator -> PromotionService -> RecallEngine

All queries, function schemas, and gold arguments are real BFCL v3 data. The
only injected part is the failure: one required parameter is renamed on the real
schema. The experiment measures four conditions:

* ``in_pool``: every function's repair memory is in the pool; the target is known.
* ``held_out``: the query function's memory is excluded through a real
  ``MemorySnapshot`` (as if that memory did not exist when the run started).
* ``sibling``: the query function's memory is excluded, so retrieval can only
  reach memories for same-name-family functions with different parameters.
* ``drift``: two schema versions inside one environment id. A stale memory is
  first the only memory, then superseded by a second promotion.

Arms: no memory, retry, BM25, dense, hybrid RRF, typed hybrid, version-filtered
hybrid, fingerprint-filtered hybrid, keyed by tool name, keyed by schema hash,
and the full RECAP policy (outcome-aware).

Data (public, one download):
    python -c "from huggingface_hub import hf_hub_download as d; \\
      [d('gorilla-llm/Berkeley-Function-Calling-Leaderboard', f, repo_type='dataset', \\
         local_dir='data/bfcl') for f in ('BFCL_v3_simple.json', \\
         'possible_answer/BFCL_v3_simple.json', 'BFCL_v3_exec_simple.json')]"

Run (services must be up; see docker compose):
    python scripts/bfcl_recap_experiment.py --embedding hashing
    python scripts/bfcl_recap_experiment.py --embedding qwen \\
        --model-path /path/to/Qwen3-Embedding-0.6B

The report is written to ``docs/evidence/real-data-bfcl-recap-<embedding>.json``
and a Markdown view is generated from that JSON. Numbers are never hand-edited.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from recap.config import EmbeddingSettings, Settings, load_settings
from recap.container import Container
from recap.domain.enums import EventType, MemoryStatus, RetrievalMode, RunStatus
from recap.domain.retrieval import RetrievalContext, RetrievalRequest
from recap.domain.runs import AgentRun, RunEventCreate
from recap.environments.bfcl import BFCLEnvironment
from recap.evals.bfcl import (
    apply_rename,
    build_transfer_scenarios,
    drift_schema,
    is_valid_call,
    load_tasks,
    sibling_pairs,
)
from recap.maintenance import reset_workspace
from recap.recorder.normalization import normalize_signature
from recap.repair_compiler.bfcl import BFCLSandboxValidator, SchemaRenameGenerator

DEFAULT_DATA = Path("data/bfcl")
DEFAULT_OUT = Path("docs/evidence")
DEFAULT_PREFIX = "recap-bfcl-recap"
QWEN_QUERY_TASK = "Given a user request, retrieve a previous repair for the same tool"
BOOTSTRAP = 2000
RNG_SEED = 20261101
UTILITY_THRESHOLD = 0.35
AGENT_ID = "bfcl-recap-experiment"

# (name, mode, context kind). ``context kind`` is "pure" (query only) or
# "typed" (tool name, schema fingerprint, error code). Keyed arms bypass search.
ARMS: list[tuple[str, RetrievalMode | None, str]] = [
    ("no_memory", None, "none"),
    ("retry_only", None, "none"),
    ("bm25", RetrievalMode.BM25, "pure"),
    ("vector", RetrievalMode.VECTOR, "pure"),
    ("hybrid_rrf", RetrievalMode.HYBRID_RRF, "pure"),
    ("rrf_typed", RetrievalMode.HYBRID_RRF, "typed"),
    ("rrf_version_filter", RetrievalMode.HYBRID_RRF, "typed"),
    ("fingerprint_filter", RetrievalMode.HYBRID_RRF, "typed"),
    ("function_keyed", None, "keyed"),
    ("schema_hash", None, "hash"),
    ("full_recap", RetrievalMode.OUTCOME_AWARE, "typed"),
]

SAFE_CONDITIONS = ("in_pool", "drift_phase1", "drift_phase3")
NO_SAFE_CONDITIONS = ("held_out", "sibling", "drift_phase2")


# --------------------------------------------------------------------------- pure
def memory_actions(memory: Any) -> list[dict[str, Any]]:
    return [action.model_dump(mode="json") for action in memory.content.strategy.actions]


def apply_actions(args: dict[str, Any], actions: list[dict[str, Any]]) -> dict[str, Any]:
    """Apply the typed strategy actions a memory asks the host agent to perform."""
    repaired = dict(args)
    for action in actions:
        action_type = str(action.get("action_type") or "")
        parameters = action.get("parameters") or {}
        if action_type == "map_parameter":
            old = str(parameters.get("from") or "")
            new = str(parameters.get("to") or "")
            if old and new and old in repaired:
                repaired[new] = repaired.pop(old)
        elif action_type == "supply_required_parameter":
            for key, value in parameters.items():
                repaired.setdefault(str(key), value)
    return repaired


def evaluate_row(
    *,
    condition: str,
    scenario: str,
    arm: str,
    has_safe_memory: bool,
    schema: dict[str, Any],
    args: dict[str, Any],
    target_function: str,
    result: SelectionResult,
    current_memory_id: str | None = None,
    stale_memory_id: str | None = None,
) -> dict[str, Any]:
    """Classify one (condition, scenario, arm) outcome.

    ``false_apply`` is a policy false positive: the condition has no memory that
    declares the current tool/schema, yet the arm selected one. ``harmful`` is
    the subset where the resulting call is invalid. ``valid_transfer`` is the
    subset where a cross-function memory happened to repair the call anyway; it
    is the cost of conservative abstention, not a policy success.
    """
    selected = result.selection
    repaired = apply_actions(args, selected.actions) if selected is not None else dict(args)
    call_valid = is_valid_call(schema, repaired)
    selected_tool = selected.tool if selected is not None else ""
    same_tool = bool(selected is not None and selected_tool == target_function)
    hit_at_1 = bool(
        selected is not None
        and current_memory_id is not None
        and selected.memory_id == current_memory_id
    )
    return {
        "condition": condition,
        "scenario": scenario,
        "cluster": target_function,
        "arm": arm,
        "has_safe_memory": has_safe_memory,
        "selected": selected is not None,
        "selected_memory_id": selected.memory_id if selected is not None else None,
        "selected_tool": selected_tool,
        "selected_version": selected.version if selected is not None else None,
        "selected_is_target": same_tool,
        "selected_is_current": bool(
            selected is not None
            and current_memory_id is not None
            and selected.memory_id == current_memory_id
        ),
        "stale_selected": bool(
            selected is not None
            and stale_memory_id is not None
            and selected.memory_id == stale_memory_id
        ),
        "call_valid": call_valid,
        "abstained": selected is None,
        "false_apply": bool(selected is not None and not has_safe_memory),
        "harmful_transfer": bool(selected is not None and not same_tool and not call_valid),
        "valid_transfer": bool(selected is not None and not same_tool and call_valid),
        "candidate_count": result.candidate_count,
        "eligible_count": result.eligible_count,
        "applicable_count": result.applicable_count,
        "retrieval_hit_at_1": hit_at_1,
        "status": selected.status if selected is not None else "no_selection",
    }


@dataclass
class Selection:
    memory_id: str
    tool: str
    version: int
    actions: list[dict[str, Any]]
    status: str = "selected"


@dataclass
class SelectionResult:
    selection: Selection | None
    candidate_count: int = 0
    eligible_count: int = 0
    applicable_count: int = 0
    notes: list[str] = field(default_factory=list)


def selection_from_decision(decision: Any) -> SelectionResult:
    candidates = list(decision.candidates or [])
    eligible = sum(1 for candidate in candidates if candidate.hard_filter_passed)
    applicable = sum(
        1 for candidate in candidates if str(candidate.applicability_result) == "satisfied"
    )
    if not decision.selected_memories:
        return SelectionResult(
            None,
            candidate_count=len(candidates),
            eligible_count=eligible,
            applicable_count=applicable,
        )
    selected = decision.selected_memories[0]
    return SelectionResult(
        Selection(
            memory_id=selected.memory_id,
            tool=str(selected.actions[0].get("tool_name") or "") if selected.actions else "",
            version=selected.memory_version,
            actions=list(selected.actions),
            status=str(decision.status),
        ),
        candidate_count=len(candidates),
        eligible_count=eligible,
        applicable_count=applicable,
    )


def abstention_stats(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Abstention precision/recall where the positive class is "should abstain".

    TP: abstained on a no-safe row. FP: abstained on a safe row (unnecessary).
    FN: selected on a no-safe row (missed). Precision and recall are undefined
    when the denominator is zero, so they are reported as 0.0 rather than 1.0.
    """
    result: dict[str, dict[str, Any]] = {}
    for arm, _, _ in ARMS:
        safe = [row for row in rows if row["condition"] in SAFE_CONDITIONS and row["arm"] == arm]
        no_safe = [
            row for row in rows if row["condition"] in NO_SAFE_CONDITIONS and row["arm"] == arm
        ]
        tp = sum(1 for row in no_safe if row["abstained"])
        fp = sum(1 for row in safe if row["abstained"])
        fn = sum(1 for row in no_safe if not row["abstained"])
        result[arm] = {
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "false_abstain_rate": fp / len(safe) if safe else 0.0,
            "missed_abstentions": fn,
            "unneeded_abstentions": fp,
            "n_no_safe": len(no_safe),
            "n_safe": len(safe),
        }
    return result


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def rate(key: str) -> float:
        return float(np.mean([1.0 if row[key] else 0.0 for row in rows])) if rows else 0.0

    return {
        "n": len(rows),
        "call_valid_rate": rate("call_valid"),
        "selected_rate": rate("selected"),
        "abstain_rate": rate("abstained"),
        "correct_memory_rate": rate("selected_is_current"),
        "false_apply_rate": rate("false_apply"),
        "harmful_transfer_rate": rate("harmful_transfer"),
        "valid_transfer_rate": rate("valid_transfer"),
        "stale_selection_rate": rate("stale_selected"),
        "retrieval_hit_at_1": rate("retrieval_hit_at_1"),
    }


def cluster_bootstrap(
    values_by_cluster: dict[str, list[float]],
    rng: np.random.Generator,
    *,
    resamples: int = BOOTSTRAP,
) -> dict[str, Any]:
    """Mean with a cluster bootstrap CI; clusters are functions, not rows."""
    clusters = [key for key, values in values_by_cluster.items() if values]
    pooled = [value for key in clusters for value in values_by_cluster[key]]
    observed = float(np.mean(pooled)) if pooled else 0.0
    stats: list[float] = []
    for _ in range(resamples):
        picked = rng.integers(0, len(clusters), size=len(clusters))
        sample = [value for index in picked for value in values_by_cluster[clusters[index]]]
        stats.append(float(np.mean(sample)) if sample else 0.0)
    lower, upper = (
        (float(np.quantile(stats, 0.025)), float(np.quantile(stats, 0.975)))
        if stats
        else (observed, observed)
    )
    return {
        "n_clusters": len(clusters),
        "n_rows": len(pooled),
        "observed": observed,
        "cluster_ci95": [lower, upper],
    }


def paired_bootstrap(
    treatment: list[dict[str, Any]],
    reference: list[dict[str, Any]],
    *,
    cluster_key: str,
    value_key: str,
    rng: np.random.Generator,
    resamples: int = BOOTSTRAP,
) -> dict[str, Any]:
    """Paired cluster bootstrap over the same (condition, scenario) keys."""
    reference_map = {(row["condition"], row["scenario"]): row[value_key] for row in reference}
    by_cluster: dict[str, list[float]] = defaultdict(list)
    for row in treatment:
        key = (row["condition"], row["scenario"])
        if key not in reference_map:
            continue
        delta = float(row[value_key]) - float(reference_map[key])
        by_cluster[str(row[cluster_key])].append(delta)
    result = cluster_bootstrap(by_cluster, rng, resamples=resamples)
    result["excludes_zero"] = bool(
        result["cluster_ci95"][0] > 0.0 or result["cluster_ci95"][1] < 0.0
    )
    return result


# ------------------------------------------------------------------------ stack
def build_settings(args: argparse.Namespace) -> Settings:
    settings = load_settings()
    settings.storage.postgres_url = args.database_url
    settings.elasticsearch.url = args.es_url
    settings.elasticsearch.index_prefix = args.index_prefix
    settings.elasticsearch.index_generation = 1
    settings.elasticsearch.chunks_index_generation = 2
    settings.elasticsearch.memories_read_alias = f"{args.index_prefix}-memories-read"
    settings.elasticsearch.memories_write_alias = f"{args.index_prefix}-memories-write"
    settings.elasticsearch.chunks_read_alias = f"{args.index_prefix}-chunks-read"
    settings.elasticsearch.chunks_write_alias = f"{args.index_prefix}-chunks-write"
    settings.elasticsearch.identifier_analysis = False
    settings.embeddings.cache_enabled = False
    settings.repair_compiler.generator = "deterministic"
    settings.recall.require_tool_match = False
    if args.embedding == "qwen":
        dimension = args.dimension or 1024
        settings.embeddings = EmbeddingSettings(
            provider="sentence-transformers",
            model_path=str(args.model_path),
            dimension=dimension,
            query_prefix=f"Instruct: {QWEN_QUERY_TASK}\nQuery: ",
            cache_enabled=False,
        )
    else:
        dimension = args.dimension or 256
        settings.embeddings = EmbeddingSettings(
            provider="hashing", dimension=dimension, cache_enabled=False
        )
    settings.elasticsearch.vector_dimension = dimension
    settings.validate_consistency()
    return settings


def record_failure_run(
    container: Container,
    workspace_id: str,
    environment: BFCLEnvironment,
    failed_args: dict[str, Any],
    repaired_args: dict[str, Any],
    *,
    task_id: str,
) -> str:
    """Record a failed call and the observed repair through RunRecorder."""
    fingerprint = environment.fingerprint()
    with container.services() as services:
        from recap.infrastructure.db.repositories import ensure_workspace

        ensure_workspace(services.session, workspace_id)
        run = AgentRun(
            workspace_id=workspace_id,
            agent_id=AGENT_ID,
            task_id=task_id,
            task_family="",
            environment_id=fingerprint.environment_id,
            environment_fingerprint=fingerprint.fingerprint,
            toolset_fingerprint=fingerprint.fingerprint,
            model_provider="schema",
            model_name="bfcl-schema-adapter",
            success_criteria="call is valid against the current schema",
            config_snapshot={
                "adapter": "recap.environments.bfcl.BFCLEnvironment",
                "injected_failure": "required_parameter_renamed",
            },
        )
        run = services.recorder.start_run(run)
        sequence = 0

        def append(event_type: EventType, payload: dict[str, Any], tool: str | None = None) -> None:
            nonlocal sequence
            sequence += 1
            services.recorder.append_event(
                run,
                RunEventCreate(
                    sequence_number=sequence,
                    event_type=event_type,
                    tool_name=tool,
                    tool_version=environment.version if tool else None,
                    payload=payload,
                ),
            )

        append(
            EventType.RUN_STARTED,
            {
                "tool": environment.name,
                "environment_version": environment.version,
                "memory_snapshot_id": run.memory_snapshot_id,
            },
        )
        append(
            EventType.TASK_OBSERVATION,
            {"observation": f"tool={environment.name} args={sorted(failed_args)}"},
        )
        failed = environment.call(failed_args)
        append(EventType.TOOL_CALL_REQUESTED, {"request": failed_args}, tool=environment.name)
        append(
            EventType.TOOL_CALL_FAILED,
            {"request": failed_args, **(failed.error or {})},
            tool=environment.name,
        )
        repaired = environment.call(repaired_args)
        append(EventType.TOOL_CALL_REQUESTED, {"request": repaired_args}, tool=environment.name)
        if repaired.ok:
            append(
                EventType.TOOL_CALL_SUCCEEDED,
                {"result": repaired.payload},
                tool=environment.name,
            )
        else:
            append(
                EventType.TOOL_CALL_FAILED,
                {"request": repaired_args, **(repaired.error or {})},
                tool=environment.name,
            )
        append(
            EventType.POSTCONDITION_CHECKED,
            {"passed": repaired.ok, "detail": None if repaired.ok else "repaired call failed"},
        )
        append(
            EventType.RUN_FINISHED,
            {"success": repaired.ok, "error": None if repaired.ok else "repaired call failed"},
        )
        services.recorder.finish_run(
            run,
            status=RunStatus.SUCCEEDED if repaired.ok else RunStatus.FAILED,
            tool_calls_count=2,
            retry_count=1 if repaired.ok else 0,
        )
        return run.id


def compile_and_promote(
    container: Container,
    workspace_id: str,
    run_id: str,
    environment: BFCLEnvironment,
    schema: dict[str, Any],
) -> tuple[Any | None, dict[str, Any]]:
    """Compile, sandbox-validate, and promote one repair through the real stack."""
    fingerprint = environment.fingerprint()
    container.validators.register(BFCLSandboxValidator({environment.environment_id: environment}))
    with container.services() as services:
        services.environments.upsert(workspace_id, fingerprint)
        services.repair_compiler.generator = SchemaRenameGenerator(
            {environment.environment_id: schema}
        )
        candidates = services.repair_compiler.compile(run_id, environment=fingerprint)
        if not candidates:
            return None, {"stage": "no_candidate"}
        candidate = candidates[0]
        result = services.repair_compiler.validate(
            candidate.id, environment=fingerprint, validator_name="bfcl_sandbox"
        )
        if result.status.value != "passed":
            return None, {
                "stage": f"validation_{result.status.value}",
                "detail": result.detail,
                "candidate_id": candidate.id,
            }
        refreshed = services.repairs.get(candidate.id)
        if refreshed is None:
            return None, {"stage": "candidate_disappeared", "candidate_id": candidate.id}
        memory = services.promotion.promote(refreshed)
        return memory, {
            "stage": "promoted",
            "candidate_id": candidate.id,
            "validation": result.status.value,
            "memory_id": memory.id,
            "memory_version": memory.memory_version,
            "supersedes": memory.supersedes_memory_id,
        }


def start_evaluation_run(
    container: Container,
    workspace_id: str,
    environment: BFCLEnvironment,
    *,
    exclude_memory_ids: set[str],
    task_id: str,
    reason: str,
) -> tuple[str, str, set[str]]:
    fingerprint = environment.fingerprint()
    with container.services() as services:
        from recap.infrastructure.db.repositories import ensure_workspace

        ensure_workspace(services.session, workspace_id)
        snapshot = services.snapshot_service.create(
            workspace_id, reason=reason, exclude_memory_ids=exclude_memory_ids
        )
        allowed = {
            entry.memory_id for entry in snapshot.entries if entry.status is MemoryStatus.ACTIVE
        }
        services.environments.upsert(workspace_id, fingerprint)
        run = AgentRun(
            workspace_id=workspace_id,
            agent_id=AGENT_ID,
            task_id=task_id,
            task_family="",
            environment_id=fingerprint.environment_id,
            environment_fingerprint=fingerprint.fingerprint,
            toolset_fingerprint=fingerprint.fingerprint,
            model_provider="schema",
            model_name="bfcl-schema-adapter",
            success_criteria="select a memory only when its declared schema matches",
            memory_snapshot_id=snapshot.id,
            config_snapshot={
                "reason": reason,
                "excluded_memory_ids": sorted(exclude_memory_ids),
                "policy_threshold": UTILITY_THRESHOLD,
            },
        )
        run = services.recorder.start_run(run)
        services.recorder.append_event(
            run,
            RunEventCreate(
                sequence_number=1,
                event_type=EventType.RUN_STARTED,
                payload={
                    "tool": environment.name,
                    "memory_snapshot_id": snapshot.id,
                    "excluded": sorted(exclude_memory_ids),
                },
            ),
        )
        return run.id, snapshot.id, allowed


def finish_evaluation_run(container: Container, run_id: str, success: bool) -> None:
    with container.services() as services:
        run = services.recorder.get_run(run_id)
        services.recorder.append_event(
            run,
            RunEventCreate(
                sequence_number=2,
                event_type=EventType.POSTCONDITION_CHECKED,
                payload={"passed": success, "detail": "selection experiment"},
            ),
        )
        services.recorder.append_event(
            run,
            RunEventCreate(
                sequence_number=3,
                event_type=EventType.RUN_FINISHED,
                payload={"success": success},
            ),
        )
        services.recorder.finish_run(
            run, status=RunStatus.SUCCEEDED if success else RunStatus.FAILED
        )


def retrieve(container: Container, request: RetrievalRequest) -> Any:
    with container.services() as services:
        return services.recall.retrieve(request)


def active_memories(container: Container, workspace_id: str) -> dict[str, Any]:
    with container.services() as services:
        memories = services.memories.list_memories(
            workspace_id, status=MemoryStatus.ACTIVE, limit=10_000
        )
    return {memory.id: memory for memory in memories}


def keyed_selection(
    function: str,
    *,
    memories: dict[str, Any],
    allowed: set[str],
    by_hash: str = "",
) -> SelectionResult:
    ordered = sorted(memories.values(), key=lambda item: item.created_at, reverse=True)
    considered = [memory for memory in ordered if memory.id in allowed]
    matches: list[Any] = []
    for memory in considered:
        if by_hash:
            if memory.content.tool_schema_fingerprints.get(function) != by_hash:
                continue
        elif function not in memory.content.tool_names:
            continue
        matches.append(memory)
    if not matches:
        return SelectionResult(
            None,
            candidate_count=len(considered),
            eligible_count=0,
        )
    memory = matches[0]
    return SelectionResult(
        Selection(
            memory_id=memory.id,
            tool=function,
            version=memory.memory_version,
            actions=memory_actions(memory),
            status="keyed",
        ),
        candidate_count=len(considered),
        eligible_count=len(matches),
        applicable_count=len(matches),
    )


def filtered_selection(
    decision: Any,
    *,
    memories: dict[str, Any],
    function: str,
    by_fingerprint: str = "",
) -> SelectionResult:
    """Post-filter an engine ranking by environment id or fingerprint."""
    for candidate in decision.candidates:
        if not candidate.hard_filter_passed:
            continue
        memory = memories.get(candidate.memory_id)
        if memory is None:
            continue
        if by_fingerprint:
            if memory.environment_fingerprint != by_fingerprint:
                continue
        elif memory.environment_id != f"bfcl:{function}:v2":
            continue
        return SelectionResult(
            Selection(
                memory_id=memory.id,
                tool=function
                if function in memory.content.tool_names
                else (memory.content.tool_names[0] if memory.content.tool_names else ""),
                version=memory.memory_version,
                actions=memory_actions(memory),
                status="filtered",
            ),
            candidate_count=len(decision.candidates),
            eligible_count=sum(1 for item in decision.candidates if item.hard_filter_passed),
            applicable_count=sum(
                1 for item in decision.candidates if str(item.applicability_result) == "satisfied"
            ),
        )
    return SelectionResult(
        None,
        candidate_count=len(decision.candidates),
        eligible_count=sum(1 for item in decision.candidates if item.hard_filter_passed),
        applicable_count=sum(
            1 for item in decision.candidates if str(item.applicability_result) == "satisfied"
        ),
    )


def typed_context(
    workspace_id: str,
    environment: BFCLEnvironment,
    query: str,
    args: dict[str, Any],
) -> RetrievalContext:
    failed = environment.call(args)
    error = failed.error or {}
    message = str(error.get("error_message") or "")
    error_type = str(error.get("error_type") or "")
    error_code = str(error.get("error_code") or "")
    return RetrievalContext(
        workspace_id=workspace_id,
        query=query,
        task_family="",
        environment_id=environment.environment_id,
        environment_fingerprint=environment.fingerprint().fingerprint,
        environment_version=environment.version,
        environment_type=environment.environment_type,
        tool_names=[environment.name],
        tool_versions={environment.name: environment.version},
        tool_schema_fingerprints={environment.name: environment.schema_hash},
        schema_path_hashes=environment.schema_path_hashes(),
        error_type=error_type,
        error_code=error_code,
        error_message=message,
        failure_signature=normalize_signature(
            message=message,
            error_type=error_type,
            error_code=error_code,
            tool_name=environment.name,
        ),
    )


def pure_context(workspace_id: str, query: str) -> RetrievalContext:
    return RetrievalContext(workspace_id=workspace_id, query=query)


def run_condition(
    container: Container,
    workspace_id: str,
    *,
    condition: str,
    scenario: str,
    environment: BFCLEnvironment,
    query: str,
    args: dict[str, Any],
    target_function: str,
    allowed: set[str],
    memories: dict[str, Any],
    has_safe_memory: bool,
    current_memory_id: str | None = None,
    stale_memory_id: str | None = None,
    run_id: str,
) -> list[dict[str, Any]]:
    """Evaluate every arm for one condition row."""
    rows: list[dict[str, Any]] = []
    typed = typed_context(workspace_id, environment, query, args)
    pure = pure_context(workspace_id, query)
    for arm, mode, kind in ARMS:
        result: SelectionResult
        if kind == "none":
            result = SelectionResult(None)
        elif kind == "keyed":
            result = keyed_selection(target_function, memories=memories, allowed=allowed)
        elif kind == "hash":
            result = keyed_selection(
                target_function,
                memories=memories,
                allowed=allowed,
                by_hash=environment.schema_hash,
            )
        else:
            context = pure if kind == "pure" else typed
            decision = retrieve(
                container,
                RetrievalRequest(
                    context=context,
                    mode=mode,  # type: ignore[arg-type]
                    final_k=1,
                    run_id=run_id,
                ),
            )
            if arm in {"rrf_version_filter", "fingerprint_filter"}:
                result = filtered_selection(
                    decision,
                    memories=memories,
                    function=target_function,
                    by_fingerprint=(
                        environment.fingerprint().fingerprint if arm == "fingerprint_filter" else ""
                    ),
                )
            else:
                result = selection_from_decision(decision)
        rows.append(
            evaluate_row(
                condition=condition,
                scenario=scenario,
                arm=arm,
                has_safe_memory=has_safe_memory,
                schema=environment.schema,
                args=args,
                target_function=target_function,
                result=result,
                current_memory_id=current_memory_id,
                stale_memory_id=stale_memory_id,
            )
        )
    return rows


# ------------------------------------------------------------------------ driver
def prepare_scenarios(
    data_dir: Path, limit: int | None
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    tasks, hashes = load_tasks(data_dir)
    scenarios = sorted(build_transfer_scenarios(tasks), key=lambda item: item["function"])
    if limit is not None:
        scenarios = scenarios[:limit]
    return scenarios, hashes


def promote_pool(
    container: Container,
    workspace_id: str,
    scenarios: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    memories: dict[str, Any] = {}
    failures: list[dict[str, Any]] = []
    stages = {"runs": 0, "candidates": 0, "validated": 0, "promoted": 0}
    for scenario in scenarios:
        function = scenario["function"]
        environment = BFCLEnvironment(
            name=function,
            schema=scenario["schema"],
            version="2",
            failing_args=scenario["args1"],
        )
        repaired = apply_rename(scenario["args1"], scenario["old"], scenario["new"])
        run_id = record_failure_run(
            container,
            workspace_id,
            environment,
            scenario["args1"],
            repaired,
            task_id=f"{workspace_id}-{function}",
        )
        stages["runs"] += 1
        memory, detail = compile_and_promote(
            container, workspace_id, run_id, environment, scenario["schema"]
        )
        if detail.get("candidate_id"):
            stages["candidates"] += 1
        if detail.get("validation") == "passed":
            stages["validated"] += 1
        if memory is not None:
            stages["promoted"] += 1
            memories[function] = memory
        else:
            failures.append({"function": function, **detail})
    return memories, failures, stages


def drift_subset(
    container: Container,
    workspace_id: str,
    scenarios: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Two schema versions of one function inside one environment id.

    The transfer scenario already carries the first drift (``old`` was renamed to
    ``old_v2``). Here the *same* parameter is renamed again to ``old_v3`` without
    changing the environment id. Placeholder memory 1 maps ``old -> old_v2`` and
    becomes stale; memory 2 maps ``old -> old_v3`` and supersedes it. The three
    phases are: valid memory, stale memory only (policy must abstain), current
    memory after supersession.
    """
    rows: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    stages = {
        "functions": 0,
        "runs": 0,
        "promoted": 0,
        "superseded": 0,
        "failures": [],
    }
    for scenario in scenarios:
        function = scenario["function"]
        old = scenario["old"]
        new1 = scenario["new"]
        first = scenario["schema"]
        second, old2, new2 = drift_schema(first, parameter=new1, new_name=f"{old}_v3") or (
            None,
            "",
            "",
        )
        if second is None or old2 != new1:
            continue
        stages["functions"] += 1

        # Memory 1 was valid for the first drifted schema.
        env1 = BFCLEnvironment(
            name=function, schema=first, version="2", failing_args=scenario["args2"]
        )
        run1 = record_failure_run(
            container,
            workspace_id,
            env1,
            scenario["args2"],
            apply_rename(scenario["args2"], old, new1),
            task_id=f"{workspace_id}-{function}-v1",
        )
        stages["runs"] += 1
        memory1, detail1 = compile_and_promote(container, workspace_id, run1, env1, first)
        if memory1 is None:
            stages["failures"].append({"function": function, "phase": 1, **detail1})
            continue
        stages["promoted"] += 1

        # The environment id stays the same; only the schema changes.
        env2 = BFCLEnvironment(
            name=function, schema=second, version="2", failing_args=scenario["args2"]
        )
        memories = active_memories(container, workspace_id)

        # Phase 1: the first memory is current.
        run_id, _, allowed = start_evaluation_run(
            container,
            workspace_id,
            env1,
            exclude_memory_ids=set(),
            task_id=f"{workspace_id}-{function}-phase1",
            reason="drift_phase1",
        )
        phase1 = run_condition(
            container,
            workspace_id,
            condition="drift_phase1",
            scenario=function,
            environment=env1,
            query=scenario["query2"],
            args=scenario["args2"],
            target_function=function,
            allowed=allowed,
            memories=memories,
            has_safe_memory=True,
            current_memory_id=memory1.id,
            run_id=run_id,
        )
        finish_evaluation_run(
            container,
            run_id,
            any(row["call_valid"] for row in phase1 if row["arm"] == "full_recap"),
        )
        rows.extend(phase1)

        # Phase 2: the schema drifted again; only the stale memory exists.
        run_id, _, allowed = start_evaluation_run(
            container,
            workspace_id,
            env2,
            exclude_memory_ids=set(),
            task_id=f"{workspace_id}-{function}-phase2",
            reason="drift_phase2",
        )
        phase2 = run_condition(
            container,
            workspace_id,
            condition="drift_phase2",
            scenario=function,
            environment=env2,
            query=scenario["query2"],
            args=scenario["args2"],
            target_function=function,
            allowed=allowed,
            memories=memories,
            has_safe_memory=False,
            stale_memory_id=memory1.id,
            run_id=run_id,
        )
        finish_evaluation_run(container, run_id, False)
        rows.extend(phase2)

        # Promote the second memory; the first must be retired as superseded.
        run2 = record_failure_run(
            container,
            workspace_id,
            env2,
            scenario["args2"],
            apply_rename(scenario["args2"], old, new2),
            task_id=f"{workspace_id}-{function}-v2",
        )
        stages["runs"] += 1
        memory2, detail2 = compile_and_promote(container, workspace_id, run2, env2, second)
        if memory2 is None:
            stages["failures"].append({"function": function, "phase": 2, **detail2})
            continue
        stages["promoted"] += 1
        with container.services() as services:
            first_status = services.memories.get_memory(memory1.id, workspace_id)
        superseded = bool(
            memory2.supersedes_memory_id == memory1.id
            and first_status is not None
            and first_status.status is MemoryStatus.RETIRED
        )
        stages["superseded"] += 1 if superseded else 0
        observations.append(
            {
                "function": function,
                "first_memory_id": memory1.id,
                "second_memory_id": memory2.id,
                "supersedes": memory2.supersedes_memory_id,
                "first_status_after": str(first_status.status) if first_status else "missing",
                "superseded": superseded,
                "first_rename": [old, new1],
                "second_rename": [new1, new2],
            }
        )

        # Phase 3: only the current memory remains eligible.
        memories = active_memories(container, workspace_id)
        run_id, _, allowed = start_evaluation_run(
            container,
            workspace_id,
            env2,
            exclude_memory_ids=set(),
            task_id=f"{workspace_id}-{function}-phase3",
            reason="drift_phase3",
        )
        phase3 = run_condition(
            container,
            workspace_id,
            condition="drift_phase3",
            scenario=function,
            environment=env2,
            query=scenario["query2"],
            args=scenario["args2"],
            target_function=function,
            allowed=allowed,
            memories=memories,
            has_safe_memory=True,
            current_memory_id=memory2.id,
            stale_memory_id=memory1.id,
            run_id=run_id,
        )
        finish_evaluation_run(
            container,
            run_id,
            any(row["call_valid"] for row in phase3 if row["arm"] == "full_recap"),
        )
        rows.extend(phase3)
    return rows, observations, stages


def sibling_rows(
    container: Container,
    workspace_id: str,
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    memories: dict[str, Any],
) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    skipped = 0
    for owner, query_owner in pairs:
        target = query_owner["function"]
        target_memory = memories.get(target)
        if target_memory is None:
            skipped += 1
            continue
        environment = BFCLEnvironment(
            name=target,
            schema=query_owner["schema"],
            version="2",
            failing_args=query_owner["args2"],
        )
        pool = active_memories(container, workspace_id)
        run_id, _, allowed = start_evaluation_run(
            container,
            workspace_id,
            environment,
            exclude_memory_ids={target_memory.id},
            task_id=f"{workspace_id}-sibling-{owner['function']}-{target}",
            reason="sibling",
        )
        condition_rows = run_condition(
            container,
            workspace_id,
            condition="sibling",
            scenario=f"{owner['function']}->{target}",
            environment=environment,
            query=query_owner["query2"],
            args=query_owner["args2"],
            target_function=target,
            allowed=allowed,
            memories=pool,
            has_safe_memory=False,
            run_id=run_id,
        )
        for row in condition_rows:
            row["sibling_owner"] = owner["function"]
        rows.extend(condition_rows)
        finish_evaluation_run(container, run_id, False)
    return rows, skipped


# ------------------------------------------------------------------------ report
def _rate(value: float) -> str:
    return f"{value:.3f}"


def _ci(value: dict[str, Any]) -> str:
    interval = value["cluster_ci95"]
    return f"[{interval[0]:+.3f}, {interval[1]:+.3f}]"


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Does RECAP add value beyond retrieval relevance? (BFCL v3)",
        "",
        "> Generated by `scripts/bfcl_recap_experiment.py` from the JSON report beside "
        "this file. Do not hand-edit the numbers.",
        "",
        "## What is real and what is not",
        "",
        "Real: every query, function schema, parameter description, and gold argument comes "
        "from BFCL v3. Every memory below was recorded, compiled, sandbox-validated, and "
        "promoted by RECAP's own services, then retrieved by `RecallEngine`.",
        "",
        "Not real: the failure itself. One required parameter is renamed on the real schema "
        "to create a schema mismatch. BFCL function calls are checked for schema validity; "
        "they are not executed against live APIs. Function selection is assumed correct: the "
        "retrieval context supplies the tool the agent already chose.",
        "",
        "## Arms",
        "",
        "| Arm | Sees | Decision rule |",
        "|---|---|---|",
        "| `no_memory` | nothing | retry the failed call unchanged |",
        "| `retry_only` | nothing | retry the failed call unchanged (control) |",
        "| `bm25` | query text | top lexical hit, no tool/schema awareness |",
        "| `vector` | query text | top dense hit, no tool/schema awareness |",
        "| `hybrid_rrf` | query text | RRF fusion of BM25 + dense, no tool/schema awareness |",
        "| `rrf_typed` | query + tool + schema fingerprint + error | RRF ranking; retrieval-only, no policy |",
        "| `rrf_version_filter` | same as `rrf_typed` | RRF ranking, keep only memories for this environment id |",
        "| `fingerprint_filter` | same as `rrf_typed` | RRF ranking, keep only exact environment-fingerprint matches |",
        "| `function_keyed` | tool name | newest memory whose tool list contains the tool |",
        "| `schema_hash` | tool name + schema hash | newest memory whose recorded schema hash matches |",
        "| `full_recap` | the real `RetrievalContext` | `RecallEngine` eligibility, applicability, compatibility, utility threshold |",
        "",
        "## Provenance",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| Dataset | `{payload['dataset']}` |",
        f"| Generated | `{payload['generated_at']}` |",
        f"| Embedder | `{payload['embedding']['provider']}` / "
        f"`{payload['embedding']['model']}` ({payload['embedding']['dimension']}d) |",
        f"| Index prefix | `{payload['index_prefix']}` |",
        f"| Scenarios | {payload['funnel']['scenarios']} |",
        f"| Memories promoted | {payload['funnel']['memories_promoted']} |",
        f"| Utility threshold | {payload['utility_threshold']:.2f} |",
        f"| Bootstrap | {payload['bootstrap_resamples']} function-level resamples |",
        "",
        "## Funnel",
        "",
        "| Stage | Count |",
        "|---|---:|",
    ]
    for key, value in payload["funnel"].items():
        if key == "promotion_failures":
            continue
        lines.append(f"| `{key}` | {value} |")
    for failure in payload["funnel"].get("promotion_failures") or []:
        lines.append(f"| promotion failure `{failure['function']}` | {failure.get('stage', '')} |")
    lines.extend(
        [
            "",
            "Selection funnel (candidates -> eligible -> applicable -> selected, summed over "
            "scenario rows):",
            "",
            "| Condition | Arm | candidates | eligible | applicable | selected | abstained |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for condition in (*SAFE_CONDITIONS, *NO_SAFE_CONDITIONS):
        block = payload["conditions"].get(condition)
        if not block:
            continue
        for arm in (
            "bm25",
            "vector",
            "hybrid_rrf",
            "rrf_version_filter",
            "function_keyed",
            "full_recap",
        ):
            stats = block["results"].get(arm)
            if not stats or "funnel" not in stats:
                continue
            funnel = stats["funnel"]
            lines.append(
                f"| `{condition}` | `{arm}` | {funnel['candidates']} | "
                f"{funnel['eligible']} | {funnel['applicable']} | "
                f"{funnel['selected']} | {funnel['abstained']} |"
            )
    lines.extend(
        [
            "",
            "## Safe conditions: a matching memory exists",
            "",
            "| Condition | Arm | call valid | correct memory | harmful transfer | abstain |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for condition in SAFE_CONDITIONS:
        block = payload["conditions"].get(condition)
        if not block:
            continue
        for arm, stats in block["results"].items():
            lines.append(
                f"| `{condition}` | `{arm}` | {_rate(stats['call_valid_rate'])} | "
                f"{_rate(stats['correct_memory_rate'])} | "
                f"{_rate(stats['harmful_transfer_rate'])} | "
                f"{_rate(stats['abstain_rate'])} |"
            )
    lines.extend(
        [
            "",
            "## No-safe-memory conditions: the right answer is to abstain",
            "",
            "| Condition | Arm | false apply | harmful transfer | valid transfer | abstain |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for condition in NO_SAFE_CONDITIONS:
        block = payload["conditions"].get(condition)
        if not block:
            continue
        for arm, stats in block["results"].items():
            lines.append(
                f"| `{condition}` | `{arm}` | {_rate(stats['false_apply_rate'])} | "
                f"{_rate(stats['harmful_transfer_rate'])} | "
                f"{_rate(stats['valid_transfer_rate'])} | "
                f"{_rate(stats['abstain_rate'])} |"
            )
    lines.extend(
        [
            "",
            "## Abstention precision and recall",
            "",
            "Safe rows are `in_pool`, `drift_phase1`, `drift_phase3`; no-safe rows are "
            '`held_out`, `sibling`, `drift_phase2`. The positive class is "should abstain": '
            "precision is the share of abstentions taken on no-safe rows, recall is the share "
            "of no-safe rows where the arm abstained. A selection in a no-safe row is a false "
            "positive even when the repair happens to work; `valid_transfer` in the table "
            "above reports that conservative cost.",
            "",
            "| Arm | precision | recall | missed abstention (no-safe selected) | "
            "unneeded abstention (safe) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for arm, stats in payload["abstention"].items():
        lines.append(
            f"| `{arm}` | {_rate(stats['precision'])} | {_rate(stats['recall'])} | "
            f"{stats['missed_abstentions']}/{stats['n_no_safe']} | "
            f"{stats['unneeded_abstentions']}/{stats['n_safe']} |"
        )
    lines.extend(
        [
            "",
            "## Paired bootstrap against the full policy",
            "",
            "Positive means the full policy does better: call-valid rate on safe rows, "
            "abstention rate on no-safe rows. Intervals are 95% cluster bootstraps over "
            "functions; zero not excluded means the data do not separate the arms.",
            "",
            "| Subset | Baseline | observed | 95% CI | excludes zero |",
            "|---|---|---:|---:|---|",
        ]
    )
    for subset, block in payload["paired"].items():
        for arm, stats in block.items():
            lines.append(
                f"| `{subset}` | `{arm}` | {stats['observed']:+.3f} | "
                f"{_ci(stats)} | {stats['excludes_zero']} |"
            )
    lines.extend(
        [
            "",
            "## Drift and supersession",
            "",
            "| Observation | Value |",
            "|---|---|",
            f"| Functions with two promotions | {payload['drift']['functions']} |",
            f"| Supersessions confirmed | {payload['drift']['superseded']} |",
            f"| Active memories after supersession | {payload['drift']['active_after']} |",
        ]
    )
    for failure in payload["drift"].get("failures") or []:
        lines.append(
            f"| drift failure `{failure['function']}` (phase {failure['phase']}) | "
            f"{failure.get('stage', '')} |"
        )
    if payload["drift"].get("examples"):
        lines.extend(
            [
                "",
                "Example supersession:",
                "",
                "```",
                *[json.dumps(example, indent=2) for example in payload["drift"]["examples"][:3]],
                "```",
            ]
        )
    lines.extend(
        [
            "",
            "## Limits",
            "",
        ]
    )
    lines.extend(f"- {limit}" for limit in payload["limits"])
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--embedding", choices=["hashing", "qwen"], default="hashing")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--dimension", type=int, default=None)
    parser.add_argument(
        "--database-url", default="postgresql+psycopg://recap:recap@localhost:5433/recap"
    )
    parser.add_argument("--es-url", default="http://localhost:9200")
    parser.add_argument("--index-prefix", default=None)
    parser.add_argument("--workspace", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--conditions", default="in_pool,held_out,sibling,drift")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--doc", type=Path, default=None)
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP)
    parser.add_argument("--seed", type=int, default=RNG_SEED)
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args(argv)

    if not args.data_dir.is_dir():
        raise SystemExit(
            f"{args.data_dir} does not exist. Download BFCL v3 into it (see the module "
            "docstring) or pass --data-dir."
        )
    if args.embedding == "qwen" and args.model_path is None:
        raise SystemExit(
            "--model-path is required for --embedding qwen (a local sentence-transformers "
            "directory or Hugging Face model id)"
        )
    args.index_prefix = args.index_prefix or f"{DEFAULT_PREFIX}-{args.embedding}"
    args.workspace = args.workspace or f"bfcl-recap-{args.embedding}"
    conditions = {item.strip() for item in args.conditions.split(",") if item.strip()}
    unknown = conditions - {"in_pool", "held_out", "sibling", "drift"}
    if unknown:
        raise SystemExit(f"unknown conditions: {sorted(unknown)}")

    scenarios, hashes = prepare_scenarios(args.data_dir, args.limit)
    if len(scenarios) < 4:
        raise SystemExit("too few transfer scenarios to report")
    all_functions = sorted({scenario["function"] for scenario in scenarios})
    print(f"scenarios={len(scenarios)} functions={len(all_functions)}")

    settings = build_settings(args)
    container = Container.build(settings)
    container.migrate()
    container.init_search()
    container.init_search_chunks()

    main_workspace = args.workspace
    drift_workspace = f"{args.workspace}-drift"
    if not args.keep:
        reset_workspace(container, [main_workspace, drift_workspace])

    memories, failures, stages = promote_pool(container, main_workspace, scenarios)
    print(f"promoted={len(memories)} failures={len(failures)}")
    active = active_memories(container, main_workspace)

    rows: list[dict[str, Any]] = []
    skip_counts: dict[str, int] = {}
    sibling_pair_count = len(sibling_pairs(scenarios)) if "sibling" in conditions else 0
    if "in_pool" in conditions:
        for scenario in scenarios:
            function = scenario["function"]
            target = memories.get(function)
            if target is None:
                continue
            environment = BFCLEnvironment(
                name=function,
                schema=scenario["schema"],
                version="2",
                failing_args=scenario["args2"],
            )
            pool = active_memories(container, main_workspace)
            run_id, _, allowed = start_evaluation_run(
                container,
                main_workspace,
                environment,
                exclude_memory_ids=set(),
                task_id=f"{main_workspace}-in-pool-{function}",
                reason="in_pool",
            )
            condition_rows = run_condition(
                container,
                main_workspace,
                condition="in_pool",
                scenario=function,
                environment=environment,
                query=scenario["query2"],
                args=scenario["args2"],
                target_function=function,
                allowed=allowed,
                memories=pool,
                has_safe_memory=True,
                current_memory_id=target.id,
                run_id=run_id,
            )
            rows.extend(condition_rows)
            finish_evaluation_run(
                container,
                run_id,
                any(row["call_valid"] for row in condition_rows if row["arm"] == "full_recap"),
            )
    if "held_out" in conditions:
        for scenario in scenarios:
            function = scenario["function"]
            target = memories.get(function)
            if target is None:
                skip_counts["held_out"] = skip_counts.get("held_out", 0) + 1
                continue
            environment = BFCLEnvironment(
                name=function,
                schema=scenario["schema"],
                version="2",
                failing_args=scenario["args2"],
            )
            pool = active_memories(container, main_workspace)
            run_id, _, allowed = start_evaluation_run(
                container,
                main_workspace,
                environment,
                exclude_memory_ids={target.id},
                task_id=f"{main_workspace}-held-out-{function}",
                reason="held_out",
            )
            condition_rows = run_condition(
                container,
                main_workspace,
                condition="held_out",
                scenario=function,
                environment=environment,
                query=scenario["query2"],
                args=scenario["args2"],
                target_function=function,
                allowed=allowed,
                memories=pool,
                has_safe_memory=False,
                run_id=run_id,
            )
            rows.extend(condition_rows)
            finish_evaluation_run(container, run_id, False)
    if "sibling" in conditions:
        pairs = sibling_pairs(scenarios)
        condition_rows, skipped = sibling_rows(container, main_workspace, pairs, memories)
        rows.extend(condition_rows)
        skip_counts["sibling"] = skipped
    if "drift" in conditions:
        drift_rows, observations, drift_stages = drift_subset(container, drift_workspace, scenarios)
        rows.extend(drift_rows)
    else:
        observations = []
        drift_stages = {"functions": 0, "runs": 0, "promoted": 0, "superseded": 0, "failures": []}

    rng = np.random.default_rng(args.seed)
    conditions_payload: dict[str, Any] = {}
    for condition in sorted({row["condition"] for row in rows}):
        condition_rows = [row for row in rows if row["condition"] == condition]
        results: dict[str, Any] = {}
        for arm, _, _ in ARMS:
            arm_rows = [row for row in condition_rows if row["arm"] == arm]
            stats = summarize_rows(arm_rows)
            values: dict[str, list[float]] = defaultdict(list)
            for row in arm_rows:
                values[str(row.get("cluster") or row["scenario"])].append(
                    1.0 if row["call_valid"] else 0.0
                )
            stats["call_valid_ci95"] = cluster_bootstrap(values, rng, resamples=args.bootstrap)[
                "cluster_ci95"
            ]
            stats["funnel"] = {
                "rows": len(arm_rows),
                "candidates": sum(row["candidate_count"] for row in arm_rows),
                "eligible": sum(row["eligible_count"] for row in arm_rows),
                "applicable": sum(row["applicable_count"] for row in arm_rows),
                "selected": sum(1 for row in arm_rows if row["selected"]),
                "abstained": sum(1 for row in arm_rows if row["abstained"]),
            }
            results[arm] = stats
        conditions_payload[condition] = {
            "n_scenarios": len({row["scenario"] for row in condition_rows}),
            "results": results,
            "per_scenario": condition_rows,
        }

    # Abstention precision/recall across safe and no-safe conditions.
    abstention = abstention_stats(rows)

    paired: dict[str, Any] = {}
    for subset, allowed_conditions in (
        ("safe", SAFE_CONDITIONS),
        ("no_safe", NO_SAFE_CONDITIONS),
    ):
        treatment = [
            row
            for row in rows
            if row["arm"] == "full_recap" and row["condition"] in allowed_conditions
        ]
        block: dict[str, Any] = {}
        for arm, _, _ in ARMS:
            if arm == "full_recap":
                continue
            reference = [
                row for row in rows if row["arm"] == arm and row["condition"] in allowed_conditions
            ]
            if not reference:
                continue
            value_key = "call_valid" if subset == "safe" else "abstained"
            block[arm] = paired_bootstrap(
                treatment,
                reference,
                cluster_key="cluster",
                value_key=value_key,
                rng=rng,
                resamples=args.bootstrap,
            )
        paired[subset] = block

    active_after = len(active_memories(container, drift_workspace)) if "drift" in conditions else 0
    payload: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset": "gorilla-llm/Berkeley-Function-Calling-Leaderboard (BFCL v3)",
        "source_hashes": hashes,
        "index_prefix": args.index_prefix,
        "workspace": main_workspace,
        "drift_workspace": drift_workspace,
        "embedding": {
            "provider": settings.embeddings.provider,
            "model": getattr(container.embeddings, "model_id", settings.embeddings.model_id),
            "dimension": settings.embeddings.dimension,
            "query_prefix": settings.embeddings.query_prefix,
        },
        "utility_threshold": UTILITY_THRESHOLD,
        "bootstrap_resamples": args.bootstrap,
        "rng_seed": args.seed,
        "funnel": {
            "scenarios": len(scenarios),
            "functions_with_labeled_calls": len(
                {task["function"] for task in load_tasks(args.data_dir)[0]}
            ),
            "failure_runs": stages["runs"],
            "candidates_compiled": stages["candidates"],
            "validations_passed": stages["validated"],
            "memories_promoted": stages["promoted"],
            "active_pool": len(active),
            "sibling_pairs": sibling_pair_count,
            "promotion_failures": failures,
        },
        "conditions": conditions_payload,
        "abstention": abstention,
        "paired": paired,
        "drift": {
            "functions": drift_stages["functions"],
            "runs": drift_stages["runs"],
            "promoted": drift_stages["promoted"],
            "superseded": drift_stages["superseded"],
            "active_after": active_after,
            "failures": drift_stages["failures"],
            "examples": observations,
        },
        "skip_counts": skip_counts,
        "limits": [
            "The failure is injected: one required parameter is renamed on the real schema. "
            "No production failure log was used.",
            "Calls are checked for schema validity against the real BFCL schema; they are "
            "not executed against live APIs.",
            "Function selection is assumed correct. The context supplies the tool the agent "
            "already chose, so this isolates memory selection from function selection.",
            "The policy is the shipped default (utility threshold 0.35); no weight or "
            "threshold was tuned on these outcomes.",
            "Drift is simulated inside one environment id (no version bump), which is what "
            "makes the stale memory look applicable to an environment-id filter.",
            "Sibling pairs use a name-family heuristic (shared prefix before the first "
            "dot/underscore or `_by_`); 24 pairs over 8 query functions give wide intervals.",
            "Promotion hardcodes environment_type=mock_api and does not store schema path "
            "hashes, so field-scoped compatibility is not exercised here.",
            "Multi-turn BFCL categories, real model failures, and non-schema execution "
            "errors are out of scope for this report.",
        ],
    }

    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"real-data-bfcl-recap-{args.embedding}"
    json_path = args.out / f"{stem}.json"
    doc_path = args.doc or args.out / f"{stem}.md"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    doc_path.write_text(render_markdown(payload), encoding="utf-8")
    print(f"wrote {json_path}")
    print(f"wrote {doc_path}")
    for condition, block in conditions_payload.items():
        for arm in ("bm25", "vector", "hybrid_rrf", "function_keyed", "schema_hash", "full_recap"):
            stats = block["results"].get(arm)
            if stats:
                print(
                    f"  {condition:13s} {arm:14s} n={stats['n']:3d} "
                    f"valid={stats['call_valid_rate']:.3f} "
                    f"false={stats['false_apply_rate']:.3f} "
                    f"abstain={stats['abstain_rate']:.3f}"
                )
    return 0


if __name__ == "__main__":
    sys.exit(main())
