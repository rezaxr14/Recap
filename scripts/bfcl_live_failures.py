"""Real model failures on BFCL v3, repaired by the real RECAP stack (Task 3).

Unlike ``bfcl_recap_experiment.py`` (Task 2), nothing here is injected: a real
local model is called through an OpenAI-compatible endpoint with the real BFCL
function schemas, and its own invalid calls become the failure evidence.

For every function with at least two labeled queries, the first query is the
*dev* split and the rest are held out. On dev the model gets exactly one tool
(the correct one) and its call is validated against the real schema, including
declared JSON types and enums. A type failure is recorded through RECAP's
``RunRecorder``, compiled by ``BFCLTypeCoercionGenerator``, replayed by
``BFCLSandboxValidator``, and promoted. The held-out queries are then run with
three decoding seeds and six arms:

    no_memory, retry_only, bm25, vector, rrf_typed, full_recap

``retry_only`` asks the model again with the schema error as feedback. Memory
arms apply the selected memory's typed actions deterministically to the first
call, which is RECAP's contract: the memory is guidance the host agent applies.
The first model call is cached and shared by every arm, so arm differences are
selection differences, not sampling differences.

Discovery is greedy (temperature 0). Evaluation uses nonzero temperature so the
held-out split contains genuine sampling failures as well as systematic ones.

Data (public, one download):
    python -c "from huggingface_hub import hf_hub_download as d; \\
      [d('gorilla-llm/Berkeley-Function-Calling-Leaderboard', f, repo_type='dataset', \\
         local_dir='data/bfcl') for f in ('BFCL_v3_simple.json', \\
         'possible_answer/BFCL_v3_simple.json', 'BFCL_v3_exec_simple.json')]"

Run (LM Studio on 1234, Postgres/Elasticsearch from docker compose up):
    python scripts/bfcl_live_failures.py --model llama-3.2-3b-instruct \\
        --embedding qwen --model-path /path/to/Qwen3-Embedding-0.6B

The JSON and Markdown reports are written under ``docs/evidence``. Numbers are
generated from the JSON report; they are never hand-edited.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from recap.config import (
    EmbeddingSettings,
    LlmSettings,
    Settings,
    load_settings,
    redact_url,
)
from recap.container import Container
from recap.domain.enums import EventType, MemoryStatus, RetrievalMode, RunStatus
from recap.domain.retrieval import RetrievalContext, RetrievalRequest
from recap.domain.runs import AgentRun, RunEventCreate
from recap.environments.bfcl import (
    BFCLEnvironment,
    argument_errors,
    openai_tool_schema,
)
from recap.evals.bfcl import load_tasks
from recap.infrastructure.db.repositories import ensure_workspace
from recap.maintenance import reset_workspace
from recap.providers.base import ChatMessage
from recap.providers.factory import build_chat_provider
from recap.recall.utility_policy import POLICY_VERSION
from recap.recorder.normalization import normalize_signature
from recap.repair_compiler.bfcl import (
    BFCLSandboxValidator,
    BFCLTypeCoercionGenerator,
    coerce_argument,
    coerce_to_schema,
)

DEFAULT_DATA = Path("data/bfcl")
DEFAULT_OUT = Path("docs/evidence")
DEFAULT_PREFIX = "recap-bfcl-live"
BOOTSTRAP = 2000
RNG_SEED = 20261103
UTILITY_THRESHOLD = 0.35
AGENT_ID = "bfcl-live-failures"

PROMPT_VERSION = "bfcl-live-tools-v1"
SYSTEM_PROMPT = (
    "You are a tool-calling agent. Call exactly one provided tool with arguments "
    "that satisfy its JSON schema. Never invent argument names. Do not explain."
)
RETRY_PROMPT = (
    "The tool rejected that call: {error}. Return a corrected call to the same "
    "tool. Keep the values that were accepted."
)

ARMS = (
    "no_memory",
    "retry_only",
    "bm25",
    "vector",
    "rrf_typed",
    "full_recap",
)

SUBSETS = (
    "all",
    "selected_ok",
    "execution_failed",
    "safe_failed",
    "no_safe",
    "no_safe_failed",
    "initially_valid",
)


# --------------------------------------------------------------------------- pure
def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def split_tasks(
    tasks: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Split real labeled tasks into one dev query and held-out queries per function.

    A function is usable only when it has at least two distinct queries and a
    single schema variant in the loaded data. Rows carry no gold arguments: the
    experiment scores the model's own calls, not agreement with BFCL labels.
    """
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        grouped[str(task["function"])].append(task)

    dev: list[dict[str, Any]] = []
    evaluation: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for function, items in sorted(grouped.items()):
        variants: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in items:
            variants[canonical(item.get("schema") or {})].append(item)
        chosen_key = sorted(
            variants,
            key=lambda key: (-len({row["query"] for row in variants[key]}), key),
        )[0]
        unique: dict[str, dict[str, Any]] = {}
        for item in variants[chosen_key]:
            unique.setdefault(str(item["query"]), item)
        ordered = [unique[key] for key in sorted(unique)]
        if len(ordered) < 2:
            skipped.append(
                {
                    "function": function,
                    "reason": "fewer_than_two_distinct_queries",
                    "variants": len(variants),
                }
            )
            continue
        base = {
            "function": function,
            "schema": ordered[0]["schema"],
            "schema_hash": BFCLEnvironment(name=function, schema=ordered[0]["schema"]).schema_hash,
        }
        dev.append({**base, "task_id": ordered[0]["id"], "query": ordered[0]["query"]})
        for item in ordered[1:]:
            evaluation.append({**base, "task_id": item["id"], "query": item["query"]})
    return dev, evaluation, skipped


def apply_actions(
    args: dict[str, Any],
    actions: list[dict[str, Any]],
    schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply the typed strategy actions a memory asks the host agent to perform."""
    repaired = dict(args)
    properties = ((schema or {}).get("parameters") or {}).get("properties") or {} if schema else {}
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
        elif action_type == "coerce_argument_type":
            targets = parameters.get("arguments")
            if isinstance(targets, dict):
                for key, target in targets.items():
                    if key not in repaired:
                        continue
                    spec = properties.get(key)
                    repaired[key] = (
                        coerce_to_schema(spec, repaired[key])
                        if isinstance(spec, dict)
                        else coerce_argument(repaired[key], str(target))
                    )
    return repaired


def error_category(error_code: str) -> str:
    return {
        "INVALID_ARGUMENT_TYPE": "argument_type_mismatch",
        "INVALID_ENUM_VALUE": "enum_value_mismatch",
        "UNKNOWN_PARAMETER": "tool_parameter_mismatch",
        "MISSING_REQUIRED_PARAMETER": "missing_required_parameter",
    }.get(error_code, "unclassified_failure")


def argument_diagnostics(schema: dict[str, Any], args: dict[str, Any]) -> list[dict[str, Any]]:
    """Every schema problem we can name, for the report.

    ``BFCLEnvironment.call`` stops at the first error; this lists all of them so
    a single missing argument does not hide three type mismatches. It is not
    used to decide anything, only to describe the observed failure.
    """
    params = schema.get("parameters") or {}
    properties = dict(params.get("properties") or {})
    required = list(params.get("required") or properties)
    diagnostics: list[dict[str, Any]] = []
    for name in sorted(set(args) - set(properties)):
        diagnostics.append(
            {
                "argument": name,
                "code": "UNKNOWN_PARAMETER",
                "message": f"unknown parameter '{name}'",
            }
        )
    for name in required:
        if name not in args:
            diagnostics.append(
                {
                    "argument": name,
                    "code": "MISSING_REQUIRED_PARAMETER",
                    "message": f"missing required parameter '{name}'",
                }
            )
    for name in sorted(set(args) & set(properties)):
        spec = properties.get(name)
        if not isinstance(spec, dict):
            continue
        for code, message in argument_errors(spec, args[name], name):
            diagnostics.append({"argument": name, "code": code, "message": message})
    return diagnostics


def classify_execution(environment: BFCLEnvironment, args: dict[str, Any]) -> dict[str, Any]:
    outcome = environment.call(args)
    if outcome.ok:
        return {
            "execution_ok": True,
            "error_code": "",
            "error_message": "",
            "failure_category": "none",
            "diagnostics": [],
        }
    error = outcome.error or {}
    code = str(error.get("error_code") or "")
    return {
        "execution_ok": False,
        "error_code": code,
        "error_message": str(error.get("error_message") or ""),
        "failure_category": error_category(code),
        "diagnostics": argument_diagnostics(environment.schema, args),
    }


def cache_key(
    *,
    model: str,
    base_url: str,
    task_id: str,
    seed: int | None,
    temperature: float,
    kind: str,
    feedback: str = "",
    previous: str = "",
) -> str:
    payload = canonical(
        {
            "model": model,
            "base_url": base_url,
            "prompt_version": PROMPT_VERSION,
            "task_id": task_id,
            "seed": seed,
            "temperature": temperature,
            "kind": kind,
            "feedback": feedback,
            "previous": previous,
        }
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def is_correct_selection(call_name: str | None, call_count: int, function: str) -> bool:
    return call_count == 1 and call_name == function


def rate(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([1.0 if row[key] else 0.0 for row in rows])) if rows else 0.0


def summarize_arm(rows: list[dict[str, Any]]) -> dict[str, Any]:
    selection_ok = [row for row in rows if row["selection_ok"]]
    execution_failed = [row for row in selection_ok if not row["initial_valid"]]
    initially_valid = [row for row in rows if row["initial_valid"]]
    return {
        "n": len(rows),
        "selection_ok_rate": rate(rows, "selection_ok"),
        "initial_valid_rate": rate(rows, "initial_valid"),
        "call_valid_rate": rate(rows, "call_valid"),
        "execution_repair_rate": rate(execution_failed, "call_valid"),
        "repair_rate_all_invalid": rate(
            [row for row in rows if not row["initial_valid"]], "call_valid"
        ),
        "broken_valid_rate": rate(initially_valid, "broken_valid_call"),
        "selected_rate": rate(rows, "selected"),
        "abstain_rate": rate(rows, "abstained"),
        "correct_memory_rate": rate(rows, "selected_is_current"),
        "false_apply_rate": rate(rows, "false_apply"),
        "harmful_apply_rate": rate(rows, "harmful_apply"),
        "hit_at_1": rate(rows, "retrieval_hit_at_1"),
        "retried_rate": rate(rows, "retried"),
        "funnel": {
            "rows": len(rows),
            "candidates": sum(row["candidate_count"] for row in rows),
            "eligible": sum(row["eligible_count"] for row in rows),
            "applicable": sum(row["applicable_count"] for row in rows),
            "selected": sum(1 for row in rows if row["selected"]),
            "abstained": sum(1 for row in rows if row["abstained"]),
        },
    }


def subset_rows(rows: list[dict[str, Any]], subset: str) -> list[dict[str, Any]]:
    if subset == "all":
        return list(rows)
    if subset == "selected_ok":
        return [row for row in rows if row["selection_ok"]]
    if subset == "execution_failed":
        return [row for row in rows if row["selection_ok"] and not row["initial_valid"]]
    if subset == "safe_failed":
        return [
            row
            for row in rows
            if row["condition"] == "in_pool" and row["selection_ok"] and not row["initial_valid"]
        ]
    if subset == "no_safe":
        return [row for row in rows if not row["has_safe_memory"]]
    if subset == "no_safe_failed":
        return [row for row in rows if not row["has_safe_memory"] and not row["initial_valid"]]
    if subset == "initially_valid":
        return [row for row in rows if row["initial_valid"]]
    raise KeyError(subset)


def cluster_bootstrap(
    values_by_cluster: dict[str, list[float]],
    rng: np.random.Generator,
    *,
    resamples: int = BOOTSTRAP,
) -> dict[str, Any]:
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
    """Paired cluster bootstrap over the same (condition, task, seed) keys."""
    reference_map = {
        (row["condition"], row["task_id"], row["seed"]): row[value_key] for row in reference
    }
    by_cluster: dict[str, list[float]] = defaultdict(list)
    for row in treatment:
        key = (row["condition"], row["task_id"], row["seed"])
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
    settings.embeddings.cache_enabled = bool(getattr(args, "embedding_cache", False))
    settings.repair_compiler.generator = "deterministic"
    settings.recall.require_tool_match = False
    settings.recall.no_safe_memory_threshold = UTILITY_THRESHOLD
    if args.embedding == "qwen":
        dimension = args.dimension or 1024
        settings.embeddings = EmbeddingSettings(
            provider="sentence-transformers",
            model_path=str(args.model_path),
            dimension=dimension,
            query_prefix="Instruct: Given a failed tool call, retrieve a repair\nQuery: ",
            cache_enabled=bool(getattr(args, "embedding_cache", False)),
        )
    else:
        dimension = args.dimension or 256
        settings.embeddings = EmbeddingSettings(
            provider="hashing", dimension=dimension, cache_enabled=False
        )
    settings.elasticsearch.vector_dimension = dimension
    settings.validate_consistency()
    return settings


def build_provider(args: argparse.Namespace) -> Any:
    settings = LlmSettings(
        provider="openai_compatible",
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        connect_timeout_seconds=args.connect_timeout,
        read_timeout_seconds=args.read_timeout,
        overall_timeout_seconds=args.overall_timeout,
        no_progress_timeout_seconds=args.no_progress_timeout,
        max_retries=args.max_retries,
        temperature=args.discovery_temperature,
        max_tokens=args.max_tokens,
        structured_mode="json_object",
        stream=False,
        max_concurrency=1,
    )
    provider = build_chat_provider("agent", settings)
    if provider is None:
        raise SystemExit("could not build the chat provider")
    return provider


@dataclass
class ModelCall:
    task_id: str
    seed: int | None
    kind: str
    name: str | None
    arguments: dict[str, Any]
    call_count: int
    error: str
    finish_reason: str
    latency_ms: float
    attempts: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "seed": self.seed,
            "kind": self.kind,
            "name": self.name,
            "arguments": self.arguments,
            "call_count": self.call_count,
            "error": self.error,
            "finish_reason": self.finish_reason,
            "latency_ms": self.latency_ms,
            "attempts": self.attempts,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ModelCall:
        return cls(
            task_id=str(payload["task_id"]),
            seed=payload.get("seed"),
            kind=str(payload["kind"]),
            name=payload.get("name"),
            arguments=dict(payload.get("arguments") or {}),
            call_count=int(payload.get("call_count") or 0),
            error=str(payload.get("error") or ""),
            finish_reason=str(payload.get("finish_reason") or ""),
            latency_ms=float(payload.get("latency_ms") or 0.0),
            attempts=int(payload.get("attempts") or 0),
        )


def call_model(
    provider: Any,
    cache: dict[str, dict[str, Any]],
    *,
    model: str,
    base_url: str,
    task: dict[str, Any],
    seed: int | None,
    temperature: float,
    kind: str,
    previous: ModelCall | None = None,
    feedback: str = "",
) -> ModelCall:
    key = cache_key(
        model=model,
        base_url=base_url,
        task_id=str(task["task_id"]),
        seed=seed,
        temperature=temperature,
        kind=kind,
        feedback=feedback,
        previous=canonical(previous.to_dict()) if previous is not None else "",
    )
    if key in cache:
        return ModelCall.from_dict(cache[key])

    messages = [
        ChatMessage(role="system", content=SYSTEM_PROMPT),
        ChatMessage(role="user", content=str(task["query"])),
    ]
    if previous is not None and feedback:
        messages.append(
            ChatMessage(
                role="assistant",
                content=(
                    f"I called {previous.name} with arguments "
                    f"{json.dumps(previous.arguments, sort_keys=True)}."
                ),
            )
        )
        messages.append(ChatMessage(role="user", content=feedback))

    tools = [
        {
            "type": "function",
            "function": openai_tool_schema(task["schema"]),
        }
    ]
    started = time.perf_counter()
    error = ""
    name: str | None = None
    arguments: dict[str, Any] = {}
    call_count = 0
    finish_reason = ""
    attempts = 0
    try:
        result = provider.chat(
            messages,
            tools=tools,
            temperature=temperature,
            seed=seed,
        )
        call_count = len(result.tool_calls)
        if result.tool_calls:
            name = result.tool_calls[0].name
            arguments = dict(result.tool_calls[0].arguments)
        finish_reason = result.finish_reason
        attempts = result.attempts
    except Exception as exc:  # noqa: BLE001 - the provider boundary is reported verbatim
        error = f"{type(exc).__name__}: {exc}"[:1000]
    call = ModelCall(
        task_id=str(task["task_id"]),
        seed=seed,
        kind=kind,
        name=name,
        arguments=arguments,
        call_count=call_count,
        error=error,
        finish_reason=finish_reason,
        latency_ms=round((time.perf_counter() - started) * 1000, 3),
        attempts=attempts,
    )
    cache[key] = call.to_dict()
    return call


def classify_call(
    *,
    task: dict[str, Any],
    call: ModelCall,
) -> dict[str, Any]:
    environment = BFCLEnvironment(name=str(task["function"]), schema=task["schema"], strict=True)
    selection_ok = bool(
        not call.error and is_correct_selection(call.name, call.call_count, str(task["function"]))
    )
    if not selection_ok:
        return {
            "selection_ok": False,
            "call_name": call.name,
            "call_count": call.call_count,
            "execution_ok": False,
            "error_code": "",
            "error_message": call.error,
            "failure_category": "provider_or_selection_failure",
            "diagnostics": [],
        }
    execution = classify_execution(environment, call.arguments)
    return {
        "selection_ok": True,
        "call_name": call.name,
        "call_count": call.call_count,
        **execution,
    }


def record_live_failure_run(
    container: Container,
    workspace_id: str,
    environment: BFCLEnvironment,
    task: dict[str, Any],
    call: ModelCall,
    classification: dict[str, Any],
    *,
    model_config: dict[str, Any],
) -> str:
    """Record the model's real failed call through RunRecorder."""
    fingerprint = environment.fingerprint()
    with container.services() as services:
        ensure_workspace(services.session, workspace_id)
        run = AgentRun(
            workspace_id=workspace_id,
            agent_id=AGENT_ID,
            task_id=f"{workspace_id}-{task['task_id']}",
            task_family="bfcl_live",
            environment_id=fingerprint.environment_id,
            environment_fingerprint=fingerprint.fingerprint,
            toolset_fingerprint=fingerprint.fingerprint,
            model_provider="openai_compatible",
            model_name=model_config["model"],
            success_criteria="the recorded call is schema-valid after the typed repair",
            config_snapshot={
                "adapter": "recap.environments.bfcl.BFCLEnvironment",
                "failure_source": "real_model_call",
                "model": model_config["model"],
                "base_url": model_config["base_url"],
                "prompt_version": PROMPT_VERSION,
                "temperature": model_config["discovery_temperature"],
                "seed": model_config["discovery_seed"],
                "task_id": task["task_id"],
                "query": str(task["query"])[:500],
                "failure_category": classification["failure_category"],
            },
        )
        run = services.recorder.start_run(run)
        sequence = 0

        def append(event_type: EventType, payload: dict[str, Any]) -> None:
            nonlocal sequence
            sequence += 1
            services.recorder.append_event(
                run,
                RunEventCreate(
                    sequence_number=sequence,
                    event_type=event_type,
                    tool_name=environment.name,
                    tool_version=environment.version,
                    payload=payload,
                ),
            )

        append(
            EventType.RUN_STARTED,
            {
                "tool": environment.name,
                "environment_version": environment.version,
                "memory_snapshot_id": run.memory_snapshot_id,
                "model": model_config["model"],
                "prompt_version": PROMPT_VERSION,
            },
        )
        append(
            EventType.TASK_OBSERVATION,
            {
                "observation": str(task["query"])[:500],
                "task_id": task["task_id"],
            },
        )
        append(EventType.TOOL_CALL_REQUESTED, {"request": call.arguments})
        append(
            EventType.TOOL_CALL_FAILED,
            {
                "request": call.arguments,
                "error_type": "InvalidRequest",
                "error_code": classification["error_code"],
                "error_message": classification["error_message"],
                "http_status": 400,
                "diagnostics": classification["diagnostics"],
            },
        )
        append(
            EventType.POSTCONDITION_CHECKED,
            {
                "passed": False,
                "detail": classification["error_message"],
                "error_code": classification["error_code"],
            },
        )
        append(
            EventType.RUN_FINISHED,
            {"success": False, "error": classification["error_message"]},
        )
        services.recorder.finish_run(
            run,
            status=RunStatus.FAILED,
            tool_calls_count=1,
            retry_count=0,
        )
        return run.id


def compile_and_promote(
    container: Container,
    workspace_id: str,
    run_id: str,
    environment: BFCLEnvironment,
    schema: dict[str, Any],
) -> tuple[Any | None, dict[str, Any]]:
    """Compile a typed coercion, sandbox-validate it, and promote the memory."""
    fingerprint = environment.fingerprint()
    container.validators.register(BFCLSandboxValidator({environment.environment_id: environment}))
    with container.services() as services:
        services.environments.upsert(workspace_id, fingerprint)
        services.repair_compiler.generator = BFCLTypeCoercionGenerator(
            {environment.environment_id: schema}
        )
        candidates = services.repair_compiler.compile(run_id, environment=fingerprint)
        if not candidates:
            return None, {"stage": "no_candidate"}
        last: dict[str, Any] = {"stage": "validation_not_run"}
        for candidate in candidates:
            result = services.repair_compiler.validate(
                candidate.id, environment=fingerprint, validator_name="bfcl_sandbox"
            )
            last = {
                "stage": f"validation_{result.status.value}",
                "detail": result.detail,
                "candidate_id": candidate.id,
            }
            if result.status.value != "passed":
                continue
            refreshed = services.repairs.get(candidate.id)
            if refreshed is None:
                last = {"stage": "candidate_disappeared", "candidate_id": candidate.id}
                continue
            memory = services.promotion.promote(refreshed)
            return memory, {
                "stage": "promoted",
                "candidate_id": candidate.id,
                "validation": result.status.value,
                "memory_id": memory.id,
                "memory_version": memory.memory_version,
                "supersedes": memory.supersedes_memory_id,
                "actions": [action.action_type.value for action in memory.content.strategy.actions],
            }
        return None, last


def start_evaluation_run(
    container: Container,
    workspace_id: str,
    environment: BFCLEnvironment,
    *,
    task_id: str,
    seed: int,
    reason: str,
    exclude_memory_ids: set[str],
) -> str:
    fingerprint = environment.fingerprint()
    with container.services() as services:
        ensure_workspace(services.session, workspace_id)
        snapshot = services.snapshot_service.create(
            workspace_id, reason=reason, exclude_memory_ids=exclude_memory_ids
        )
        services.environments.upsert(workspace_id, fingerprint)
        run = AgentRun(
            workspace_id=workspace_id,
            agent_id=AGENT_ID,
            task_id=f"{task_id}-s{seed}",
            task_family="bfcl_live_eval",
            environment_id=fingerprint.environment_id,
            environment_fingerprint=fingerprint.fingerprint,
            toolset_fingerprint=fingerprint.fingerprint,
            model_provider="openai_compatible",
            model_name="",
            success_criteria="the model's held-out call is schema-valid after the selected memory",
            memory_snapshot_id=snapshot.id,
            config_snapshot={
                "reason": reason,
                "seed": seed,
                "policy_threshold": UTILITY_THRESHOLD,
                "policy_version": POLICY_VERSION,
                "excluded_memory_ids": sorted(exclude_memory_ids),
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
                    "seed": seed,
                },
            ),
        )
        return run.id


def finish_evaluation_run(container: Container, run_id: str, success: bool) -> None:
    with container.services() as services:
        run = services.recorder.get_run(run_id)
        services.recorder.append_event(
            run,
            RunEventCreate(
                sequence_number=2,
                event_type=EventType.POSTCONDITION_CHECKED,
                payload={"passed": success, "detail": "live failure experiment"},
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


def selection_from_decision(
    decision: Any,
    *,
    memories: dict[str, Any],
    target_function: str,
) -> SelectionResult:
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
    memory = memories.get(selected.memory_id)
    tool_names = list(getattr(getattr(memory, "content", None), "tool_names", []) or [])
    if target_function in tool_names:
        tool = target_function
    elif tool_names:
        tool = tool_names[0]
    else:
        tool = ""
    return SelectionResult(
        Selection(
            memory_id=selected.memory_id,
            tool=tool,
            version=selected.memory_version,
            actions=list(selected.actions),
            status=str(decision.status),
        ),
        candidate_count=len(candidates),
        eligible_count=eligible,
        applicable_count=applicable,
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


# ------------------------------------------------------------------------ driver
def promote_pool(
    container: Container,
    workspace_id: str,
    dev: list[dict[str, Any]],
    calls: dict[str, ModelCall],
    classifications: dict[str, dict[str, Any]],
    *,
    model_config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    memories: dict[str, Any] = {}
    stages: dict[str, Any] = {
        "dev_calls": 0,
        "provider_or_selection_failures": 0,
        "execution_failures": 0,
        "type_failures": 0,
        "other_execution_failures": 0,
        "failure_runs": 0,
        "candidates_compiled": 0,
        "validations_passed": 0,
        "memories_promoted": 0,
    }
    failures: list[dict[str, Any]] = []
    for task in dev:
        function = str(task["function"])
        call = calls[function]
        classification = classifications[function]
        stages["dev_calls"] += 1
        if not classification["selection_ok"]:
            stages["provider_or_selection_failures"] += 1
            failures.append(
                {
                    "function": function,
                    "task_id": task["task_id"],
                    "stage": "selection_failure",
                    "detail": classification["error_message"],
                }
            )
            continue
        if classification["execution_ok"]:
            failures.append(
                {
                    "function": function,
                    "task_id": task["task_id"],
                    "stage": "no_failure",
                    "detail": "the dev call was already valid",
                }
            )
            continue
        stages["execution_failures"] += 1
        if classification["failure_category"] != "argument_type_mismatch":
            stages["other_execution_failures"] += 1
            failures.append(
                {
                    "function": function,
                    "task_id": task["task_id"],
                    "stage": "unrepaired_failure",
                    "category": classification["failure_category"],
                    "detail": classification["error_message"],
                }
            )
            continue
        stages["type_failures"] += 1
        environment = BFCLEnvironment(
            name=function,
            schema=task["schema"],
            strict=True,
            failing_args=call.arguments,
        )
        run_id = record_live_failure_run(
            container,
            workspace_id,
            environment,
            task,
            call,
            classification,
            model_config=model_config,
        )
        stages["failure_runs"] += 1
        memory, detail = compile_and_promote(
            container, workspace_id, run_id, environment, task["schema"]
        )
        if detail.get("candidate_id"):
            stages["candidates_compiled"] += 1
        if detail.get("validation") == "passed":
            stages["validations_passed"] += 1
        if memory is not None:
            stages["memories_promoted"] += 1
            memories[function] = memory
        else:
            failures.append(
                {
                    "function": function,
                    "task_id": task["task_id"],
                    "stage": detail.get("stage", "not_promoted"),
                    "detail": detail.get("detail", ""),
                }
            )
    return memories, stages, failures


def choose_memory_for_function(memories: dict[str, Any], function: str) -> Any | None:
    candidates = [memory for memory in memories.values() if function in memory.content.tool_names]
    if not candidates:
        return None
    return max(candidates, key=lambda memory: (memory.created_at, memory.memory_version))


def evaluate_arms(
    container: Container,
    workspace_id: str,
    *,
    task: dict[str, Any],
    environment: BFCLEnvironment,
    first: ModelCall,
    retry: ModelCall | None,
    first_classification: dict[str, Any],
    retry_classification: dict[str, Any] | None,
    seed: int,
    run_id: str,
    memories: dict[str, Any],
    memory_id: str | None,
    condition: str,
    has_safe_memory: bool,
    seed_temperature: float,
) -> list[dict[str, Any]]:
    function = str(task["function"])
    has_memory = has_safe_memory
    typed = typed_context(workspace_id, environment, str(task["query"]), first.arguments)
    pure = pure_context(workspace_id, str(task["query"]))
    initial_valid = bool(first_classification["execution_ok"])
    retry_valid = bool(retry_classification and retry_classification["execution_ok"])
    rows: list[dict[str, Any]] = []
    for arm in ARMS:
        result = SelectionResult(None)
        final_args = dict(first.arguments)
        final_selection_ok = bool(first_classification["selection_ok"])
        retried = False
        if arm == "retry_only":
            if not initial_valid and retry is not None and retry_classification is not None:
                retried = True
                if retry_classification["selection_ok"]:
                    final_selection_ok = True
                    final_args = dict(retry.arguments)
        elif arm != "no_memory":
            mode = RetrievalMode.BM25 if arm == "bm25" else None
            if arm == "vector":
                mode = RetrievalMode.VECTOR
            elif arm == "rrf_typed":
                mode = RetrievalMode.HYBRID_RRF
            elif arm == "full_recap":
                mode = RetrievalMode.OUTCOME_AWARE
            assert mode is not None
            context = pure if arm in {"bm25", "vector"} else typed
            decision = retrieve(
                container,
                RetrievalRequest(
                    context=context,
                    mode=mode,
                    final_k=1,
                    run_id=run_id,
                ),
            )
            result = selection_from_decision(decision, memories=memories, target_function=function)
            if result.selection is not None:
                final_args = apply_actions(
                    first.arguments, result.selection.actions, task["schema"]
                )
        final_execution = classify_execution(environment, final_args)
        call_valid = bool(final_selection_ok and final_execution["execution_ok"])
        rows.append(
            {
                "condition": condition,
                "task_id": str(task["task_id"]),
                "seed": seed,
                "cluster": function,
                "arm": arm,
                "selection_ok": bool(first_classification["selection_ok"]),
                "initial_valid": initial_valid,
                "retry_valid": retry_valid,
                "retried": retried,
                "call_valid": call_valid,
                "repaired": bool(not initial_valid and call_valid),
                "broken_valid_call": bool(initial_valid and not call_valid),
                "selected": result.selection is not None,
                "abstained": result.selection is None,
                "selected_memory_id": (
                    result.selection.memory_id if result.selection is not None else None
                ),
                "selected_tool": (result.selection.tool if result.selection is not None else ""),
                "selected_is_current": bool(
                    result.selection is not None
                    and memory_id is not None
                    and result.selection.memory_id == memory_id
                ),
                "has_safe_memory": has_memory,
                "false_apply": bool(result.selection is not None and not has_memory),
                "harmful_apply": bool(
                    result.selection is not None and not has_memory and not call_valid
                ),
                "retrieval_hit_at_1": bool(
                    result.selection is not None
                    and memory_id is not None
                    and result.selection.memory_id == memory_id
                ),
                "candidate_count": result.candidate_count,
                "eligible_count": result.eligible_count,
                "applicable_count": result.applicable_count,
                "initial_error_code": first_classification["error_code"],
                "final_error_code": final_execution["error_code"],
                "failure_category": first_classification["failure_category"],
                "selected_action_types": (
                    [str(action.get("action_type")) for action in result.selection.actions]
                    if result.selection is not None
                    else []
                ),
                "eval_temperature": seed_temperature,
            }
        )
    return rows


def run_evaluation(
    container: Container,
    workspace_id: str,
    evaluation: list[dict[str, Any]],
    memories: dict[str, Any],
    provider: Any,
    cache: dict[str, dict[str, Any]],
    *,
    model: str,
    base_url: str,
    seeds: list[int],
    temperature: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    calls_log: list[dict[str, Any]] = []
    consecutive_errors = 0
    for index, task in enumerate(evaluation, start=1):
        function = str(task["function"])
        environment = BFCLEnvironment(name=function, schema=task["schema"], strict=True)
        memory = choose_memory_for_function(memories, function)
        memory_id = memory.id if memory is not None else None
        if memory_id is None:
            conditions = [("no_memory_available", set())]
        else:
            conditions = [
                ("in_pool", set()),
                ("held_out", {memory_id}),
            ]
        for seed in seeds:
            first = call_model(
                provider,
                cache,
                model=model,
                base_url=base_url,
                task=task,
                seed=seed,
                temperature=temperature,
                kind="first",
            )
            first_classification = classify_call(task=task, call=first)
            if first.error:
                consecutive_errors += 1
                if consecutive_errors >= 3:
                    raise SystemExit(
                        "three consecutive model transport failures; "
                        f"last error: {first.error}. Is the endpoint at {base_url} running?"
                    )
            else:
                consecutive_errors = 0
            retry: ModelCall | None = None
            retry_classification: dict[str, Any] | None = None
            if not first_classification["execution_ok"]:
                feedback = RETRY_PROMPT.format(
                    error=first_classification["error_message"]
                    or "the tool was not called correctly"
                )
                retry = call_model(
                    provider,
                    cache,
                    model=model,
                    base_url=base_url,
                    task=task,
                    seed=seed,
                    temperature=temperature,
                    kind="retry",
                    previous=first,
                    feedback=feedback,
                )
                retry_classification = classify_call(task=task, call=retry)
            for condition, excluded in conditions:
                run_id = start_evaluation_run(
                    container,
                    workspace_id,
                    environment,
                    task_id=f"{workspace_id}-{task['task_id']}-{condition}",
                    seed=seed,
                    reason=f"{condition}_live_failure",
                    exclude_memory_ids=excluded,
                )
                arm_rows = evaluate_arms(
                    container,
                    workspace_id,
                    task=task,
                    environment=environment,
                    first=first,
                    retry=retry,
                    first_classification=first_classification,
                    retry_classification=retry_classification,
                    seed=seed,
                    run_id=run_id,
                    memories=memories,
                    memory_id=memory_id,
                    condition=condition,
                    has_safe_memory=bool(condition == "in_pool" and memory_id is not None),
                    seed_temperature=temperature,
                )
                rows.extend(arm_rows)
                finish_evaluation_run(
                    container,
                    run_id,
                    any(row["call_valid"] for row in arm_rows if row["arm"] == "full_recap"),
                )
            calls_log.append(
                {
                    "task_id": task["task_id"],
                    "function": function,
                    "seed": seed,
                    "first": first.to_dict(),
                    "first_classification": first_classification,
                    "retry": retry.to_dict() if retry is not None else None,
                    "retry_classification": retry_classification,
                }
            )
            print(
                f"eval {index}/{len(evaluation)} seed={seed} {function} "
                f"conditions={[name for name, _ in conditions]} "
                f"initial_valid={first_classification['execution_ok']} "
                f"retry_valid={retry_classification['execution_ok'] if retry_classification else None}",
                flush=True,
            )
    return rows, calls_log


# ------------------------------------------------------------------------ report
def _pct(value: float) -> str:
    return f"{value:.3f}"


def _ci(value: dict[str, Any]) -> str:
    interval = value["cluster_ci95"]
    return f"[{interval[0]:+.3f}, {interval[1]:+.3f}]"


def _arm_table(block: dict[str, Any]) -> list[str]:
    lines = [
        "| Arm | n | selection ok | initial valid | call valid | repair execution | "
        "broken valid | harmful apply | selected | abstained |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ARMS:
        stats = block.get(arm)
        if not stats:
            continue
        lines.append(
            f"| `{arm}` | {stats['n']} | {_pct(stats['selection_ok_rate'])} | "
            f"{_pct(stats['initial_valid_rate'])} | {_pct(stats['call_valid_rate'])} | "
            f"{_pct(stats['execution_repair_rate'])} | {_pct(stats['broken_valid_rate'])} | "
            f"{_pct(stats['harmful_apply_rate'])} | "
            f"{_pct(stats['selected_rate'])} | {_pct(stats['abstain_rate'])} |"
        )
    if block:
        lines.append("")
    return lines


def render_markdown(payload: dict[str, Any]) -> str:
    eval_block = payload["eval"]
    subsets = eval_block["subsets"]
    primary = subsets["safe_failed"]
    all_rows = subsets["all"]
    valid_rows = subsets["initially_valid"]
    failed_rows = subsets["execution_failed"]
    no_safe_failed = subsets["no_safe_failed"]
    funnel = payload["funnel"]
    discovery = payload["discovery"]["summary"]
    full = primary.get("full_recap", {})
    n_full = primary.get("no_memory", {})
    retry = primary.get("retry_only", {})
    full_all = all_rows.get("full_recap", {})
    no_all = all_rows.get("no_memory", {})
    full_typed = primary.get("rrf_typed", {})
    typed_safety = no_safe_failed.get("rrf_typed", {})
    full_safety = no_safe_failed.get("full_recap", {})
    lines = [
        "# Do real model failures get repaired by RECAP memory? (BFCL v3 live)",
        "",
        "> Generated by `scripts/bfcl_live_failures.py` from the JSON report beside this "
        "file. Do not hand-edit the numbers.",
        "",
        "## Headline",
        "",
        f"- `{payload['model']['model']}` made {funnel['dev_calls']} dev calls: "
        f"{discovery['selection_failures']} selection failures, "
        f"{discovery['execution_failures']} execution failures "
        f"({discovery['type_failures']} declared-type mismatches).",
        f"- {funnel['memories_promoted']} type-coercion memories were compiled, "
        "sandbox-validated, and promoted.",
        f"- On in-pool held-out calls whose first attempt failed on an argument type "
        f"(n={full.get('n', 0)}), `no_memory` stayed valid on "
        f"{_pct(n_full.get('call_valid_rate', 0.0))}, `retry_only` on "
        f"{_pct(retry.get('call_valid_rate', 0.0))}, and `full_recap` on "
        f"{_pct(full.get('call_valid_rate', 0.0))}.",
        f"- Across all held-out rows, `no_memory` was valid on "
        f"{_pct(no_all.get('call_valid_rate', 0.0))} and `full_recap` on "
        f"{_pct(full_all.get('call_valid_rate', 0.0))}.",
        f"- On the same in-pool subset `rrf_typed` (typed retrieval, no policy) reached "
        f"{_pct(full_typed.get('call_valid_rate', 0.0))}, so relevance alone explains the "
        "success side here.",
        f"- On rows where the function's memory was excluded and the first call failed "
        f"(n={full_safety.get('n', 0)}), relevance-only `rrf_typed` applied a cross-function "
        f"memory on {_pct(typed_safety.get('selected_rate', 0.0))} of rows and did harm on "
        f"{_pct(typed_safety.get('harmful_apply_rate', 0.0))}; `full_recap` applied one on "
        f"{_pct(full_safety.get('selected_rate', 0.0))} and did harm on "
        f"{_pct(full_safety.get('harmful_apply_rate', 0.0))}.",
        "",
        "## What is real and what is not",
        "",
        "Real: every query and function schema is BFCL v3; every failure is the local "
        "model's own output, validated against the declared JSON types and enums; every "
        "memory was recorded, compiled, sandbox-validated, and promoted by RECAP's "
        "services, then retrieved by `RecallEngine`.",
        "",
        "Not real: calls are schema-checked, not executed against live APIs. Function "
        "selection is constrained to one correct tool per task, so this measures argument "
        "execution failures and isolates memory selection. The dev split has one query per "
        "function, so the memory's argument list comes from that one observed call. "
        "Memory arms apply the selected memory deterministically; only `retry_only` asks "
        "the model again.",
        "",
        "## Provenance",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| Dataset | `{payload['dataset']}` |",
        f"| Generated | `{payload['generated_at']}` |",
        f"| Model | `{payload['model']['model']}` at `{payload['model']['base_url']}` |",
        f"| Prompt | `{payload['prompt']['version']}` |",
        f"| Discovery decoding | temperature {payload['decoding']['discovery_temperature']}, "
        f"seed {payload['decoding']['discovery_seed']} |",
        f"| Eval decoding | temperature {payload['decoding']['eval_temperature']}, "
        f"seeds {payload['decoding']['seeds']} |",
        f"| Timeouts | read {payload['decoding']['read_timeout_seconds']}s, "
        f"overall {payload['decoding']['overall_timeout_seconds']}s |",
        f"| Embedder | `{payload['embedding']['provider']}` / "
        f"`{payload['embedding']['model']}` ({payload['embedding']['dimension']}d) |",
        f"| Policy | `{payload['policy_version']}`, utility threshold "
        f"{payload['utility_threshold']:.2f} |",
        f"| Bootstrap | {payload['bootstrap_resamples']} function-level resamples |",
        "",
        "## Funnel",
        "",
        "| Stage | Count |",
        "|---|---:|",
    ]
    for key, value in funnel.items():
        if key == "promotion_failures":
            continue
        lines.append(f"| `{key}` | {value} |")
    for failure in payload["discovery"].get("promotion_failures", []):
        if failure.get("stage") == "no_failure":
            continue
        lines.append(
            f"| `{failure['function']}`: {failure.get('stage', '')} | {failure.get('detail', '')} |"
        )

    lines.extend(["", "## Held-out arms", "", "All held-out rows:", ""])
    lines.extend(_arm_table(all_rows))
    lines.extend(["", "Rows whose first held-out call failed execution:", ""])
    lines.extend(_arm_table(failed_rows))
    lines.extend(
        [
            "",
            "Rows whose first held-out call failed execution with the function's memory "
            "available (primary subset):",
            "",
        ]
    )
    lines.extend(_arm_table(primary))
    lines.extend(
        [
            "",
            "Rows with no memory for the function and a failed first call (safety subset). "
            "`full_recap` should abstain; relevance-only arms may still apply a memory:",
            "",
        ]
    )
    lines.extend(_arm_table(no_safe_failed))
    lines.extend(
        [
            "",
            "Rows whose first held-out call was already valid (harm check):",
            "",
        ]
    )
    lines.extend(_arm_table(valid_rows))

    conditions = sorted({row["condition"] for row in payload["eval"]["per_row"]})
    if conditions:
        lines.extend(
            [
                "",
                "## With vs without the function's memory",
                "",
                "`in_pool` keeps the function's own memory in the pool; `held_out` excludes "
                "exactly that memory through a real `MemorySnapshot`; "
                "`no_memory_available` is used when no memory was promoted for the function.",
                "",
                "| Condition | Arm | n | call valid | selected | harmful apply |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for condition in conditions:
            condition_rows = [
                row for row in payload["eval"]["per_row"] if row["condition"] == condition
            ]
            for arm in ("no_memory", "rrf_typed", "full_recap"):
                stats = summarize_arm([row for row in condition_rows if row["arm"] == arm])
                lines.append(
                    f"| `{condition}` | `{arm}` | {stats['n']} | "
                    f"{_pct(stats['call_valid_rate'])} | {_pct(stats['selected_rate'])} | "
                    f"{_pct(stats['harmful_apply_rate'])} |"
                )

    lines.extend(
        [
            "",
            "## Paired comparisons",
            "",
            "`full_recap` minus each arm on the same rows; positive is better. Intervals are "
            "function-level cluster bootstraps.",
            "",
            "| Subset | Reference arm | observed | 95% CI | excludes zero |",
            "|---|---|---:|---:|---|",
        ]
    )
    for subset in SUBSETS:
        block = payload["paired"].get(subset, {})
        for arm in ARMS:
            if arm in {"full_recap", "no_memory"}:
                continue
            stats = block.get(arm)
            if not stats:
                continue
            lines.append(
                f"| `{subset}` | `{arm}` | {stats['observed']:+.3f} | "
                f"{_ci(stats)} | {stats['excludes_zero']} |"
            )

    lines.extend(
        [
            "",
            "Harmful applies on the no-safe subset (`full_recap` minus each arm; negative "
            "means `full_recap` applied fewer inapplicable memories):",
            "",
            "| Reference arm | observed | 95% CI | excludes zero |",
            "|---|---:|---:|---|",
        ]
    )
    for arm, stats in payload.get("paired_harm", {}).items():
        lines.append(
            f"| `{arm}` | {stats['observed']:+.3f} | {_ci(stats)} | {stats['excludes_zero']} |"
        )

    lines.extend(
        [
            "",
            "## Failure categories",
            "",
            "| Category | Dev |",
            "|---|---:|",
        ]
    )
    for category, count in sorted(payload["discovery"]["summary"]["categories"].items()):
        lines.append(f"| `{category}` | {count} |")

    lines.extend(
        [
            "",
            "## Limits",
            "",
        ]
    )
    for limit in payload["limits"]:
        lines.append(f"- {limit}")
    lines.append("")
    return "\n".join(lines)


# -------------------------------------------------------------------------- main
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
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--doc", type=Path, default=None)
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP)
    parser.add_argument("--rng-seed", type=int, default=RNG_SEED)
    parser.add_argument("--keep", action="store_true")
    parser.add_argument("--embedding-cache", action="store_true")
    parser.add_argument("--cache-file", type=Path, default=None)
    parser.add_argument("--base-url", default="http://localhost:1234/v1")
    parser.add_argument("--api-key", default="not-needed")
    parser.add_argument("--model", default="llama-3.2-3b-instruct")
    parser.add_argument("--discovery-temperature", type=float, default=0.0)
    parser.add_argument("--discovery-seed", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seeds", default="11,23,37")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--connect-timeout", type=float, default=10.0)
    parser.add_argument("--read-timeout", type=float, default=300.0)
    parser.add_argument("--overall-timeout", type=float, default=600.0)
    parser.add_argument("--no-progress-timeout", type=float, default=180.0)
    parser.add_argument("--max-retries", type=int, default=1)
    args = parser.parse_args(argv)

    if not args.data_dir.is_dir():
        raise SystemExit(
            f"{args.data_dir} does not exist. Download BFCL v3 into it (see the module "
            "docstring) or pass --data-dir."
        )
    if args.embedding == "qwen" and args.model_path is None:
        raise SystemExit("--model-path is required for --embedding qwen")
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    if len(seeds) < 3:
        raise SystemExit("--seeds must list at least three seeds")

    tasks, hashes = load_tasks(args.data_dir)
    dev, evaluation, skipped = split_tasks(tasks)
    if args.limit is not None:
        dev = dev[: args.limit]
        keep = {row["function"] for row in dev}
        evaluation = [row for row in evaluation if row["function"] in keep]
    if len(dev) < 4:
        raise SystemExit("too few functions with two distinct queries")
    print(
        f"tasks={len(tasks)} functions={len({row['function'] for row in tasks})} "
        f"dev={len(dev)} eval={len(evaluation)} skipped={len(skipped)}"
    )

    args.index_prefix = args.index_prefix or f"{DEFAULT_PREFIX}-{args.embedding}"
    args.workspace = args.workspace or f"bfcl-live-{args.embedding}"

    settings = build_settings(args)
    container = Container.build(settings)
    container.migrate()
    container.init_search()
    container.init_search_chunks()
    if not args.keep:
        reset_workspace(container, [args.workspace])

    provider = build_provider(args)
    cache: dict[str, dict[str, Any]] = {}
    if args.cache_file is not None and args.cache_file.is_file():
        cache = json.loads(args.cache_file.read_text(encoding="utf-8"))
        print(f"loaded {len(cache)} cached model calls from {args.cache_file}")

    model_config = {
        "model": args.model,
        "base_url": redact_url(args.base_url),
        "discovery_temperature": args.discovery_temperature,
        "discovery_seed": args.discovery_seed,
    }

    dev_calls: dict[str, ModelCall] = {}
    dev_classifications: dict[str, dict[str, Any]] = {}
    consecutive_errors = 0
    for index, task in enumerate(dev, start=1):
        call = call_model(
            provider,
            cache,
            model=args.model,
            base_url=args.base_url,
            task=task,
            seed=args.discovery_seed,
            temperature=args.discovery_temperature,
            kind="dev",
        )
        if call.error:
            consecutive_errors += 1
            if consecutive_errors >= 3:
                raise SystemExit(
                    "three consecutive model transport failures; "
                    f"last error: {call.error}. Is the endpoint at {args.base_url} running?"
                )
        else:
            consecutive_errors = 0
        classification = classify_call(task=task, call=call)
        dev_calls[str(task["function"])] = call
        dev_classifications[str(task["function"])] = classification
        print(
            f"dev {index}/{len(dev)} {task['function']} "
            f"select={classification['selection_ok']} "
            f"exec={classification['execution_ok']} "
            f"category={classification['failure_category']}",
            flush=True,
        )
        if args.cache_file is not None:
            args.cache_file.parent.mkdir(parents=True, exist_ok=True)
            args.cache_file.write_text(json.dumps(cache, indent=2), encoding="utf-8")

    memories, stages, promotion_failures = promote_pool(
        container,
        args.workspace,
        dev,
        dev_calls,
        dev_classifications,
        model_config=model_config,
    )
    active = active_memories(container, args.workspace)
    print(f"promoted={len(memories)} active={len(active)} type_failures={stages['type_failures']}")

    rows, calls_log = run_evaluation(
        container,
        args.workspace,
        evaluation,
        active,
        provider,
        cache,
        model=args.model,
        base_url=args.base_url,
        seeds=seeds,
        temperature=args.temperature,
    )
    if args.cache_file is not None:
        args.cache_file.parent.mkdir(parents=True, exist_ok=True)
        args.cache_file.write_text(json.dumps(cache, indent=2), encoding="utf-8")

    rng = np.random.default_rng(args.rng_seed)
    subsets_payload: dict[str, Any] = {}
    for subset in SUBSETS:
        subset_rows_list = subset_rows(rows, subset)
        block: dict[str, Any] = {}
        for arm in ARMS:
            arm_rows = [row for row in subset_rows_list if row["arm"] == arm]
            stats = summarize_arm(arm_rows)
            values: dict[str, list[float]] = defaultdict(list)
            for row in arm_rows:
                values[str(row["cluster"])].append(1.0 if row["call_valid"] else 0.0)
            stats["call_valid_ci95"] = cluster_bootstrap(values, rng, resamples=args.bootstrap)[
                "cluster_ci95"
            ]
            block[arm] = stats
        subsets_payload[subset] = block

    paired: dict[str, Any] = {}
    for subset in SUBSETS:
        subset_rows_list = subset_rows(rows, subset)
        treatment = [row for row in subset_rows_list if row["arm"] == "full_recap"]
        block = {}
        for arm in ARMS:
            if arm in {"full_recap", "no_memory"}:
                continue
            reference = [row for row in subset_rows_list if row["arm"] == arm]
            if not treatment or not reference:
                continue
            block[arm] = paired_bootstrap(
                treatment,
                reference,
                cluster_key="cluster",
                value_key="call_valid",
                rng=rng,
                resamples=args.bootstrap,
            )
        paired[subset] = block

    paired_harm: dict[str, Any] = {}
    harm_rows = subset_rows(rows, "no_safe_failed")
    full_harm = [row for row in harm_rows if row["arm"] == "full_recap"]
    for arm in ARMS:
        if arm in {"full_recap", "no_memory"}:
            continue
        reference = [row for row in harm_rows if row["arm"] == arm]
        if not full_harm or not reference:
            continue
        paired_harm[arm] = paired_bootstrap(
            full_harm,
            reference,
            cluster_key="cluster",
            value_key="harmful_apply",
            rng=rng,
            resamples=args.bootstrap,
        )
    categories: dict[str, int] = defaultdict(int)
    for classification in dev_classifications.values():
        categories[str(classification["failure_category"])] += 1
    discovery_summary = {
        "dev_calls": len(dev),
        "selection_failures": sum(
            1 for item in dev_classifications.values() if not item["selection_ok"]
        ),
        "execution_failures": sum(
            1
            for item in dev_classifications.values()
            if item["selection_ok"] and not item["execution_ok"]
        ),
        "type_failures": sum(
            1
            for item in dev_classifications.values()
            if item["failure_category"] == "argument_type_mismatch"
        ),
        "categories": dict(sorted(categories.items())),
    }

    funnel = {
        "labeled_tasks": len(tasks),
        "functions_with_two_queries": len(dev),
        "dev_calls": stages["dev_calls"],
        "dev_selection_failures": stages["provider_or_selection_failures"],
        "dev_execution_failures": stages["execution_failures"],
        "dev_type_failures": stages["type_failures"],
        "dev_other_execution_failures": stages["other_execution_failures"],
        "failure_runs": stages["failure_runs"],
        "candidates_compiled": stages["candidates_compiled"],
        "validations_passed": stages["validations_passed"],
        "memories_promoted": stages["memories_promoted"],
        "active_pool": len(active),
        "eval_tasks": len(evaluation),
        "eval_first_calls": len(calls_log),
        "eval_conditions": len({row["condition"] for row in rows}),
        "eval_rows": len(rows),
        "eval_selection_failures": sum(
            1 for call in calls_log if not call["first_classification"]["selection_ok"]
        ),
        "eval_initial_execution_failures": sum(
            1
            for call in calls_log
            if call["first_classification"]["selection_ok"]
            and not call["first_classification"]["execution_ok"]
        ),
        "promotion_failures": promotion_failures,
    }

    payload: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset": "gorilla-llm/Berkeley-Function-Calling-Leaderboard (BFCL v3)",
        "source_hashes": hashes,
        "index_prefix": args.index_prefix,
        "workspace": args.workspace,
        "model": {
            "provider": "openai_compatible",
            "model": args.model,
            "base_url": redact_url(args.base_url),
            "api_key_configured": bool(args.api_key),
        },
        "prompt": {
            "version": PROMPT_VERSION,
            "system": SYSTEM_PROMPT,
            "retry": RETRY_PROMPT,
        },
        "decoding": {
            "discovery_temperature": args.discovery_temperature,
            "discovery_seed": args.discovery_seed,
            "eval_temperature": args.temperature,
            "seeds": seeds,
            "max_tokens": args.max_tokens,
            "read_timeout_seconds": args.read_timeout,
            "overall_timeout_seconds": args.overall_timeout,
            "max_retries": args.max_retries,
        },
        "embedding": {
            "provider": settings.embeddings.provider,
            "model": getattr(container.embeddings, "model_id", settings.embeddings.model_id),
            "dimension": settings.embeddings.dimension,
            "query_prefix": settings.embeddings.query_prefix,
        },
        "policy_version": POLICY_VERSION,
        "utility_threshold": UTILITY_THRESHOLD,
        "bootstrap_resamples": args.bootstrap,
        "rng_seed": args.rng_seed,
        "split": {
            "dev": len(dev),
            "eval": len(evaluation),
            "functions": len({row["function"] for row in dev}),
            "skipped": skipped,
        },
        "funnel": funnel,
        "discovery": {
            "summary": discovery_summary,
            "per_function": [
                {
                    "function": function,
                    "task_id": dev_calls[function].task_id,
                    "call": dev_calls[function].to_dict(),
                    "classification": dev_classifications[function],
                }
                for function in sorted(dev_calls)
            ],
            "promotion_failures": promotion_failures,
        },
        "eval": {
            "results": subsets_payload["all"],
            "subsets": subsets_payload,
            "paired": paired,
            "paired_harm": paired_harm,
            "per_row": rows,
            "per_call": calls_log,
        },
        "paired": paired,
        "paired_harm": paired_harm,
        "limits": [
            "Calls are validated against the real BFCL schema (declared types, enums, "
            "required parameters); they are not executed against live APIs.",
            "Function selection is constrained to one correct tool per task. Selection "
            "failures here mean the model did not call that tool; cross-tool confusion is "
            "out of scope.",
            "Discovery uses one query per function, so a promoted memory only lists the "
            "arguments that were wrong in that one call. A different held-out query may "
            "make a different optional argument fail.",
            "Memory arms apply the selected typed action deterministically. Only "
            "`retry_only` asks the model again, with the schema error as feedback.",
            "Only simple and exec_simple BFCL files are available locally. Multi-turn and "
            "irrelevance categories are not covered.",
            "The local model runs at temperature 0 during discovery and nonzero temperature "
            "during evaluation; a server that ignores `seed` makes evaluation draws "
            "independent rather than exactly reproducible.",
            "Promotion hardcodes environment_type=mock_api and stores no schema path hashes, "
            "so field-scoped compatibility is not exercised.",
            "The utility threshold is the shipped default (0.35); no weight or threshold was "
            "tuned on these outcomes.",
        ],
    }

    args.out.mkdir(parents=True, exist_ok=True)
    stem = f"real-data-bfcl-live-{args.model.replace('/', '-').replace('.', '-')}"
    json_path = args.out / f"{stem}.json"
    doc_path = args.doc or args.out / f"{stem}.md"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    doc_path.write_text(render_markdown(payload), encoding="utf-8")
    print(f"wrote {json_path}")
    print(f"wrote {doc_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
