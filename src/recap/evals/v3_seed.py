"""Seed the v3 benchmark workspace.

Gold memories are deterministic, explicitly labelled ``synthetic_fixture`` (the
benchmark must not pretend they came from production traffic), and they carry
field-scoped schema hashes. Bulk decoys create a realistic, noisy store.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from typing import Any, cast

from recap.container import Container
from recap.domain.enums import CreatedFrom, MemoryStatus, RiskLevel, SideEffectClass
from recap.domain.memories import ExperienceMemory, MemoryContent, OutcomeStats
from recap.domain.predicates import Predicate
from recap.domain.strategies import RepairStrategy, StrategyAction
from recap.environments.v3_api import V3_FAILURE_FAMILIES, V3Environment
from recap.evals.datasets import BenchmarkTask
from recap.infrastructure.elasticsearch.search_repository import (
    ElasticsearchMemoryRepository,
)
from recap.logging import get_logger
from recap.providers.base import EmbeddingProvider

logger = get_logger(__name__)

GOLD_DEFINITIONS: dict[str, dict[str, Any]] = {
    "opaque_required": {
        "title": "Opaque request rejection: supply account_id before listing records",
        "summary": (
            "When the records endpoint rejects a request with a reference id instead of a "
            "field name, the deployment expects an account_id. Send the account id with the "
            "tenant and continue pagination normally."
        ),
        "actions": [
            StrategyAction(
                action_type="supply_required_parameter",  # type: ignore[arg-type]
                tool_name="list_records",
                description="Add account_id=acct-bench to the list request.",
                parameters={"account_id": "acct-bench"},
                side_effect_class=SideEffectClass.READ_ONLY,
            )
        ],
    },
    "silent_cursor": {
        "title": "Pagination silently ignores the legacy cursor; follow page_token",
        "summary": (
            "The records endpoint accepts a legacy cursor and returns the first page again. "
            "It does not report an error. Drop the legacy cursor and follow the page_token "
            "returned in each response until has_more is false."
        ),
        "actions": [
            StrategyAction(
                action_type="map_parameter",  # type: ignore[arg-type]
                tool_name="list_records",
                description="Use page_token instead of the ignored cursor parameter.",
                parameters={"from": "cursor", "to": "page_token"},
                side_effect_class=SideEffectClass.READ_ONLY,
            )
        ],
    },
    "silent_field": {
        "title": "Records are under the data response field; legacy items is empty",
        "summary": (
            "The service still accepts the old request and returns an empty items array. The "
            "actual records are in the data field. Read data and report the complete count; "
            "there is no error to signal the change."
        ),
        "actions": [
            StrategyAction(
                action_type="map_response_field",  # type: ignore[arg-type]
                tool_name="list_records",
                description="Read records from the data field.",
                parameters={"field": "data"},
                side_effect_class=SideEffectClass.READ_ONLY,
            )
        ],
    },
    "silent_enum": {
        "title": "Sort value created silently reverses order; use created_at",
        "summary": (
            "Passing sort=created is accepted but the service falls back to reverse insertion "
            "order. Passing created_at produces the canonical order. No error is returned."
        ),
        "actions": [
            StrategyAction(
                action_type="map_parameter_value",  # type: ignore[arg-type]
                tool_name="list_records",
                description="Use the canonical sort value created_at.",
                parameters={"field": "sort", "to": "created_at"},
                side_effect_class=SideEffectClass.READ_ONLY,
            )
        ],
    },
    "silent_account": {
        "title": "Complete tenant records require the acct-north account convention",
        "summary": (
            "Without account_id the service returns an empty page and no error. This tenant's "
            "records are stored under the organisation account acct-north. Supply that value "
            "once and paginate normally."
        ),
        "actions": [
            StrategyAction(
                action_type="supply_required_parameter",  # type: ignore[arg-type]
                tool_name="list_records",
                description="Supply the organisation account id.",
                parameters={"account_id": "acct-north"},
                side_effect_class=SideEffectClass.READ_ONLY,
            )
        ],
    },
}


def _gold_memory(task: BenchmarkTask, container: Container) -> ExperienceMemory | None:
    if not task.gold_memory_ids:
        return None
    definition = GOLD_DEFINITIONS.get(task.failure_mode)
    if definition is None:
        return None
    memory_id = task.gold_memory_ids[0]
    environment = V3Environment(
        version=task.environment_version,
        failure_family=task.failure_mode,
    )
    fingerprint = environment.fingerprint()
    path_hashes = environment.schema_path_hashes()
    preconditions = [
        Predicate(
            field=f"schema_path_hashes.{path}",
            operator="eq",
            value=digest,
            description="relevant schema path matches the validated version",
        )
        for path, digest in path_hashes.items()
    ]
    if task.failure_mode == "opaque_required":
        preconditions.append(
            Predicate(
                field="error_code",
                operator="eq",
                value="INVALID_REQUEST",
                description="opaque request rejection",
            )
        )
    content = MemoryContent(
        title=f"{definition['title']} [{task.scenario_id}]",
        summary=f"{definition['summary']} Scenario fixture: {task.scenario_id}.",
        failure_signature="SilentBehaviorError|COMPLETION_CHECK_FAILED|list_records",
        failure_code="INVALID_REQUEST"
        if task.failure_mode == "opaque_required"
        else "COMPLETION_CHECK_FAILED",
        task_families=[task.failure_mode],
        tool_names=["list_records"],
        tool_versions={"list_records": task.environment_version},
        tool_schema_fingerprints={"list_records": fingerprint.tool_schema_hashes["list_records"]},
        schema_path_hashes=environment.schema_path_hashes(),
        environment_type="mock_api",
        preconditions=preconditions,
        strategy=RepairStrategy(summary=definition["summary"], actions=definition["actions"]),
        postconditions=[
            Predicate(
                field="records_collected",
                operator="gte",
                value=len(environment.records),
                description="complete record set collected",
            )
        ],
        known_limitations=["Synthetic benchmark fixture; not production evidence."],
    )
    return ExperienceMemory(
        id=memory_id,
        workspace_id="",
        content=content,
        status=MemoryStatus.ACTIVE,
        created_from=CreatedFrom.IMPORTED,
        risk_level=RiskLevel.LOW,
        environment_fingerprint=fingerprint.fingerprint,
        environment_id=fingerprint.environment_id,
        embedding_model_id=str(getattr(container.embeddings, "model_id", "")),
        embedding_model_version=str(getattr(container.embeddings, "model_version", "1")),
        embedding_dimension=container.settings.elasticsearch.vector_dimension,
        index_generation=container.settings.elasticsearch.index_generation,
        stats=OutcomeStats(verified_successes=8, verified_failures=0),
        evidence_run_ids=[f"fixture:{memory_id}"],
        last_validated_at=datetime.now(UTC),
    )


def _decoy_memories(workspace_id: str, container: Container, count: int) -> list[ExperienceMemory]:
    """Create a noisy, realistic store.

    The first third look like stale versions of the gold families, the second
    third are wrong-family memories with overlapping vocabulary, and the rest
    have a bad track record. They are explicitly synthetic fixtures.
    """
    memories: list[ExperienceMemory] = []
    families = list(V3_FAILURE_FAMILIES)
    for index in range(count):
        family = families[index % len(families)]
        stale = index % 3 == 0
        bad_track_record = index % 7 == 0
        version = "1" if stale else "2"
        environment = V3Environment(version=version, failure_family=family)
        fingerprint = environment.fingerprint()
        variant_text = (
            "stale legacy pagination cursor bug"
            if stale
            else (
                "record collection wrong family overlapping vocabulary"
                if index % 3 == 1
                else "pagination retry recovery"
            )
        )
        content = MemoryContent(
            title=f"Pagination recovery note {index:04d}: {variant_text}",
            summary=(
                f"Collected tenant records with {family.replace('_', ' ')} behavior. "
                "This fixture mentions cursor, page token, account, sort, records, count and "
                "retry so it is a realistic near-duplicate for lexical retrieval."
            ),
            failure_signature="SilentBehaviorError|COMPLETION_CHECK_FAILED|list_records",
            failure_code="COMPLETION_CHECK_FAILED",
            task_families=[family, "pagination"],
            tool_names=["list_records"],
            tool_versions={"list_records": version},
            tool_schema_fingerprints={
                "list_records": fingerprint.tool_schema_hashes["list_records"]
            },
            environment_type="mock_api",
            preconditions=[],
            strategy=RepairStrategy(
                summary="Retry pagination and verify the complete record count.",
                actions=[
                    StrategyAction(
                        action_type="retry_with_backoff",  # type: ignore[arg-type]
                        tool_name="list_records",
                        description="Retry the read-only request with bounded backoff.",
                        max_attempts=2,
                        side_effect_class=SideEffectClass.READ_ONLY,
                    )
                ],
            ),
            postconditions=[Predicate(field="records_collected", operator="gte", value=1)],
            known_limitations=["Synthetic fixture; not production evidence."],
        )
        stats = OutcomeStats(
            verified_successes=0 if bad_track_record else 2,
            verified_failures=4 if bad_track_record else 0,
            unknown=1,
        )
        memories.append(
            ExperienceMemory(
                id=f"v3-decoy-{index:04d}",
                workspace_id=workspace_id,
                content=content,
                status=MemoryStatus.ACTIVE,
                created_from=CreatedFrom.IMPORTED,
                risk_level=RiskLevel.LOW,
                environment_fingerprint=fingerprint.fingerprint,
                environment_id=fingerprint.environment_id,
                embedding_model_id=str(getattr(container.embeddings, "model_id", "")),
                embedding_model_version=str(getattr(container.embeddings, "model_version", "1")),
                embedding_dimension=container.settings.elasticsearch.vector_dimension,
                index_generation=container.settings.elasticsearch.index_generation,
                stats=stats,
                evidence_run_ids=[f"fixture:v3-decoy-{index:04d}"],
            )
        )
    return memories


def build_v3_corpus(
    container: Container,
    workspace_id: str,
    tasks: list[BenchmarkTask],
    *,
    fixture_count: int = 2000,
) -> tuple[list[ExperienceMemory], list[BenchmarkTask], dict[str, str]]:
    """Build the in-memory v3 corpus without touching Postgres or Elasticsearch.

    Memory ids are namespaced by workspace. Postgres uses a global primary key
    for ``memories.id``, so two benchmark processes sharing one database would
    otherwise collide on the deterministic fixture ids even though their
    workspaces are different. Gold tasks are returned with their ids remapped.
    """
    import hashlib

    namespace = hashlib.sha256(workspace_id.encode()).hexdigest()[:8]

    def scoped(memory_id: str) -> str:
        return f"{namespace}-{memory_id}"

    gold: list[ExperienceMemory] = []
    seen: set[str] = set()
    id_map: dict[str, str] = {}
    for task in tasks:
        memory = _gold_memory(task, container)
        if memory is None:
            continue
        original_id = memory.id
        memory.id = scoped(original_id)
        memory.family_id = memory.id
        memory.retrieval_document_id = f"{memory.id}:v{memory.memory_version}"
        id_map[original_id] = memory.id
        if memory.id in seen:
            continue
        seen.add(memory.id)
        memory.workspace_id = workspace_id
        gold.append(memory)

    updated_tasks = list(tasks)
    for index, task in enumerate(tasks):
        if task.gold_memory_ids:
            updated_tasks[index] = replace(
                task,
                gold_memory_ids=tuple(
                    id_map.get(memory_id, memory_id) for memory_id in task.gold_memory_ids
                ),
            )

    decoys = _decoy_memories(workspace_id, container, fixture_count)
    for memory in decoys:
        memory.id = scoped(memory.id)
        memory.family_id = memory.id
        memory.retrieval_document_id = f"{memory.id}:v{memory.memory_version}"
    return gold + decoys, updated_tasks, id_map


def seed_v3_workspace(
    container: Container,
    workspace_id: str,
    tasks: list[BenchmarkTask],
    *,
    fixture_count: int = 2000,
) -> dict[str, Any]:
    """Create gold memories and bulk decoys, then index them in one batch."""
    all_memories, updated_tasks, id_map = build_v3_corpus(
        container, workspace_id, tasks, fixture_count=fixture_count
    )
    # The runner and its per-task rows use the remapped gold ids. Preserve the
    # historical in-place contract so callers cannot accidentally keep the
    # unscoped ids after seeding.
    tasks[:] = updated_tasks
    from recap.domain.memories import build_embedding_text
    from recap.infrastructure.db.repositories import ensure_workspace

    with container.services() as services:
        ensure_workspace(services.session, workspace_id)
        for memory in all_memories:
            persisted = services.memories.create_memory(memory)
            memory.id = persisted.id
    embeddings = cast(EmbeddingProvider, container.embeddings).embed_documents(
        [build_embedding_text(memory) for memory in all_memories]
    )
    container.search_repository.bulk_index(list(zip(all_memories, embeddings, strict=True)))
    return {
        "gold_memories": len(id_map),
        "synthetic_fixtures": fixture_count,
        "total": len(all_memories),
        "provenance": "real_gold_synthetic_decoys",
    }


def ensure_v3_index(container: Container) -> None:
    repository: ElasticsearchMemoryRepository = container.search_repository
    repository.ensure_ready()
