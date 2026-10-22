"""Unit tests for the v3 tiered benchmark contract.

Live end-to-end execution is covered by ``tests/integration/test_benchmark_v3``.
These tests keep the dataset and deterministic environment honest without a
database or Elasticsearch.
"""

from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

from recap.domain.enums import CompatibilityStatus
from recap.domain.retrieval import RetrievalContext
from recap.environments.reference_agent import ReferenceAgent
from recap.environments.v3_api import V3Environment
from recap.evals.datasets import build_benchmark_v3_tasks, split_hash
from recap.recall.applicability import field_scoped_compatibility


def test_v3_dataset_is_deterministic_and_tiered() -> None:
    first = build_benchmark_v3_tasks()
    second = build_benchmark_v3_tasks()
    assert [task.to_dict() for task in first] == [task.to_dict() for task in second]
    assert split_hash(first) == split_hash(second)
    assert {task.tier for task in first} == {"A", "B", "C"}
    assert {task.split for task in first} == {"train", "dev", "test"}
    assert len({task.task_id for task in first}) == len(first)


def test_v3_instructions_never_name_the_drift_or_the_fix() -> None:
    forbidden = {
        "cursor",
        "page_token",
        "account_id",
        "sort",
        "created_at",
        "data field",
        "parameter",
        "drift",
        "schema",
        "version",
        "retry",
        "memory",
    }
    for task in build_benchmark_v3_tasks():
        if task.tier == "A":
            continue
        lowered = task.instruction.lower()
        leaked = sorted(word for word in forbidden if word in lowered)
        assert not leaked, (task.task_id, leaked, task.instruction)


def test_v3_test_tier_c_has_no_memory_gate_candidates() -> None:
    test_tier_c = [
        task for task in build_benchmark_v3_tasks() if task.split == "test" and task.tier == "C"
    ]
    assert len(test_tier_c) >= 12
    assert Counter(task.should_select for task in test_tier_c) == Counter({True: 18, False: 6})


def test_reference_agent_fails_every_silent_family_without_memory() -> None:
    tasks = [
        task
        for task in build_benchmark_v3_tasks()
        if task.silent_drift and task.schema_variant == ""
    ]
    assert {task.failure_mode for task in tasks} >= {
        "silent_cursor",
        "silent_field",
        "silent_enum",
        "silent_account",
        "novel_failure",
    }
    for task in tasks:
        environment = V3Environment(
            version=task.environment_version,
            failure_family=task.failure_mode,
            tenant=task.tenant,
        )
        result = ReferenceAgent(tenant=task.tenant).run(environment, max_tool_calls=3)
        assert not result.success, (task.task_id, result.ordered_record_ids)


def test_tier_a_control_succeeds_without_memory_and_tier_b_probes() -> None:
    tasks = build_benchmark_v3_tasks()
    control = next(task for task in tasks if task.tier == "A")
    opaque = next(task for task in tasks if task.tier == "B")

    control_env = V3Environment(
        version=control.environment_version,
        failure_family=control.failure_mode,
        tenant=control.tenant,
    )
    control_result = ReferenceAgent(tenant=control.tenant).run(control_env)
    assert control_result.success

    opaque_env = V3Environment(
        version=opaque.environment_version,
        failure_family=opaque.failure_mode,
        tenant=opaque.tenant,
    )
    opaque_result = ReferenceAgent(tenant=opaque.tenant).run(opaque_env, max_tool_calls=3)
    assert opaque_result.success
    assert opaque_result.retries == 1


def test_field_scoped_compatibility_separates_irrelevant_from_relevant_drift() -> None:
    base = V3Environment(version="2", failure_family="silent_cursor")
    irrelevant = V3Environment(
        version="2", failure_family="silent_cursor", schema_variant="irrelevant"
    )
    relevant = V3Environment(version="2", failure_family="silent_cursor", schema_variant="relevant")
    assert base.schema_path_hashes() == irrelevant.schema_path_hashes()
    assert base.schema_path_hashes() != relevant.schema_path_hashes()

    memory = SimpleNamespace(content=SimpleNamespace(schema_path_hashes=base.schema_path_hashes()))
    context = RetrievalContext(
        workspace_id="test",
        query="collect records",
        schema_path_hashes=irrelevant.schema_path_hashes(),
    )
    status, reason = field_scoped_compatibility(memory, context)  # type: ignore[arg-type]
    assert status is CompatibilityStatus.COMPATIBLE, reason

    context.schema_path_hashes = relevant.schema_path_hashes()
    status, reason = field_scoped_compatibility(memory, context)  # type: ignore[arg-type]
    assert status is CompatibilityStatus.INCOMPATIBLE, reason
    assert "relevant schema path changed" in reason
