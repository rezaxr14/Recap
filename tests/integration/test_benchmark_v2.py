"""End-to-end smoke test for the ten-family benchmark v2 suite.

This intentionally runs only four tasks (one per dev family/version). It proves
that both environment versions stay active at the same time, that the v2
dataset is wired into the real runner, and that the version-filter baseline is
evaluated with the same task/agent path as the other baselines. The full dev
and test matrices are run manually and committed under ``docs/evidence``.
"""

from __future__ import annotations

import uuid

import pytest

from recap.domain.enums import MemoryStatus
from recap.evals.datasets import build_benchmark_v2_tasks
from recap.evals.runner import _seed_memories, run_benchmark
from recap.maintenance import reset_workspace

pytestmark = pytest.mark.integration


def test_v2_dev_smoke_keeps_versions_distinct_and_measures_safety(container, tmp_path) -> None:
    dev = [task for task in build_benchmark_v2_tasks() if task.split == "dev"]
    selected = [
        next(
            task
            for task in dev
            if task.task_family == family and task.environment_version == version
        )
        for family in ("idempotency_key", "response_field")
        for version in ("1", "2")
    ]

    result = run_benchmark(
        container,
        suite="v2",
        split="dev",
        tasks=selected,
        output_dir=tmp_path,
        cleanup=True,
    )
    summary = result["summary"]
    rows = [row for row in result["per_task"] if row["baseline"] == "outcome_aware"]

    # Both versions are seeded and remain relevant; the old supersession rule
    # retired v1 when v2 was promoted, which made the v1 tasks unanswerable.
    assert len(rows) == 4
    assert all(row["relevant_memory_ids"] for row in rows)
    assert {row["environment_version"] for row in rows} == {"1", "2"}

    # The full policy recovers every task without selecting a stale memory.
    assert summary["outcome_aware"]["task_success_rate"] == 1.0
    assert summary["outcome_aware"]["harmful_transfer_rate"] == 0.0
    assert summary["outcome_aware"]["incompatible_selection_rate"] == 0.0
    assert summary["outcome_aware"]["false_abstention_rate"] == 0.0

    # Explicit version filtering is a fair, non-policy baseline. It should fix
    # the stale-version selection without importing the full policy.
    assert summary["hybrid_rrf_version_filter"]["incompatible_selection_rate"] == 0.0
    assert summary["hybrid_rrf_version_filter"]["task_success_rate"] == 1.0

    # Unfiltered card retrieval selects the stale memory and transfers it.
    assert summary["bm25"]["incompatible_selection_rate"] == 1.0
    assert summary["bm25"]["harmful_transfer_rate"] == 0.5
    assert summary["bm25"]["hard_negative_outrank_rate"] == 0.5

    # The environment is not solvable from retries or from nothing.
    assert summary["no_memory"]["task_success_rate"] == 0.0
    assert summary["retry_only"]["task_success_rate"] == 0.0


def test_v2_seeding_adds_an_active_bad_track_record_distractor(container) -> None:
    workspace = f"bench-fixture-{uuid.uuid4().hex[:10]}"
    tasks = [
        task
        for task in build_benchmark_v2_tasks()
        if task.split == "dev" and task.environment_version == "1"
    ][:2]
    try:
        _seed_memories(container, workspace, tasks)
        with container.services() as services:
            memories = services.memories.list_memories(workspace, limit=100)
        expected_ids = {
            f"mock-api-{task.failure_mode}-v{task.environment_version}" for task in tasks
        }
        distractors = [
            memory
            for memory in memories
            if memory.environment_id not in expected_ids and memory.stats.verified_failures >= 2
        ]
        assert distractors, [(m.environment_id, m.status) for m in memories]
        assert all(memory.status is MemoryStatus.ACTIVE for memory in distractors)
        assert all(memory.stats.consecutive_failures < 3 for memory in distractors)
    finally:
        reset_workspace(container, [workspace])
