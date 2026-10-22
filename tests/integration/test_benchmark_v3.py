"""Live smoke test for the v3 suite.

The full 2,000-fixture store and all seeds are run manually and published under
``docs/evidence``. This test uses a small store so CI keeps the same runner and
seeding path without adding minutes to every integration run.
"""

from __future__ import annotations

import pytest

from recap.evals.datasets import build_benchmark_v3_tasks, split_hash
from recap.evals.runner import run_benchmark

pytestmark = pytest.mark.integration


def test_v3_dev_smoke_proves_silent_drift_needs_memory(container, tmp_path) -> None:
    all_tasks = build_benchmark_v3_tasks()
    selected = [
        next(task for task in all_tasks if task.tier == "B"),
        next(
            task
            for task in all_tasks
            if task.scenario_id == "tier-c-silent_cursor" and task.schema_variant == ""
        ),
        next(task for task in all_tasks if task.scenario_id == "tier-c-silent-cursor-irrelevant"),
        next(task for task in all_tasks if task.scenario_id == "tier-c-silent-cursor-relevant"),
    ]
    selected_ids = {task.task_id for task in selected}
    unique = [task for task in selected if task.task_id in selected_ids]

    result = run_benchmark(
        container,
        suite="v3",
        split="dev",
        tasks=unique,
        baselines=["no_memory", "hybrid_rrf_version_filter", "outcome_aware"],
        output_dir=tmp_path,
        cleanup=True,
        fixture_count=24,
    )
    summary = result["summary"]
    rows = {(row["task_id"], row["baseline"]): row for row in result["per_task"]}

    # Tier B is intentionally opaque but discoverable by probing, so it is the
    # no-memory positive control in this sample.
    control_id = unique[0].task_id
    assert rows[(control_id, "no_memory")]["task_success"] is True

    # Silent drift has no error and is not solvable by retrying or probing.
    silent = next(task for task in unique if task.scenario_id == "tier-c-silent_cursor")
    assert rows[(silent.task_id, "no_memory")]["task_success"] is False

    # Full policy selects the gold memory and succeeds. The exact-fingerprint
    # oracle rejects the irrelevant-drift task, demonstrating why field scope is
    # needed instead of a whole-fingerprint gate.
    # Three selectable tasks succeed. The relevant-drift task is deliberately
    # unanswerable and must be a safe abstention, not a harmful transfer.
    assert summary["outcome_aware"]["task_success_rate"] == 0.75
    assert summary["outcome_aware"]["false_abstention_rate"] == 0.0
    assert summary["outcome_aware"]["selection_recall"] == 1.0
    assert summary["outcome_aware"]["harmful_transfer_rate"] == 0.0
    relevant = next(task for task in unique if task.scenario_id == "tier-c-silent-cursor-relevant")
    assert rows[(relevant.task_id, "outcome_aware")]["task_success"] is False
    assert rows[(relevant.task_id, "outcome_aware")]["selected_memory_ids"] == []
    irrelevant = next(
        task for task in unique if task.scenario_id == "tier-c-silent-cursor-irrelevant"
    )
    assert rows[(irrelevant.task_id, "outcome_aware")]["task_success"] is True

    # The task dataset is stable and the flag store is reported explicitly.
    assert result["dataset_hash"] == split_hash(unique)
    assert result["store"]["synthetic_fixtures"] == 24
    assert result["store"]["provenance"] == "real_gold_synthetic_decoys"
