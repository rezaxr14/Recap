from __future__ import annotations

import pytest

from recap.evals.datasets import check_leakage, default_tasks, split_hash
from recap.evals.metrics import (
    bootstrap_ci,
    mean,
    ndcg_at_k,
    paired_difference_ci,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


def test_recall_and_precision() -> None:
    relevant = {"a", "b"}
    retrieved = ["x", "a", "y", "b"]
    assert recall_at_k(relevant, retrieved, 2) == pytest.approx(0.5)
    assert recall_at_k(relevant, retrieved, 4) == pytest.approx(1.0)
    assert precision_at_k(relevant, retrieved, 4) == pytest.approx(0.5)
    assert recall_at_k(set(), retrieved, 4) == 0.0


def test_mrr_uses_first_relevant_rank() -> None:
    assert reciprocal_rank({"b"}, ["x", "y", "b"]) == pytest.approx(1 / 3)
    assert reciprocal_rank({"z"}, ["x", "y"]) == 0.0


def test_ndcg_is_one_for_ideal_ordering() -> None:
    assert ndcg_at_k({"a", "b"}, ["a", "b", "c"], 3) == pytest.approx(1.0)
    assert ndcg_at_k({"a"}, ["x", "y"], 2) == 0.0


def test_bootstrap_ci_bounds_mean() -> None:
    lower, upper = bootstrap_ci([1.0, 0.0, 1.0, 1.0], samples=200, seed=3)
    assert lower <= mean([1.0, 0.0, 1.0, 1.0]) <= upper
    assert 0.0 <= lower <= upper <= 1.0


def test_paired_difference_requires_matching_lengths() -> None:
    with pytest.raises(ValueError):
        paired_difference_ci([1.0], [1.0, 0.0])


def test_paired_difference_ci_detects_consistent_improvement() -> None:
    baseline = [0.0, 0.0, 0.0, 0.0, 0.0]
    treatment = [1.0, 1.0, 1.0, 1.0, 1.0]
    observed, lower, upper = paired_difference_ci(baseline, treatment, samples=200)
    assert observed == pytest.approx(1.0)
    assert lower > 0.0


def test_dataset_splits_and_hash_are_deterministic() -> None:
    tasks = default_tasks()
    assert {task.split for task in tasks} == {"train", "dev", "test"}
    assert split_hash(tasks) == split_hash(default_tasks())


def test_leakage_check_flags_duplicate_instructions() -> None:
    from recap.evals.datasets import BenchmarkTask

    duplicate = BenchmarkTask(
        task_id="dup",
        task_family="f",
        task_version="1",
        instruction="Collect every record from the paginated API.",
        environment_version="1",
        split="test",
        failure_mode="invalid_cursor",
    )
    issues = check_leakage(default_tasks() + [duplicate])
    assert issues
