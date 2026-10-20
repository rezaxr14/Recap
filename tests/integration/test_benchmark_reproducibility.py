"""Benchmark runs must be reproducible: stale state was the original drift bug."""

from __future__ import annotations

import pytest

from recap.evals.runner import run_benchmark

pytestmark = pytest.mark.integration


def test_two_benchmark_runs_on_one_container_have_identical_summaries(container, tmp_path) -> None:
    first = run_benchmark(container, split="dev", output_dir=tmp_path, cleanup=True)
    second = run_benchmark(container, split="dev", output_dir=tmp_path, cleanup=True)

    # The workspace is per-run, so it must not be reused.
    assert first["workspace_id"] != second["workspace_id"]
    assert first["summary"] == second["summary"]

    # Every chunk mode must actually retrieve something; the original bug
    # showed up as a chunk baseline whose MRR decayed on each rerun.
    for baseline in (
        "bm25_naive_chunk",
        "vector_naive_chunk",
        "hybrid_rrf_naive_chunk",
        "outcome_aware_naive_chunk",
        "bm25_chunk",
        "vector_chunk",
        "hybrid_rrf_chunk",
        "outcome_aware_chunk",
    ):
        stats = first["summary"][baseline]
        assert stats["recall_at_3"] > 0, baseline
        assert stats["mrr"] > 0, baseline


def test_benchmark_workspace_is_torn_down(container, tmp_path) -> None:
    result = run_benchmark(container, split="dev", output_dir=tmp_path, cleanup=True)
    workspace = result["workspace_id"]

    with container.services() as services:
        assert services.memories.list_memories(workspace) == []
        assert services.chunk_store.list_for_workspace(workspace) == []
    assert container.chunk_search_repository.count(workspace_id=workspace) == 0
