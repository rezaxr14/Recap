"""CPU-only tests for the RECAP-stack BFCL experiment helpers.

The full experiment needs Postgres, Elasticsearch, and promoted memories, so it
is not run here. These tests pin the pure pieces that decide what the recorded
numbers mean: action application, outcome classification, cluster bootstrap, and
the Markdown renderer.
"""

from __future__ import annotations

import numpy as np
from scripts import bfcl_recap_experiment as experiment

SCHEMA = {
    "name": "get_weather",
    "parameters": {
        "properties": {"city": {"type": "string"}, "unit": {"type": "string"}},
        "required": ["city", "unit"],
    },
}


def _selection(
    memory_id: str = "mem_1",
    tool: str = "get_weather",
    actions: list[dict] | None = None,
) -> experiment.Selection:
    return experiment.Selection(
        memory_id=memory_id,
        tool=tool,
        version=1,
        actions=actions or [],
    )


def test_apply_actions_maps_parameter_without_mutating_input() -> None:
    args = {"city": "Paris", "unit": "c"}
    actions = [{"action_type": "map_parameter", "parameters": {"from": "city", "to": "city_v2"}}]
    repaired = experiment.apply_actions(args, actions)
    assert repaired == {"city_v2": "Paris", "unit": "c"}
    assert args == {"city": "Paris", "unit": "c"}


def test_apply_actions_supplies_required_parameters_only_when_missing() -> None:
    actions = [{"action_type": "supply_required_parameter", "parameters": {"unit": "f"}}]
    assert experiment.apply_actions({"city": "Paris"}, actions) == {"city": "Paris", "unit": "f"}
    assert experiment.apply_actions({"city": "Paris", "unit": "c"}, actions) == {
        "city": "Paris",
        "unit": "c",
    }
    assert experiment.apply_actions({"city": "Paris"}, [{"action_type": "halt"}]) == {
        "city": "Paris"
    }


def test_safe_row_marks_current_memory_and_valid_call() -> None:
    result = experiment.SelectionResult(
        _selection(), candidate_count=3, eligible_count=2, applicable_count=1
    )
    row = experiment.evaluate_row(
        condition="in_pool",
        scenario="get_weather",
        arm="full_recap",
        has_safe_memory=True,
        schema=SCHEMA,
        args={"city": "Paris", "unit": "c"},
        target_function="get_weather",
        result=result,
        current_memory_id="mem_1",
    )
    assert row["call_valid"] is True
    assert row["selected_is_current"] is True
    assert row["false_apply"] is False
    assert row["harmful_transfer"] is False
    assert row["retrieval_hit_at_1"] is True
    assert row["candidate_count"] == 3
    assert row["eligible_count"] == 2
    assert row["applicable_count"] == 1


def test_stale_same_tool_selection_is_false_apply_not_harmful_transfer() -> None:
    actions = [{"action_type": "map_parameter", "parameters": {"from": "city", "to": "city_v2"}}]
    result = experiment.SelectionResult(_selection(actions=actions))
    row = experiment.evaluate_row(
        condition="drift_phase2",
        scenario="get_weather",
        arm="function_keyed",
        has_safe_memory=False,
        schema=SCHEMA,
        args={"city": "Paris", "unit": "c"},
        target_function="get_weather",
        result=result,
        stale_memory_id="mem_1",
    )
    assert row["call_valid"] is False
    assert row["false_apply"] is True
    assert row["stale_selected"] is True
    assert row["selected_is_current"] is False
    # The memory belongs to the same tool, so this is a stale application, not
    # cross-tool transfer.
    assert row["harmful_transfer"] is False


def test_cross_tool_selection_with_invalid_call_is_harmful_transfer() -> None:
    actions = [{"action_type": "map_parameter", "parameters": {"from": "city", "to": "city_v2"}}]
    result = experiment.SelectionResult(
        _selection(memory_id="mem_other", tool="get_zip", actions=actions)
    )
    row = experiment.evaluate_row(
        condition="sibling",
        scenario="get_zip->get_weather",
        arm="bm25",
        has_safe_memory=False,
        schema=SCHEMA,
        args={"city": "Paris", "unit": "c"},
        target_function="get_weather",
        result=result,
    )
    assert row["call_valid"] is False
    assert row["false_apply"] is True
    assert row["harmful_transfer"] is True
    assert row["valid_transfer"] is False
    assert row["cluster"] == "get_weather"


def test_cross_tool_selection_that_repairs_the_call_is_valid_transfer() -> None:
    actions = [{"action_type": "map_parameter", "parameters": {"from": "city", "to": "city"}}]
    result = experiment.SelectionResult(
        _selection(memory_id="mem_other", tool="get_place", actions=actions)
    )
    row = experiment.evaluate_row(
        condition="sibling",
        scenario="get_place->get_weather",
        arm="bm25",
        has_safe_memory=False,
        schema=SCHEMA,
        args={"city": "Paris", "unit": "c"},
        target_function="get_weather",
        result=result,
    )
    assert row["call_valid"] is True
    assert row["false_apply"] is True
    assert row["valid_transfer"] is True
    assert row["harmful_transfer"] is False


def test_abstention_keeps_candidate_counts_for_the_funnel() -> None:
    result = experiment.SelectionResult(
        None, candidate_count=5, eligible_count=4, applicable_count=2
    )
    row = experiment.evaluate_row(
        condition="held_out",
        scenario="get_weather",
        arm="full_recap",
        has_safe_memory=False,
        schema=SCHEMA,
        args={"city": "Paris", "unit": "c"},
        target_function="get_weather",
        result=result,
    )
    assert row["abstained"] is True
    assert row["false_apply"] is False
    assert row["candidate_count"] == 5
    assert row["eligible_count"] == 4
    assert row["applicable_count"] == 2


def test_cluster_bootstrap_uses_clusters_not_rows() -> None:
    rng = np.random.default_rng(3)
    values = {f"fn_{index}": [1.0, 1.0] for index in range(10)}
    values.update({f"fn_{index}": [0.0, 0.0] for index in range(10, 20)})
    stats = experiment.cluster_bootstrap(values, rng, resamples=400)
    assert stats["n_clusters"] == 20
    assert stats["n_rows"] == 40
    assert stats["observed"] == 0.5
    assert stats["cluster_ci95"][0] > 0.0
    assert stats["cluster_ci95"][1] < 1.0


def test_cluster_bootstrap_of_constant_values_is_zero_width() -> None:
    rng = np.random.default_rng(3)
    stats = experiment.cluster_bootstrap({"fn": [1.0, 1.0]}, rng, resamples=50)
    assert stats["cluster_ci95"] == [1.0, 1.0]


def test_paired_bootstrap_detects_consistent_gain() -> None:
    rng = np.random.default_rng(5)
    treatment = [
        {
            "condition": "in_pool",
            "scenario": f"fn_{index}",
            "cluster": f"fn_{index}",
            "call_valid": True,
        }
        for index in range(8)
    ]
    reference = [
        {
            "condition": "in_pool",
            "scenario": f"fn_{index}",
            "cluster": f"fn_{index}",
            "call_valid": False,
        }
        for index in range(8)
    ]
    stats = experiment.paired_bootstrap(
        treatment,
        reference,
        cluster_key="cluster",
        value_key="call_valid",
        rng=rng,
        resamples=100,
    )
    assert stats["observed"] == 1.0
    assert stats["cluster_ci95"] == [1.0, 1.0]
    assert stats["excludes_zero"] is True


def test_paired_bootstrap_of_identical_arms_is_zero() -> None:
    rng = np.random.default_rng(5)
    rows = [
        {
            "condition": "held_out",
            "scenario": f"fn_{index}",
            "cluster": f"fn_{index}",
            "abstained": True,
        }
        for index in range(4)
    ]
    stats = experiment.paired_bootstrap(
        rows,
        [dict(row) for row in rows],
        cluster_key="cluster",
        value_key="abstained",
        rng=rng,
    )
    assert stats["observed"] == 0.0
    assert stats["excludes_zero"] is False


def test_abstention_stats_treats_unneeded_and_missed_abstentions_separately() -> None:
    rows = [
        {"condition": "in_pool", "arm": "full_recap", "abstained": True},
        {"condition": "in_pool", "arm": "full_recap", "abstained": False},
        {"condition": "held_out", "arm": "full_recap", "abstained": True},
        {"condition": "held_out", "arm": "full_recap", "abstained": False},
    ]
    stats = experiment.abstention_stats(rows)["full_recap"]
    assert stats["precision"] == 0.5
    assert stats["recall"] == 0.5
    assert stats["false_abstain_rate"] == 0.5
    assert stats["unneeded_abstentions"] == 1
    assert stats["missed_abstentions"] == 1
    assert stats["n_safe"] == 2
    assert stats["n_no_safe"] == 2


def test_abstention_stats_are_zero_when_the_arm_never_abstains() -> None:
    rows = [
        {"condition": "in_pool", "arm": "bm25", "abstained": False},
        {"condition": "held_out", "arm": "bm25", "abstained": False},
    ]
    stats = experiment.abstention_stats(rows)["bm25"]
    assert stats["precision"] == 0.0
    assert stats["recall"] == 0.0
    assert stats["false_abstain_rate"] == 0.0


def test_markdown_renders_payload_numbers_and_funnels() -> None:
    payload = {
        "generated_at": "2026-01-01T00:00:00+00:00",
        "dataset": "fixture",
        "index_prefix": "fixture-prefix",
        "embedding": {"provider": "hashing", "model": "fixture-model", "dimension": 256},
        "utility_threshold": 0.35,
        "bootstrap_resamples": 100,
        "funnel": {"scenarios": 4, "memories_promoted": 4},
        "conditions": {
            "in_pool": {
                "results": {
                    "bm25": {
                        **_stats(call_valid_rate=0.11),
                        "funnel": {
                            "candidates": 7,
                            "eligible": 6,
                            "applicable": 5,
                            "selected": 4,
                            "abstained": 0,
                        },
                    }
                }
            }
        },
        "abstention": {
            "bm25": {
                "precision": 0.25,
                "recall": 0.5,
                "false_abstain_rate": 0.1,
                "missed_abstentions": 2,
                "unneeded_abstentions": 1,
                "n_no_safe": 8,
                "n_safe": 10,
            }
        },
        "paired": {
            "safe": {
                "bm25": {"observed": -0.25, "cluster_ci95": [-0.5, 0.0], "excludes_zero": False}
            }
        },
        "drift": {
            "functions": 2,
            "superseded": 2,
            "active_after": 2,
            "failures": [],
            "examples": [],
        },
        "limits": ["fixture only"],
    }
    text = experiment.render_markdown(payload)
    assert "0.110" in text
    assert "| `in_pool` | `bm25` | 7 | 6 | 5 | 4 | 0 |" in text
    assert "-0.250" in text
    assert "fixture only" in text


def _stats(*, call_valid_rate: float) -> dict:
    return {
        "n": 4,
        "call_valid_rate": call_valid_rate,
        "selected_rate": 0.5,
        "abstain_rate": 0.0,
        "correct_memory_rate": 0.5,
        "false_apply_rate": 0.0,
        "harmful_transfer_rate": 0.0,
        "valid_transfer_rate": 0.0,
        "stale_selection_rate": 0.0,
        "retrieval_hit_at_1": 0.5,
    }
