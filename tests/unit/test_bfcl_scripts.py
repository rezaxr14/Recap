"""Pure-function checks for the BFCL evidence scripts.

The measurement itself needs Elasticsearch and (for the Qwen rows) local model
weights, so it is not run here. These tests pin the deterministic parts that
decide what the measured numbers mean: the dev/test split, gold ranking, the
paired bootstrap, the dev-calibrated abstention rule, the injected drift, call
validity, and the report renderer.
"""

from __future__ import annotations

import numpy as np
from scripts import bfcl_memory_transfer, bfcl_retrieval


def _metrics(**overrides: float) -> dict[str, float]:
    base = {
        "hit_at_1": 0.5,
        "hit_at_3": 0.75,
        "hit_at_5": 0.9,
        "reciprocal_rank": 0.6,
        "auc_vs_irrelevance": 0.8,
    }
    base.update(overrides)
    return base


def _payload() -> dict:
    test = {
        **_metrics(hit_at_1=0.123),
        "hit_at_1_ci95": [0.1, 0.2],
    }
    return {
        "dataset": "fixture",
        "generated_at": "2026-01-01T00:00:00+00:00",
        "labeled_tasks": 4,
        "dev_labeled": 2,
        "test_labeled": 2,
        "control_tasks": 2,
        "function_pool": 3,
        "embedding": {"provider": "hashing", "model": "hashing-256", "dimension": 256},
        "rrf_k": 60,
        "analyzer": "standard",
        "selected_variant": "vector",
        "selection_rule": "highest dev hit@1",
        "results": {
            "vector": {
                "test": test,
                "dev": _metrics(),
                "abstention": {
                    "threshold": 0.0,
                    "identified": False,
                    "test_labeled_coverage": 1.0,
                    "test_control_abstention": 0.5,
                },
            }
        },
        "paired_vs_vector": {
            "bm25_minus_vector": {
                "n_queries": 2,
                "observed": -0.25,
                "cluster_ci95": [-0.5, 0.0],
                "excludes_zero": False,
            }
        },
        "limits": ["fixture only"],
    }


def test_split_is_deterministic_and_exhaustive() -> None:
    ids = [f"task_{index}" for index in range(200)]
    first = [bfcl_retrieval._split_of(task_id) for task_id in ids]
    assert first == [bfcl_retrieval._split_of(task_id) for task_id in ids]
    assert set(first) == {"dev", "test"}
    assert 60 <= first.count("dev") <= 140


def test_gold_rank_reports_first_relevant_position_and_misses() -> None:
    row = {"gold": {"b", "c"}}
    assert bfcl_retrieval._gold_rank(row, ["a", "c", "b"]) == 2
    assert bfcl_retrieval._gold_rank(row, ["a", "z"]) == 0


def test_paired_hit1_matches_hand_computation() -> None:
    rng = np.random.default_rng(7)
    treatment = {"ranks": [1, 1, 3, 0]}
    reference = {"ranks": [1, 2, 1, 0]}
    stats = bfcl_retrieval._paired_hit1(treatment, reference, rng)
    assert stats["n_queries"] == 4
    # treated hit@1 = 0.5, base hit@1 = 0.5, so the paired difference is zero.
    assert stats["observed"] == 0.0
    assert stats["excludes_zero"] is False


def test_paired_hit1_of_identical_arms_is_zero_width() -> None:
    rng = np.random.default_rng(7)
    arm = {"ranks": [1, 0, 2, 4]}
    stats = bfcl_retrieval._paired_hit1(arm, dict(arm), rng)
    assert stats["observed"] == 0.0
    assert stats["cluster_ci95"] == [0.0, 0.0]
    assert stats["excludes_zero"] is False


def test_paired_success_detects_consistent_gain() -> None:
    rng = np.random.default_rng(11)
    stats = bfcl_memory_transfer._paired_success([1.0] * 6, [0.0] * 6, rng)
    assert stats["n_scenarios"] == 6
    assert stats["observed"] == 1.0
    assert stats["cluster_ci95"] == [1.0, 1.0]
    assert stats["excludes_zero"] is True


def test_abstention_uses_dev_threshold_and_flags_degenerate_zero() -> None:
    dev = {"provided_scores": [0.0, 0.0, 0.0, 0.0, 0.0]}
    test = {"provided_scores": [0.0, 0.5], "control_scores": [-0.1, 0.2]}
    result = bfcl_retrieval._abstention(dev, test)
    assert result["threshold"] == 0.0
    assert result["identified"] is False
    assert result["test_labeled_coverage"] == 1.0
    assert result["test_control_abstention"] == 0.5


def test_markdown_is_rendered_from_payload_not_literals() -> None:
    text = bfcl_retrieval.render_markdown(_payload())
    assert "0.123" in text
    assert "`standard`" in text
    assert "n/a (threshold 0)" in text
    assert "-0.250" in text
    assert "fixture only" in text


def test_drift_renames_the_first_required_parameter() -> None:
    schema = {
        "name": "get_weather",
        "description": "Weather lookup",
        "parameters": {
            "properties": {"city": {"type": "string"}, "unit": {"type": "string"}},
            "required": ["city", "unit"],
        },
    }
    drifted, old, new = bfcl_memory_transfer._drift(schema)
    assert (old, new) == ("city", "city_v2")
    properties = drifted["parameters"]["properties"]
    assert set(properties) == {"city_v2", "unit"}
    assert drifted["parameters"]["required"] == ["city_v2", "unit"]
    assert schema["parameters"]["properties"]["city"] == {"type": "string"}


def test_drift_returns_none_without_parameters() -> None:
    assert bfcl_memory_transfer._drift({"name": "ping"}) is None


def test_apply_rename_only_touches_present_keys() -> None:
    args = {"city": "Paris", "unit": "c"}
    assert bfcl_memory_transfer._apply_rename(args, "city", "city_v2") == {
        "city_v2": "Paris",
        "unit": "c",
    }
    assert bfcl_memory_transfer._apply_rename({"unit": "c"}, "city", "city_v2") == {"unit": "c"}
    assert args == {"city": "Paris", "unit": "c"}


def test_call_status_separates_unknown_from_missing() -> None:
    schema = {
        "parameters": {
            "properties": {"city": {}, "unit": {}},
            "required": ["city", "unit"],
        }
    }
    unknown, missing = bfcl_memory_transfer._call_status(schema, {"city": "Paris", "x": 1})
    assert unknown == {"x"}
    assert missing == {"unit"}
    assert bfcl_memory_transfer._valid(schema, {"city": "Paris", "unit": "c"})
    assert not bfcl_memory_transfer._valid(schema, {"city": "Paris"})


def test_parse_call_reads_keyword_arguments_and_rejects_junk() -> None:
    assert bfcl_memory_transfer._parse_call("f(a=1, b='two')") == {"a": 1, "b": "two"}
    assert bfcl_memory_transfer._parse_call("not a call") == {}
    assert bfcl_memory_transfer._parse_call("f(a=some_name)") == {}
