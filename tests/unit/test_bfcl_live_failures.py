"""CPU-only tests for the live BFCL failure experiment helpers.

The experiment itself needs a local model, Postgres, and Elasticsearch, so it is
not run here. These tests pin the pure pieces that decide what the recorded
numbers mean: the dev/held-out split, typed action application, failure
classification, subset summaries, paired bootstrap, and the Markdown renderer.
"""

from __future__ import annotations

import numpy as np
from scripts import bfcl_live_failures as live

SCHEMA = {
    "name": "calculate_things",
    "description": "Calculate things.",
    "parameters": {
        "type": "dict",
        "properties": {
            "n": {"type": "integer"},
            "unit": {"type": "string", "enum": ["c", "f"]},
            "values": {"type": "array", "items": {"type": "float"}},
            "options": {
                "type": "dict",
                "properties": {"label": {"type": "string"}},
                "required": ["label"],
            },
        },
        "required": ["n"],
    },
}


def _task(task_id: str, function: str, query: str, schema=None) -> dict:
    return {
        "id": task_id,
        "function": function,
        "query": query,
        "schema": schema or SCHEMA,
        "args": {"n": 1},
    }


def test_split_tasks_keeps_one_dev_query_and_holds_out_the_rest() -> None:
    tasks = [
        _task("a", "f_one", "first"),
        _task("b", "f_one", "second"),
        _task("c", "f_one", "third"),
        _task("d", "f_two", "only one"),
        _task("e", "f_three", "first"),
        _task("f", "f_three", "second"),
        _task("f", "f_three", "second"),
    ]
    dev, evaluation, skipped = live.split_tasks(tasks)

    assert [row["function"] for row in dev] == ["f_one", "f_three"]
    assert {row["function"] for row in evaluation} == {"f_one", "f_three"}
    assert len(evaluation) == 3
    assert any(item["function"] == "f_two" for item in skipped)
    assert all("args" not in row for row in dev + evaluation)
    # Dev and held-out queries never overlap for the same function.
    for function in {row["function"] for row in dev}:
        dev_queries = {row["query"] for row in dev if row["function"] == function}
        eval_queries = {row["query"] for row in evaluation if row["function"] == function}
        assert not dev_queries & eval_queries


def test_split_tasks_uses_one_schema_variant() -> None:
    other = {**SCHEMA, "description": "other variant"}
    tasks = [
        _task("a", "f_one", "first", SCHEMA),
        _task("b", "f_one", "second", other),
        _task("c", "f_one", "third", other),
    ]
    dev, evaluation, _ = live.split_tasks(tasks)
    assert len(dev) == 1
    assert len(evaluation) == 1
    assert dev[0]["schema"] is other
    assert evaluation[0]["schema"] is other


def test_apply_actions_coerces_types_recursively() -> None:
    args = {"n": "5", "values": ["1", "2.5"], "options": {"label": 7}}
    actions = [
        {
            "action_type": "coerce_argument_type",
            "parameters": {"arguments": {"n": "integer", "values": "array", "options": "object"}},
        }
    ]
    repaired = live.apply_actions(args, actions, SCHEMA)
    assert repaired == {"n": 5, "values": [1.0, 2.5], "options": {"label": "7"}}
    assert args["n"] == "5"


def test_apply_actions_is_a_noop_on_already_valid_types() -> None:
    args = {"n": 5, "values": [1, 2.5], "options": {"label": "x"}}
    actions = [
        {
            "action_type": "coerce_argument_type",
            "parameters": {"arguments": {"n": "integer", "values": "array"}},
        }
    ]
    assert live.apply_actions(args, actions, SCHEMA) == args


def test_apply_actions_keeps_rename_and_supply_semantics() -> None:
    actions = [
        {"action_type": "map_parameter", "parameters": {"from": "n", "to": "count"}},
        {"action_type": "supply_required_parameter", "parameters": {"unit": "c"}},
    ]
    assert live.apply_actions({"n": 5}, actions, SCHEMA) == {"count": 5, "unit": "c"}


def test_argument_diagnostics_lists_names_types_and_enums() -> None:
    diagnostics = live.argument_diagnostics(
        SCHEMA, {"n": "5", "unit": "kelvin", "extra": 1, "values": [1, "2"]}
    )
    codes = {(item["argument"], item["code"]) for item in diagnostics}
    assert ("n", "INVALID_ARGUMENT_TYPE") in codes
    assert ("unit", "INVALID_ENUM_VALUE") in codes
    assert ("extra", "UNKNOWN_PARAMETER") in codes
    assert ("values[1]", "INVALID_ARGUMENT_TYPE") not in codes
    assert any(
        item["code"] == "INVALID_ARGUMENT_TYPE" and "values" in item["message"]
        for item in diagnostics
    )


def test_error_category_maps_the_bfcl_codes() -> None:
    assert live.error_category("INVALID_ARGUMENT_TYPE") == "argument_type_mismatch"
    assert live.error_category("INVALID_ENUM_VALUE") == "enum_value_mismatch"
    assert live.error_category("UNKNOWN_PARAMETER") == "tool_parameter_mismatch"
    assert live.error_category("MISSING_REQUIRED_PARAMETER") == "missing_required_parameter"
    assert live.error_category("SOMETHING_ELSE") == "unclassified_failure"


def test_classify_execution_separates_types_from_names() -> None:
    from recap.environments.bfcl import BFCLEnvironment

    environment = BFCLEnvironment(name="calculate_things", schema=SCHEMA, strict=True)
    result = live.classify_execution(environment, {"n": "5", "unit": "c"})
    assert result["execution_ok"] is False
    assert result["error_code"] == "INVALID_ARGUMENT_TYPE"
    assert result["failure_category"] == "argument_type_mismatch"

    unknown = live.classify_execution(environment, {"n": 5, "extra": 1})
    assert unknown["error_code"] == "UNKNOWN_PARAMETER"
    assert unknown["failure_category"] == "tool_parameter_mismatch"

    assert live.classify_execution(environment, {"n": 5, "unit": "c"})["execution_ok"]


def test_cache_key_changes_with_every_decoding_input() -> None:
    base = dict(
        model="m",
        base_url="http://localhost/v1",
        task_id="t",
        seed=1,
        temperature=0.0,
        kind="first",
    )
    assert live.cache_key(**base) == live.cache_key(**base)
    assert live.cache_key(**{**base, "seed": 2}) != live.cache_key(**base)
    assert live.cache_key(**{**base, "temperature": 0.7}) != live.cache_key(**base)
    assert live.cache_key(**{**base, "kind": "retry"}) != live.cache_key(**base)
    assert live.cache_key(**{**base, "feedback": "x"}) != live.cache_key(**base)


def test_is_correct_selection_requires_exactly_one_named_call() -> None:
    assert live.is_correct_selection("f", 1, "f") is True
    assert live.is_correct_selection(None, 0, "f") is False
    assert live.is_correct_selection("f", 2, "f") is False
    assert live.is_correct_selection("other", 1, "f") is False


def _row(
    *,
    arm: str = "no_memory",
    condition: str = "held_out",
    task_id: str = "t1",
    seed: int = 1,
    cluster: str = "f",
    selection_ok: bool = True,
    initial_valid: bool = False,
    call_valid: bool = True,
    has_safe_memory: bool = True,
    selected: bool = False,
    abstained: bool = True,
    selected_is_current: bool = False,
    broken: bool = False,
    retried: bool = False,
) -> dict:
    return {
        "condition": condition,
        "task_id": task_id,
        "seed": seed,
        "cluster": cluster,
        "arm": arm,
        "selection_ok": selection_ok,
        "initial_valid": initial_valid,
        "retry_valid": False,
        "retried": retried,
        "call_valid": call_valid,
        "repaired": not initial_valid and call_valid,
        "broken_valid_call": broken,
        "selected": selected,
        "abstained": abstained,
        "selected_memory_id": "m" if selected else None,
        "selected_tool": "f",
        "selected_is_current": selected_is_current,
        "has_safe_memory": has_safe_memory,
        "false_apply": bool(selected and not has_safe_memory),
        "harmful_apply": bool(selected and not has_safe_memory and not call_valid),
        "retrieval_hit_at_1": selected_is_current,
        "candidate_count": 1,
        "eligible_count": 1,
        "applicable_count": 1,
        "initial_error_code": "",
        "final_error_code": "",
        "failure_category": "argument_type_mismatch",
        "selected_action_types": [],
        "eval_temperature": 0.7,
    }


def test_summarize_arm_reports_repair_and_harm_separately() -> None:
    rows = [
        _row(task_id="t1", selection_ok=True, initial_valid=False, call_valid=True),
        _row(task_id="t2", selection_ok=True, initial_valid=False, call_valid=False),
        _row(task_id="t3", selection_ok=True, initial_valid=True, call_valid=False, broken=True),
        _row(task_id="t4", selection_ok=False, initial_valid=False, call_valid=False),
    ]
    stats = live.summarize_arm(rows)
    assert stats["n"] == 4
    assert stats["selection_ok_rate"] == 0.75
    assert stats["initial_valid_rate"] == 0.25
    assert stats["call_valid_rate"] == 0.25
    assert stats["execution_repair_rate"] == 0.5
    assert stats["broken_valid_rate"] == 1.0


def test_subset_rows_use_the_documented_definitions() -> None:
    rows = [
        _row(
            task_id="t1",
            condition="in_pool",
            selection_ok=True,
            initial_valid=False,
            has_safe_memory=True,
        ),
        _row(task_id="t2", selection_ok=True, initial_valid=False, has_safe_memory=False),
        _row(task_id="t3", selection_ok=True, initial_valid=True, has_safe_memory=True),
        _row(task_id="t4", selection_ok=False, initial_valid=False, has_safe_memory=True),
    ]
    assert len(live.subset_rows(rows, "all")) == 4
    assert len(live.subset_rows(rows, "selected_ok")) == 3
    assert len(live.subset_rows(rows, "execution_failed")) == 2
    assert len(live.subset_rows(rows, "safe_failed")) == 1
    assert len(live.subset_rows(rows, "no_safe")) == 1
    assert len(live.subset_rows(rows, "no_safe_failed")) == 1
    assert len(live.subset_rows(rows, "initially_valid")) == 1


def test_summarize_arm_counts_harmful_applies() -> None:
    rows = [
        _row(
            arm="rrf_typed",
            task_id="t1",
            initial_valid=False,
            call_valid=False,
            has_safe_memory=False,
            selected=True,
            abstained=False,
        ),
        _row(
            arm="rrf_typed",
            task_id="t2",
            initial_valid=False,
            call_valid=True,
            has_safe_memory=False,
            selected=True,
            abstained=False,
        ),
        _row(arm="rrf_typed", task_id="t3", initial_valid=False, call_valid=False),
    ]
    stats = live.summarize_arm(rows)
    assert stats["false_apply_rate"] == 2 / 3
    assert stats["harmful_apply_rate"] == 1 / 3


def test_paired_harm_bootstrap_is_negative_when_recap_harms_less() -> None:
    rng = np.random.default_rng(7)

    def block(arm: str, harmful: bool) -> list[dict]:
        rows = []
        for index in range(6):
            rows.append(
                _row(
                    arm=arm,
                    task_id=f"t{index}",
                    cluster=f"f{index}",
                    initial_valid=False,
                    call_valid=not harmful,
                    has_safe_memory=False,
                    selected=harmful,
                    abstained=not harmful,
                )
            )
        return rows

    stats = live.paired_bootstrap(
        block("full_recap", False),
        block("rrf_typed", True),
        cluster_key="cluster",
        value_key="harmful_apply",
        rng=rng,
        resamples=100,
    )
    assert stats["observed"] == -1.0
    assert stats["excludes_zero"] is True


def test_paired_bootstrap_detects_a_consistent_gain() -> None:
    rng = np.random.default_rng(5)

    def block(arm: str, value: bool) -> list[dict]:
        return [
            _row(arm=arm, task_id=f"t{index}", cluster=f"f{index}", call_valid=value)
            for index in range(8)
        ]

    stats = live.paired_bootstrap(
        block("full_recap", True),
        block("no_memory", False),
        cluster_key="cluster",
        value_key="call_valid",
        rng=rng,
        resamples=100,
    )
    assert stats["observed"] == 1.0
    assert stats["cluster_ci95"] == [1.0, 1.0]
    assert stats["excludes_zero"] is True


def _stats(n: int, call_valid: float, repair: float = 0.0) -> dict:
    return {
        "n": n,
        "selection_ok_rate": 1.0,
        "initial_valid_rate": 0.0,
        "call_valid_rate": call_valid,
        "execution_repair_rate": repair,
        "repair_rate_all_invalid": repair,
        "broken_valid_rate": 0.0,
        "selected_rate": 0.5,
        "abstain_rate": 0.5,
        "correct_memory_rate": 0.5,
        "false_apply_rate": 0.0,
        "harmful_apply_rate": 0.0,
        "hit_at_1": 0.5,
        "retried_rate": 0.0,
        "funnel": {
            "rows": n,
            "candidates": 1,
            "eligible": 1,
            "applicable": 1,
            "selected": 1,
            "abstained": 1,
        },
    }


def test_condition_paired_bootstrap_uses_the_same_task_and_seed() -> None:
    rng = np.random.default_rng(9)
    rows = []
    for index in range(4):
        for condition, valid in (("in_pool", True), ("held_out", False)):
            rows.append(
                _row(
                    arm="full_recap",
                    condition=condition,
                    task_id=f"t{index}",
                    seed=11,
                    cluster=f"f{index}",
                    call_valid=valid,
                )
            )
    stats = live.condition_paired_bootstrap(
        rows,
        treatment="in_pool",
        reference="held_out",
        cluster_key="cluster",
        value_key="call_valid",
        rng=rng,
        resamples=100,
    )
    assert stats["full_recap"]["observed"] == 1.0
    assert stats["full_recap"]["excludes_zero"] is True


def _payload() -> dict:
    subset = {
        "no_memory": _stats(8, 0.125),
        "retry_only": _stats(8, 0.25),
        "full_recap": _stats(8, 0.75),
    }
    return {
        "generated_at": "2026-01-01T00:00:00+00:00",
        "dataset": "fixture",
        "index_prefix": "fixture-prefix",
        "model": {"model": "fixture-model", "base_url": "http://localhost/v1"},
        "prompt": {"version": "v1"},
        "decoding": {
            "discovery_temperature": 0.0,
            "discovery_seed": None,
            "eval_temperature": 0.7,
            "seeds": [1, 2, 3],
            "read_timeout_seconds": 1,
            "overall_timeout_seconds": 2,
        },
        "embedding": {"provider": "hashing", "model": "fixture", "dimension": 256},
        "policy_version": "v1",
        "utility_threshold": 0.35,
        "bootstrap_resamples": 100,
        "funnel": {
            "dev_calls": 2,
            "memories_promoted": 2,
            "eval_rows": 8,
            "promotion_failures": [],
        },
        "discovery": {
            "summary": {
                "selection_failures": 0,
                "execution_failures": 2,
                "type_failures": 2,
                "categories": {"argument_type_mismatch": 2},
            },
            "promotion_failures": [],
        },
        "eval": {
            "subsets": {
                "all": subset,
                "execution_failed": subset,
                "safe_failed": subset,
                "no_safe": subset,
                "no_safe_failed": subset,
                "initially_valid": subset,
            },
            "per_row": [],
        },
        "paired": {
            "safe_failed": {
                "retry_only": {
                    "observed": 0.5,
                    "cluster_ci95": [0.1, 0.9],
                    "excludes_zero": True,
                }
            }
        },
        "paired_harm": {
            "rrf_typed": {
                "observed": -0.25,
                "cluster_ci95": [-0.4, -0.1],
                "excludes_zero": True,
            }
        },
        "limits": ["fixture only"],
    }


def test_markdown_renders_numbers_from_the_payload() -> None:
    text = live.render_markdown(_payload())
    assert "fixture-model" in text
    assert "0.750" in text
    assert "0.125" in text
    assert "0.500" in text
    assert "[+0.100, +0.900]" in text
    assert "fixture only" in text


def test_render_only_regenerates_markdown_from_json(tmp_path) -> None:
    import json

    json_path = tmp_path / "report.json"
    json_path.write_text(json.dumps(_payload()), encoding="utf-8")
    assert live.main(["--render-only", str(json_path)]) == 0
    markdown = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "fixture-model" in markdown
    assert "0.750" in markdown
