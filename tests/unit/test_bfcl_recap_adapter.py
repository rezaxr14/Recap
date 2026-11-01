"""CPU-only tests for the BFCL environment, drift scenarios, and repair adapter."""

from __future__ import annotations

from typing import Any

from recap.domain.enums import EventType
from recap.domain.repairs import FailureWindow
from recap.domain.runs import RunEvent
from recap.environments.bfcl import BFCLEnvironment
from recap.evals.bfcl import (
    build_transfer_scenarios,
    drift_schema,
    sibling_pairs,
)
from recap.repair_compiler.bfcl import BFCLSandboxValidator, SchemaRenameGenerator
from recap.repair_compiler.candidate_generation import build_candidate

SCHEMA: dict[str, Any] = {
    "name": "get_weather",
    "description": "Look up the weather for a city.",
    "parameters": {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name"},
            "unit": {"type": "string", "description": "Unit", "enum": ["c", "f"]},
        },
        "required": ["city", "unit"],
    },
}


def _events(args: dict[str, Any], code: str = "UNKNOWN_PARAMETER") -> list[RunEvent]:
    return [
        RunEvent(
            run_id="run_test",
            sequence_number=1,
            event_type=EventType.TOOL_CALL_REQUESTED,
            tool_name="get_weather",
            payload_redacted={"request": args},
        ),
        RunEvent(
            run_id="run_test",
            sequence_number=2,
            event_type=EventType.TOOL_CALL_FAILED,
            tool_name="get_weather",
            payload_redacted={
                "request": args,
                "error_code": code,
                "error_message": "unknown parameter 'city'; required 'city_v2'",
            },
        ),
    ]


def _window() -> FailureWindow:
    return FailureWindow(
        run_id="run_test",
        start_sequence=1,
        end_sequence=2,
        start_event_id="evt_1",
        end_event_id="evt_2",
        failure_category="tool_parameter_mismatch",
        normalized_signature="sig",
        raw_signature="unknown parameter 'city'; required 'city_v2'",
        error_code="UNKNOWN_PARAMETER",
        error_type="InvalidRequest",
        tool_name="get_weather",
        event_ids=["evt_1", "evt_2"],
        retry_count=0,
        incomplete=False,
        notes=[],
    )


def _draft_and_candidate():
    drifted, old, new = drift_schema(SCHEMA)
    env = BFCLEnvironment(
        name="get_weather",
        schema=drifted,
        version="2",
        failing_args={"city": "Paris", "unit": "c"},
    )
    generator = SchemaRenameGenerator({env.environment_id: drifted})
    info = {
        "environment_id": env.environment_id,
        "environment_version": env.version,
        "tool_schema_hashes": {"get_weather": env.schema_hash},
    }
    drafts = generator.generate(
        window=_window(), environment=info, events=_events({"city": "Paris", "unit": "c"})
    )
    candidate = build_candidate(
        drafts[0],
        window=_window(),
        workspace_id="ws",
        environment_id=env.environment_id,
        environment_fingerprint=env.fingerprint().fingerprint,
        generated_by=generator.name,
        task_family="weather",
    )
    return env, drafts, candidate, old, new


def test_environment_reports_unknown_and_missing_parameters() -> None:
    env = BFCLEnvironment(name="get_weather", schema=SCHEMA)
    unknown = env.call({"city": "Paris", "unit": "c", "extra": 1})
    assert not unknown.ok
    assert unknown.error is not None
    assert unknown.error["error_code"] == "UNKNOWN_PARAMETER"
    assert "unknown parameter 'extra'" in unknown.error["error_message"]

    missing = env.call({"city": "Paris"})
    assert not missing.ok
    assert missing.error is not None
    assert missing.error["error_code"] == "MISSING_REQUIRED_PARAMETER"

    assert env.call({"city": "Paris", "unit": "c"}).ok


def test_fingerprint_is_stable_and_versioned() -> None:
    first = BFCLEnvironment(name="get_weather", schema=SCHEMA, version="1")
    second = BFCLEnvironment(name="get_weather", schema=SCHEMA, version="1")
    third = BFCLEnvironment(name="get_weather", schema=SCHEMA, version="2")
    assert first.fingerprint().fingerprint == second.fingerprint().fingerprint
    assert first.fingerprint().fingerprint != third.fingerprint().fingerprint
    assert "parameters.properties.city" in first.schema_path_hashes()


def test_drift_rename_and_pair_filter() -> None:
    drifted, old, new = drift_schema(SCHEMA)
    assert (old, new) == ("city", "city_v2")
    assert set(drifted["parameters"]["properties"]) == {"city_v2", "unit"}
    assert set(SCHEMA["parameters"]["properties"]) == {"city", "unit"}

    tasks = [
        {
            "id": "t1",
            "function": "get_weather",
            "query": "Weather in Paris?",
            "args": {"city": "Paris", "unit": "c"},
            "schema": SCHEMA,
        },
        {
            "id": "t2",
            "function": "get_weather",
            "query": "Weather in Berlin?",
            "args": {"city": "Berlin", "unit": "c"},
            "schema": SCHEMA,
        },
        {
            "id": "t3",
            "function": "get_weather",
            "query": "Weather in Rome?",
            "args": {"city": "Rome", "unit": "c"},
            "schema": SCHEMA,
        },
    ]
    scenarios = build_transfer_scenarios(tasks)
    assert len(scenarios) == 1
    scenario = scenarios[0]
    assert scenario["old"] == "city"
    assert scenario["new"] == "city_v2"
    assert "Berlin" in scenario["query2"] or "Paris" in scenario["query2"]


def test_sibling_pairs_require_shared_family_and_different_parameters() -> None:
    def scenario(name: str, params: set[str]) -> dict[str, Any]:
        return {
            "function": name,
            "schema": {
                "name": name,
                "parameters": {
                    "properties": {param: {} for param in params},
                    "required": sorted(params),
                },
            },
        }

    scenarios = [
        scenario("item_by_amazon_ASIN", {"asin"}),
        scenario("item_by_walmart_id", {"walmart_id"}),
        scenario("item_by_ebay_id", {"ebay_id"}),
        scenario("math.add", {"a", "b"}),
        scenario("math.subtract", {"a", "b"}),
    ]
    pairs = sibling_pairs(scenarios)
    names = {(owner["function"], query["function"]) for owner, query in pairs}
    assert ("item_by_amazon_ASIN", "item_by_walmart_id") in names
    assert ("math.add", "math.subtract") not in names
    assert all(owner != query for owner, query in names)


def test_schema_generator_requires_a_single_unknown_and_single_missing() -> None:
    drifted, _, _ = drift_schema(SCHEMA)
    env = BFCLEnvironment(
        name="get_weather",
        schema=drifted,
        failing_args={"city": "Paris", "unit": "c"},
    )
    info = {
        "environment_id": env.environment_id,
        "tool_schema_hashes": {"get_weather": env.schema_hash},
    }
    generator = SchemaRenameGenerator({env.environment_id: drifted})
    assert (
        generator.generate(
            window=_window(), environment=info, events=_events({"city": "Paris", "unit": "c"})
        )
        != []
    )
    two_unknown = generator.generate(
        window=_window(),
        environment=info,
        events=_events({"city": "Paris", "unit": "c", "x": 1, "y": 2}),
    )
    assert two_unknown == []
    assert generator.generate(window=_window(), environment={}, events=[]) == []


def test_generated_candidate_validates_against_the_sandbox() -> None:
    env, drafts, candidate, old, new = _draft_and_candidate()
    assert drafts[0].actions[0].parameters == {"from": old, "to": new}
    assert candidate.preconditions
    assert candidate.postconditions

    validator = BFCLSandboxValidator({env.environment_id: env})
    result = validator.validate(candidate, environment=env.fingerprint())
    assert result.status.value == "passed"
    assert any(check["check"] == "failure_reproduced" for check in result.checks)


def test_validator_fails_when_the_repair_is_wrong() -> None:
    env, _, candidate, _, _ = _draft_and_candidate()
    candidate.strategy.actions[0].parameters = {"from": "unit", "to": "unit_v2"}
    validator = BFCLSandboxValidator({env.environment_id: env})
    result = validator.validate(candidate, environment=env.fingerprint())
    assert result.status.value == "failed"
