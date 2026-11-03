"""CPU-only tests for the BFCL environment, drift scenarios, and repair adapter."""

from __future__ import annotations

from typing import Any

from recap.domain.enums import EventType
from recap.domain.repairs import FailureWindow
from recap.domain.runs import RunEvent
from recap.environments.bfcl import (
    BFCLEnvironment,
    argument_errors,
    openai_parameters,
    openai_tool_schema,
)
from recap.evals.bfcl import (
    build_transfer_scenarios,
    drift_schema,
    sibling_pairs,
)
from recap.repair_compiler.bfcl import (
    BFCLSandboxValidator,
    BFCLTypeCoercionGenerator,
    SchemaRenameGenerator,
    coerce_argument,
)
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


STRICT_SCHEMA: dict[str, Any] = {
    "name": "calculate_things",
    "description": "Calculate things.",
    "parameters": {
        "type": "dict",
        "properties": {
            "n": {"type": "integer", "description": "Count"},
            "unit": {"type": "string", "description": "Unit", "enum": ["c", "f"]},
            "values": {
                "type": "array",
                "description": "Values",
                "items": {"type": "float"},
            },
            "options": {
                "type": "dict",
                "properties": {"label": {"type": "string"}},
            },
        },
        "required": ["n"],
    },
}


def test_strict_environment_rejects_declared_type_and_enum_errors() -> None:
    strict = BFCLEnvironment(name="calculate_things", schema=STRICT_SCHEMA, strict=True)

    wrong_type = strict.call({"n": "5"})
    assert not wrong_type.ok
    assert wrong_type.error is not None
    assert wrong_type.error["error_code"] == "INVALID_ARGUMENT_TYPE"
    assert "must be integer" in wrong_type.error["error_message"]

    wrong_enum = strict.call({"n": 5, "unit": "kelvin"})
    assert not wrong_enum.ok
    assert wrong_enum.error is not None
    assert wrong_enum.error["error_code"] == "INVALID_ENUM_VALUE"

    wrong_item = strict.call({"n": 5, "values": [1, "2.5"]})
    assert not wrong_item.ok
    assert wrong_item.error is not None
    assert wrong_item.error["error_code"] == "INVALID_ARGUMENT_TYPE"
    assert "values[1]" in wrong_item.error["error_message"]

    assert strict.call({"n": 5, "unit": "c", "values": [1, 2.5]}).ok


def test_non_strict_environment_keeps_injected_rename_semantics() -> None:
    environment = BFCLEnvironment(name="calculate_things", schema=STRICT_SCHEMA)
    assert environment.call({"n": "5"}).ok
    errors = argument_errors({"type": "integer"}, "5", "n")
    assert errors[0][0] == "INVALID_ARGUMENT_TYPE"


def _type_failure_env():
    environment = BFCLEnvironment(
        name="calculate_things",
        schema=STRICT_SCHEMA,
        strict=True,
        failing_args={"n": "5", "values": ["1", "2"]},
    )
    window = FailureWindow(
        run_id="run_type",
        start_sequence=1,
        end_sequence=2,
        start_event_id="evt_1",
        end_event_id="evt_2",
        failure_category="argument_type_mismatch",
        normalized_signature="sig",
        raw_signature="argument 'n' must be integer, got str",
        error_code="INVALID_ARGUMENT_TYPE",
        error_type="InvalidRequest",
        tool_name="calculate_things",
        event_ids=["evt_1", "evt_2"],
        retry_count=0,
        incomplete=False,
        notes=[],
    )
    events = [
        RunEvent(
            run_id="run_type",
            sequence_number=1,
            event_type=EventType.TOOL_CALL_REQUESTED,
            tool_name="calculate_things",
            payload_redacted={"request": {"n": "5", "values": ["1", "2"]}},
        ),
        RunEvent(
            run_id="run_type",
            sequence_number=2,
            event_type=EventType.TOOL_CALL_FAILED,
            tool_name="calculate_things",
            payload_redacted={
                "request": {"n": "5", "values": ["1", "2"]},
                "error_code": "INVALID_ARGUMENT_TYPE",
                "error_message": "argument 'n' must be integer, got str",
            },
        ),
    ]
    info = {
        "environment_id": environment.environment_id,
        "environment_version": environment.version,
        "tool_schema_hashes": {"calculate_things": environment.schema_hash},
    }
    return environment, window, events, info


def test_type_coercion_generator_repairs_a_real_type_failure() -> None:
    environment, window, events, info = _type_failure_env()
    generator = BFCLTypeCoercionGenerator({environment.environment_id: STRICT_SCHEMA})
    drafts = generator.generate(window=window, environment=info, events=events)

    assert len(drafts) == 1
    action = drafts[0].actions[0]
    assert action.parameters == {"arguments": {"n": "integer", "values": "array"}}

    candidate = build_candidate(
        drafts[0],
        window=window,
        workspace_id="ws",
        environment_id=environment.environment_id,
        environment_fingerprint=environment.fingerprint().fingerprint,
        generated_by=generator.name,
    )
    validator = BFCLSandboxValidator({environment.environment_id: environment})
    result = validator.validate(candidate, environment=environment.fingerprint())
    assert result.status.value == "passed"
    assert any(check["check"] == "postcondition_tool_call_succeeded" for check in result.checks)


def test_type_coercion_generator_ignores_other_failures() -> None:
    environment, window, events, info = _type_failure_env()
    window.failure_category = "tool_parameter_mismatch"
    generator = BFCLTypeCoercionGenerator({environment.environment_id: STRICT_SCHEMA})
    assert generator.generate(window=window, environment=info, events=events) == []


def test_coerce_argument_handles_json_strings_and_refuses_garbage() -> None:
    assert coerce_argument("5", "integer") == 5
    assert coerce_argument(" 2.5 ", "number") == 2.5
    assert coerce_argument("true", "boolean") is True
    assert coerce_argument("[1, 2]", "array") == [1, 2]
    assert coerce_argument('{"a": 1}', "object") == {"a": 1}
    assert coerce_argument("not-a-number", "integer") == "not-a-number"
    assert coerce_argument("[not json", "array") == "[not json"


def test_openai_parameters_convert_bfcl_types_recursively() -> None:
    bfcl = {
        "type": "dict",
        "properties": {
            "coord": {"type": "tuple", "items": {"type": "float"}},
            "interval": {"type": "array", "items": {"type": "integer"}},
            "route": {"type": "string", "enum": ["fast", "short"], "default": "fast"},
            "nested": {
                "type": "dict",
                "properties": {"depth": {"type": "integer"}},
                "required": ["depth"],
            },
            "data": {"type": "any", "description": "training data"},
        },
        "required": ["coord", "interval", "nested"],
    }
    converted = openai_parameters(bfcl)

    assert converted["type"] == "object"
    assert converted["additionalProperties"] is False
    assert converted["required"] == ["coord", "interval", "nested"]
    assert converted["properties"]["coord"] == {
        "type": "array",
        "items": {"type": "number"},
    }
    assert converted["properties"]["interval"]["items"] == {"type": "integer"}
    assert converted["properties"]["route"]["enum"] == ["fast", "short"]
    assert converted["properties"]["route"]["default"] == "fast"
    assert converted["properties"]["nested"]["type"] == "object"
    assert converted["properties"]["nested"]["additionalProperties"] is False
    assert converted["properties"]["data"]["description"] == "training data"
    assert converted["properties"]["data"]["type"] == [
        "string",
        "number",
        "integer",
        "boolean",
        "array",
        "object",
        "null",
    ]


def test_openai_conversion_keeps_parameter_names_and_fingerprints() -> None:
    bfcl = {
        "name": "calculate_distance",
        "description": "Distance between coordinates.",
        "parameters": {
            "type": "dict",
            "properties": {
                "coord1": {"type": "tuple", "items": {"type": "float"}},
                "unit": {"type": "string"},
            },
            "required": ["coord1", "unit"],
        },
    }
    environment = BFCLEnvironment(name="calculate_distance", schema=bfcl)
    document = openai_tool_schema(bfcl)

    assert document["name"] == "calculate_distance"
    assert document["parameters"]["type"] == "object"
    assert set(document["parameters"]["properties"]) == set(bfcl["parameters"]["properties"])
    # The fingerprint still hashes the dataset's declared schema.
    assert environment.tool_document["parameters"]["type"] == "dict"
    assert (
        environment.schema_hash
        == BFCLEnvironment(name="calculate_distance", schema=bfcl).schema_hash
    )


def test_openai_parameters_fall_back_to_an_object() -> None:
    assert openai_parameters(None) == {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }
    assert openai_parameters({"type": "string"})["type"] == "object"


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
