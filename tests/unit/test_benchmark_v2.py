from __future__ import annotations

from typing import Any

from recap.domain.enums import ValidationStatus
from recap.domain.repairs import FailureWindow
from recap.domain.retrieval import SelectedMemory, UtilityFeatures
from recap.environments.mock_api import PaginatedApiEnvironment
from recap.environments.reference_agent import ReferenceAgent
from recap.evals.datasets import (
    FAMILY_SIGNATURES,
    FAMILY_SPLITS,
    BenchmarkTask,
    build_benchmark_v2_tasks,
    check_leakage,
    split_hash,
)
from recap.repair_compiler.candidate_generation import (
    DeterministicGenerator,
    build_candidate,
)
from recap.repair_compiler.validation import MockSandboxValidator

_CATEGORY = {
    "invalid_cursor": "invalid_pagination_cursor",
    "renamed_parameter": "tool_parameter_mismatch",
    "rate_limit": "transient_rate_limit",
    "timeout": "transient_timeout",
    "auth_expiry": "authentication_expired",
    "idempotency_key": "missing_idempotency_key",
    "response_field": "response_shape_changed",
    "required_parameter": "missing_required_parameter",
    "enum_change": "enum_value_mismatch",
    "date_format": "timestamp_format_mismatch",
}


def _recovery_actions(family: str, version: str) -> list[dict[str, Any]]:
    if family == "invalid_cursor":
        return [{"action_type": "refresh_pagination_cursor", "tool_name": "list_records"}]
    if family == "renamed_parameter":
        old, new = ("page_token", "cursor") if version == "1" else ("cursor", "page_token")
        return [
            {
                "action_type": "map_parameter",
                "tool_name": "list_records",
                "parameters": {"from": old, "to": new},
            }
        ]
    if family in {"rate_limit", "timeout"}:
        return [{"action_type": "retry_with_backoff", "tool_name": "list_records"}]
    if family == "auth_expiry":
        return [
            {
                "action_type": "refresh_auth_token",
                "tool_name": "list_records",
                "parameters": {"token": f"fresh-token-v{version}"},
            }
        ]
    if family == "idempotency_key":
        key = "idempotency_key" if version == "1" else "request_id"
        return [
            {
                "action_type": "supply_required_parameter",
                "tool_name": "list_records",
                "parameters": {key: "bench-key"},
            }
        ]
    if family == "response_field":
        field = "items" if version == "1" else "data"
        return [
            {
                "action_type": "map_response_field",
                "tool_name": "list_records",
                "parameters": {"field": field},
            }
        ]
    if family == "required_parameter":
        key = "account" if version == "1" else "account_id"
        return [
            {
                "action_type": "supply_required_parameter",
                "tool_name": "list_records",
                "parameters": {key: "acct-bench"},
            }
        ]
    if family == "enum_change":
        value = "created" if version == "1" else "created_at"
        return [
            {
                "action_type": "map_parameter_value",
                "tool_name": "list_records",
                "parameters": {"field": "sort", "to": value},
            }
        ]
    if family == "date_format":
        value = "2026/01/01" if version == "1" else "2026-01-01T00:00:00Z"
        return [
            {
                "action_type": "normalize_timestamp",
                "tool_name": "list_records",
                "parameters": {"field": "since", "value": value},
            }
        ]
    raise ValueError(family)


def _memory(actions: list[dict[str, Any]]) -> SelectedMemory:
    return SelectedMemory(
        memory_id="mem_benchmark",
        memory_version=1,
        title="benchmark recovery",
        summary="benchmark recovery",
        strategy_summary="benchmark recovery",
        actions=actions,
        features=UtilityFeatures(),
        compatibility_status="compatible",
        content_hash="sha256:benchmark",
    )


def test_implicit_and_explicit_failure_family_are_distinguished() -> None:
    default = PaginatedApiEnvironment(version="1")
    explicit = PaginatedApiEnvironment(version="1", failure_family="rate_limit")

    assert default.failure_family == "invalid_cursor"
    assert default.family_explicit is False
    assert explicit.failure_family == "rate_limit"
    assert explicit.family_explicit is True


def test_v2_dataset_has_120_tasks_across_ten_families() -> None:
    tasks = build_benchmark_v2_tasks()
    assert len(tasks) == 120
    assert len({task.task_family for task in tasks}) == 10
    assert {task.task_family for task in tasks} == set(FAMILY_SPLITS)
    assert {task.environment_version for task in tasks} == {"1", "2"}
    assert all(task.hard_negative_kinds for task in tasks)


def test_v2_dataset_holds_out_entire_families_by_split() -> None:
    tasks = build_benchmark_v2_tasks()
    by_split: dict[str, set[str]] = {}
    for task in tasks:
        by_split.setdefault(task.split, set()).add(task.task_family)
    assert set(by_split) == {"train", "dev", "test"}
    assert by_split["train"].isdisjoint(by_split["dev"])
    assert by_split["train"].isdisjoint(by_split["test"])
    assert by_split["dev"].isdisjoint(by_split["test"])
    assert len(by_split["test"]) == 3


def test_v2_dataset_is_deterministic_and_paraphrases_are_unique() -> None:
    first = build_benchmark_v2_tasks()
    second = build_benchmark_v2_tasks()
    assert split_hash(first) == split_hash(second)
    assert [task.to_dict() for task in first] == [task.to_dict() for task in second]
    by_family_version: dict[tuple[str, str, str], list[str]] = {}
    for task in first:
        key = (task.split, task.task_family, task.environment_version)
        by_family_version.setdefault(key, []).append(task.instruction)
    for instructions in by_family_version.values():
        assert len(instructions) == len(set(instructions))


def test_v2_dataset_has_no_exact_or_near_duplicate_split_leakage() -> None:
    tasks = build_benchmark_v2_tasks()
    assert check_leakage(tasks, require_family_holdout=True) == []


def test_leakage_check_flags_near_duplicate_wording_across_splits() -> None:
    left = BenchmarkTask(
        task_id="left",
        task_family="invalid_cursor",
        task_version="1",
        instruction="Collect every record from the paginated API.",
        environment_version="1",
        split="train",
        failure_mode="invalid_cursor",
    )
    right = BenchmarkTask(
        task_id="right",
        task_family="renamed_parameter",
        task_version="1",
        instruction="Collect every record from the paginated API now.",
        environment_version="1",
        split="test",
        failure_mode="renamed_parameter",
    )
    issues = check_leakage([left, right], near_duplicate_threshold=0.8)
    assert any("near-duplicate instruction" in issue for issue in issues)


def test_each_family_fails_without_memory_and_recovers_with_typed_memory() -> None:
    for family in FAMILY_SPLITS:
        for version in ("1", "2"):
            without = ReferenceAgent().run(
                PaginatedApiEnvironment(version=version, failure_family=family)
            )
            assert without.success is False, (family, version)

            with_memory = ReferenceAgent().run(
                PaginatedApiEnvironment(version=version, failure_family=family),
                memory=_memory(_recovery_actions(family, version)),
            )
            assert with_memory.success is True, (family, version, with_memory.final_error)


def test_provoked_failure_codes_match_the_dataset_signatures() -> None:
    for family, signature in FAMILY_SIGNATURES.items():
        expected_code = signature.split("|")[1]
        for version in ("1", "2"):
            environment = PaginatedApiEnvironment(version=version, failure_family=family)
            failure = environment.provoke_failure()
            assert failure.ok is False, (family, version)
            assert (failure.error or {})["error_code"] == expected_code, (family, version)


def test_generated_candidates_pass_the_mock_sandbox_for_every_family() -> None:
    for family, signature in FAMILY_SIGNATURES.items():
        error_type, error_code, _ = signature.split("|")
        for version in ("1", "2"):
            environment = PaginatedApiEnvironment(version=version, failure_family=family)
            fingerprint = environment.fingerprint()
            window = FailureWindow(
                run_id="run_benchmark_seed",
                start_sequence=1,
                end_sequence=2,
                start_event_id="evt_1",
                end_event_id="evt_2",
                failure_category=_CATEGORY[family],
                normalized_signature=signature,
                error_type=error_type,
                error_code=error_code,
                tool_name="list_records",
                event_ids=["evt_1", "evt_2"],
            )
            drafts = DeterministicGenerator().generate(
                window=window,
                environment={
                    "tool_schema_hash": fingerprint.tool_schema_hashes["list_records"],
                    "environment_version": version,
                },
            )
            assert len(drafts) == 1, (family, version, len(drafts))
            candidate = build_candidate(
                drafts[0],
                window=window,
                workspace_id="benchmark",
                environment_id=fingerprint.environment_id,
                environment_fingerprint=fingerprint.fingerprint,
                generated_by=DeterministicGenerator.name,
            )
            result = MockSandboxValidator().validate(candidate, environment=fingerprint)
            assert result.status is ValidationStatus.PASSED, (
                family,
                version,
                result.detail,
                result.checks,
            )
