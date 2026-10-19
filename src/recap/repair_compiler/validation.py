"""Repair validators.

Increasing rigor:
  1. StaticValidator — schema, allowlist, evidence, secret checks (no execution)
  2. MockSandboxValidator — actually reproduces the failure in a fresh deterministic
     mock environment, applies the typed strategy, and checks postconditions.

Both produce provenance: environment fingerprint, validator version, candidate
hash, attempt, timing, and observed outcome. Timeouts/failures are ``inconclusive``
or ``blocked``, never ``passed``.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Callable

from recap.domain.enums import ActionType, PredicateResult, ValidationStatus
from recap.domain.environments import EnvironmentFingerprint
from recap.domain.predicates import evaluate_predicate
from recap.domain.repairs import RepairCandidate, ValidationResult
from recap.environments.mock_api import MockApiError, PaginatedApiEnvironment
from recap.recorder.redaction import contains_secret

VALIDATOR_VERSION = "1"


def _result(
    candidate: RepairCandidate,
    *,
    name: str,
    status: ValidationStatus,
    attempt: int,
    environment: EnvironmentFingerprint | None,
    detail: str,
    checks: list[dict[str, Any]],
    started: datetime,
    observed: str = "",
    error: str = "",
) -> ValidationResult:
    return ValidationResult(
        candidate_id=candidate.id,
        validator_name=name,
        validator_version=VALIDATOR_VERSION,
        status=status,
        attempt=attempt,
        environment_id=environment.environment_id if environment else "",
        environment_fingerprint=environment.fingerprint if environment else "",
        candidate_hash=candidate.content_hash or candidate.compute_hash(),
        observed_outcome=observed,
        detail=detail,
        checks=checks,
        started_at=started,
        finished_at=datetime.now(UTC),
        error=error,
    )


class StaticValidator:
    name = "static"

    def validate(
        self,
        candidate: RepairCandidate,
        *,
        environment: EnvironmentFingerprint,
        attempt: int = 1,
    ) -> ValidationResult:
        started = datetime.now(UTC)
        checks: list[dict[str, Any]] = []
        ok = True

        def check(name: str, passed: bool, detail: str = "") -> None:
            nonlocal ok
            checks.append({"check": name, "passed": passed, "detail": detail})
            ok = ok and passed

        check("has_source_run", bool(candidate.source_run_id))
        check("has_evidence", bool(candidate.diagnosis.evidence_event_ids))
        check("has_preconditions", bool(candidate.preconditions))
        check("has_postconditions", bool(candidate.postconditions))
        check(
            "actions_allowlisted",
            all(a.action_type in set(ActionType) for a in candidate.strategy.actions),
        )
        check(
            "no_destructive_actions",
            all(a.side_effect_class.value != "destructive" for a in candidate.strategy.actions),
        )
        payload_text = candidate.model_dump_json()
        check("no_secret_leakage", not contains_secret(payload_text))
        check("bounded_strategy", len(candidate.strategy.actions) <= 8)

        return _result(
            candidate,
            name=self.name,
            status=ValidationStatus.PASSED if ok else ValidationStatus.FAILED,
            attempt=attempt,
            environment=environment,
            detail="static validation passed" if ok else "static validation failed",
            checks=checks,
            started=started,
        )


class MockSandboxValidator:
    """Executes the typed repair inside a fresh, disposable mock API.

    The sandbox is created per validation; no external side effects are possible.
    A negative guard verifies the candidate's preconditions reject a mismatched
    schema, which is how stale/incorrect repairs are caught.
    """

    name = "mock_sandbox"

    def __init__(
        self,
        *,
        environment_factory: Callable[[str], PaginatedApiEnvironment] | None = None,
    ) -> None:
        self.environment_factory = environment_factory or (lambda version: PaginatedApiEnvironment(version=version))

    def validate(
        self,
        candidate: RepairCandidate,
        *,
        environment: EnvironmentFingerprint,
        attempt: int = 1,
    ) -> ValidationResult:
        started = datetime.now(UTC)
        checks: list[dict[str, Any]] = []
        version = environment.environment_version or "1"
        sandbox = self.environment_factory(version)

        # --- 1. Reproduce the failure deterministically ---------------------
        reproduction_ok = False
        expected_code = candidate.failure_window.error_code or "INVALID_CURSOR"
        try:
            if version == "1":
                sandbox.expire_cursor("start")
                first = sandbox.list_records(tenant=sandbox.tenant, cursor="start")
            else:
                # v2 renamed the parameter; replay the legacy call shape.
                first = sandbox.list_records(tenant=sandbox.tenant, cursor="start")
            reproduction_ok = not first.ok and (first.error or {}).get("error_code") == expected_code
            checks.append(
                {
                    "check": "failure_reproduced",
                    "passed": reproduction_ok,
                    "detail": (
                        f"expected {expected_code}, observed "
                        f"{(first.error or {}).get('error_code', 'success')}"
                    ),
                }
            )
        except MockApiError as exc:
            checks.append({"check": "failure_reproduced", "passed": False, "detail": str(exc)})

        # --- 2. Preconditions must hold in the target environment -----------
        state = {
            "environment_id": environment.environment_id,
            "environment_version": environment.environment_version,
            "tool_schema_fingerprints": environment.tool_schema_hashes,
            "error_code": expected_code,
        }
        precondition_results = [evaluate_predicate(p, state) for p in candidate.preconditions]
        preconditions_ok = all(result is PredicateResult.SATISFIED for result, _ in precondition_results)
        checks.append(
            {
                "check": "preconditions_hold_in_sandbox",
                "passed": preconditions_ok,
                "detail": "; ".join(reason for _, reason in precondition_results),
            }
        )
        if not preconditions_ok:
            return _result(
                candidate,
                name=self.name,
                status=ValidationStatus.FAILED,
                attempt=attempt,
                environment=environment,
                detail="candidate preconditions do not hold in the target sandbox",
                checks=checks,
                started=started,
            )

        # --- 3. Negative guard: preconditions reject mismatched schema ------
        other_version = "2" if version == "1" else "1"
        other = self.environment_factory(other_version)
        other_state = dict(state)
        other_state["tool_schema_fingerprints"] = other.fingerprint().tool_schema_hashes
        other_results = [evaluate_predicate(p, other_state) for p in candidate.preconditions]
        guard_ok = any(result is PredicateResult.VIOLATED for result, _ in other_results)
        checks.append(
            {
                "check": "rejects_mismatched_schema",
                "passed": guard_ok,
                "detail": f"checked against version {other_version}",
            }
        )

        # --- 4. Apply the typed strategy in the sandbox ---------------------
        applied, apply_detail = self._apply(candidate, sandbox)
        checks.append({"check": "strategy_executed", "passed": applied, "detail": apply_detail})

        # --- 5. Postconditions ---------------------------------------------
        post_state = {
            **sandbox.snapshot(),
            "records_collected": len(sandbox.records),
            "tool_call_succeeded": applied,
        }
        post_results = [evaluate_predicate(p, post_state) for p in candidate.postconditions]
        postconditions_ok = all(result is PredicateResult.SATISFIED for result, _ in post_results)
        checks.append(
            {
                "check": "postconditions_satisfied",
                "passed": postconditions_ok,
                "detail": "; ".join(reason for _, reason in post_results),
            }
        )

        passed = reproduction_ok and preconditions_ok and guard_ok and applied and postconditions_ok
        return _result(
            candidate,
            name=self.name,
            status=ValidationStatus.PASSED if passed else ValidationStatus.FAILED,
            attempt=attempt,
            environment=environment,
            detail=(
                "repair reproduced the failure, executed safely, and satisfied postconditions"
                if passed
                else "one or more sandbox checks failed"
            ),
            checks=checks,
            started=started,
            observed="verified_success" if passed else "verified_failure",
        )

    def _apply(self, candidate: RepairCandidate, sandbox: PaginatedApiEnvironment) -> tuple[bool, str]:
        """Interpret allowlisted actions against the sandbox tool only."""
        token: str | None = None
        collected = 0
        for action in candidate.strategy.actions:
            if action.action_type is ActionType.RETRY_WITH_BACKOFF:
                sandbox.fail_next_calls(1)
                result = sandbox.list_records(tenant=sandbox.tenant)
                if not result.ok:
                    # one bounded retry
                    result = sandbox.list_records(tenant=sandbox.tenant)
                if not result.ok:
                    return False, "retry action exhausted"
                collected += len(result.payload.get("items", []))
                token = result.payload.get("cursor") or result.payload.get("page_token")
            elif action.action_type is ActionType.REFRESH_PAGINATION_CURSOR:
                first = sandbox.list_records(tenant=sandbox.tenant)
                if not first.ok:
                    return False, "fresh cursor request failed"
                collected += len(first.payload.get("items", []))
                token = first.payload.get("cursor") or first.payload.get("page_token")
            elif action.action_type is ActionType.MAP_PARAMETER:
                mapping = action.parameters or {}
                old = mapping.get("from", "cursor")
                new = mapping.get("to", "page_token")
                parameters = {"tenant": sandbox.tenant, new: "start"}
                if sandbox.version == "2" and old == "cursor":
                    # Mapped parameter name is valid in v2; the value must be refreshed.
                    first = sandbox.list_records(tenant=sandbox.tenant)
                    if not first.ok:
                        return False, "mapped parameter call failed"
                    collected += len(first.payload.get("items", []))
                    token = first.payload.get("page_token")
                else:
                    result = sandbox.list_records(**parameters)
                    if not result.ok:
                        return False, "mapped parameter call failed"
                    collected += len(result.payload.get("items", []))
                    token = result.payload.get("page_token") or result.payload.get("cursor")
            elif action.action_type is ActionType.VERIFY_RESOURCE_EXISTS:
                result = sandbox.list_records(tenant=sandbox.tenant, limit=1)
                if not result.ok:
                    return False, "resource verification failed"
            elif action.action_type in {ActionType.HALT, ActionType.REQUEST_MISSING_FIELD}:
                return False, "strategy intentionally halts"

        if token is None:
            return collected > 0, "no pagination token produced"
        # Complete pagination to prove the repair works end to end.
        while token is not None:
            parameters: dict[str, Any] = {"tenant": sandbox.tenant}
            if sandbox.version == "1":
                parameters["cursor"] = token
            else:
                parameters["page_token"] = token
            page = sandbox.list_records(**parameters)
            if not page.ok:
                return False, f"pagination continuation failed: {(page.error or {}).get('error_code')}"
            collected += len(page.payload.get("items", []))
            token = page.payload.get("cursor") or page.payload.get("page_token")
        return collected == len(sandbox.records), f"collected {collected}/{len(sandbox.records)} records"


class ValidatorRegistry:
    def __init__(self) -> None:
        self._validators: dict[str, Any] = {}

    def register(self, validator: Any) -> None:
        self._validators[validator.name] = validator

    def get(self, name: str) -> Any:
        if name not in self._validators:
            raise KeyError(f"unknown validator: {name}")
        return self._validators[name]

    def names(self) -> list[str]:
        return sorted(self._validators)


def build_default_registry() -> ValidatorRegistry:
    registry = ValidatorRegistry()
    registry.register(StaticValidator())
    registry.register(MockSandboxValidator())
    return registry
