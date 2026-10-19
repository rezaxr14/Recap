"""Structured, safe precondition/postcondition predicates.

This is deliberately a tiny language, not an expression evaluator. Unknown
fields and missing observations evaluate to ``unknown``; unknown never equals
satisfied.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from recap.domain.enums import PredicateResult

ALLOWED_OPERATORS = {
    "eq",
    "neq",
    "in",
    "not_in",
    "exists",
    "not_exists",
    "version_gte",
    "version_lt",
    "gte",
    "lte",
    "contains",
    "matches",
}

_FIELD_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_.]{0,127}$")
_VERSION_RE = re.compile(r"^[0-9]+(\.[0-9]+){0,3}$")


class Predicate(BaseModel):
    """A single typed predicate evaluated against a bounded state snapshot."""

    field: str
    operator: str
    value: Any = None
    description: str = ""

    @field_validator("field")
    @classmethod
    def _safe_field(cls, value: str) -> str:
        if not _FIELD_RE.match(value):
            raise ValueError(f"unsafe predicate field: {value!r}")
        if "__" in value:
            raise ValueError("dunder paths are not allowed")
        return value

    @field_validator("operator")
    @classmethod
    def _safe_operator(cls, value: str) -> str:
        if value not in ALLOWED_OPERATORS:
            raise ValueError(f"unsupported predicate operator: {value!r}")
        return value

    @model_validator(mode="after")
    def _validate_value(self) -> Predicate:
        if self.operator in {"exists", "not_exists"}:
            return self
        if self.value is None:
            raise ValueError(f"operator {self.operator} requires a value")
        if self.operator in {"in", "not_in"} and not isinstance(self.value, list):
            raise ValueError(f"operator {self.operator} requires a list value")
        if self.operator in {"version_gte", "version_lt"} and (
            not isinstance(self.value, str) or not _VERSION_RE.match(self.value)
        ):
            raise ValueError("version predicates require a dotted numeric version")
        if self.operator in {"gte", "lte"} and not isinstance(self.value, (int, float)):
            raise ValueError("numeric bounds require a number")
        return self


def _lookup(state: dict[str, Any], path: str) -> tuple[bool, Any]:
    current: Any = state
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return False, None
        current = current[part]
    return True, current


def _version_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split("."))


def evaluate_predicate(predicate: Predicate, state: dict[str, Any]) -> tuple[PredicateResult, str]:
    found, actual = _lookup(state, predicate.field)
    op = predicate.operator
    if op == "exists":
        return (PredicateResult.SATISFIED if found else PredicateResult.VIOLATED), (
            "field present" if found else "field absent"
        )
    if op == "not_exists":
        return (PredicateResult.SATISFIED if not found else PredicateResult.VIOLATED), (
            "field absent" if not found else "field present"
        )
    if not found:
        return PredicateResult.UNKNOWN, f"field {predicate.field!r} is unknown"

    expected = predicate.value
    if op == "eq":
        result = actual == expected
    elif op == "neq":
        result = actual != expected
    elif op == "in":
        result = actual in expected
    elif op == "not_in":
        result = actual not in expected
    elif op in {"version_gte", "version_lt"}:
        try:
            actual_v = _version_tuple(str(actual))
            expected_v = _version_tuple(str(expected))
        except (TypeError, ValueError):
            return PredicateResult.UNKNOWN, "unparseable version value"
        result = actual_v >= expected_v if op == "version_gte" else actual_v < expected_v
    elif op == "gte":
        result = actual >= expected
    elif op == "lte":
        result = actual <= expected
    elif op == "contains":
        if isinstance(actual, str):
            result = str(expected) in actual
        elif isinstance(actual, (list, tuple, set)):
            result = expected in actual
        else:
            return PredicateResult.UNKNOWN, "contains applied to unsupported type"
    elif op == "matches":
        try:
            result = re.search(str(expected), str(actual)) is not None
        except re.error:
            return PredicateResult.UNKNOWN, "invalid match pattern"
    else:  # pragma: no cover - guarded by validator
        return PredicateResult.UNKNOWN, "unsupported operator"
    return (
        PredicateResult.SATISFIED if result else PredicateResult.VIOLATED,
        f"{predicate.field} {op} {expected!r}",
    )


class PredicateEvaluation(BaseModel):
    predicate: Predicate
    result: PredicateResult
    reason: str


class ApplicabilityResult(BaseModel):
    result: PredicateResult
    evaluations: list[PredicateEvaluation] = Field(default_factory=list)
    reason: str = ""

    @property
    def satisfied_count(self) -> int:
        return sum(1 for item in self.evaluations if item.result is PredicateResult.SATISFIED)

    @property
    def violated_count(self) -> int:
        return sum(1 for item in self.evaluations if item.result is PredicateResult.VIOLATED)

    @property
    def unknown_count(self) -> int:
        return sum(1 for item in self.evaluations if item.result is PredicateResult.UNKNOWN)

    @property
    def satisfaction_ratio(self) -> float:
        if not self.evaluations:
            return 1.0
        return self.satisfied_count / len(self.evaluations)


def evaluate_all(predicates: list[Predicate], state: dict[str, Any]) -> ApplicabilityResult:
    evaluations = [
        PredicateEvaluation(predicate=p, result=r, reason=reason)
        for p, (r, reason) in ((p, evaluate_predicate(p, state)) for p in predicates)
    ]
    if not evaluations:
        return ApplicabilityResult(result=PredicateResult.UNKNOWN, reason="no predicates declared")
    if any(e.result is PredicateResult.VIOLATED for e in evaluations):
        return ApplicabilityResult(
            result=PredicateResult.VIOLATED,
            evaluations=evaluations,
            reason="one or more preconditions are violated",
        )
    if any(e.result is PredicateResult.UNKNOWN for e in evaluations):
        return ApplicabilityResult(
            result=PredicateResult.UNKNOWN,
            evaluations=evaluations,
            reason="one or more preconditions are unknown",
        )
    return ApplicabilityResult(
        result=PredicateResult.SATISFIED,
        evaluations=evaluations,
        reason="all preconditions satisfied",
    )
