"""Deterministic mock paginated REST API with interchangeable failure families.

The default family is cursor pagination, preserved for the demo and the first
benchmark. The environment can also reproduce the common drift families used by
the Phase 2 benchmark (renamed parameters, auth expiry, rate limits, missing
required fields, response-field renames, enum changes, timestamp format
changes, timeouts, and idempotency keys). Everything is in-memory and read-only.

Two API versions exist so staleness is reproducible. Not every family changes
the request schema between versions; response-field and date-format changes are
deliberately represented as observable behavior rather than artificial errors.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from recap.domain.enums import SideEffectClass
from recap.domain.environments import (
    EnvironmentFingerprint,
    ToolDescriptor,
    build_fingerprint,
    hash_tool_schema,
)

FAILURE_FAMILIES: tuple[str, ...] = (
    "invalid_cursor",
    "renamed_parameter",
    "rate_limit",
    "timeout",
    "auth_expiry",
    "idempotency_key",
    "response_field",
    "required_parameter",
    "enum_change",
    "date_format",
)

# A stable, human-readable error for each family. The environment uses these
# for the first failing call so the repair compiler always sees the same code.
FAMILY_ERRORS: dict[str, tuple[str, str, str]] = {
    "invalid_cursor": ("ValidationError", "INVALID_CURSOR", "cursor is invalid or expired"),
    "renamed_parameter": (
        "ValidationError",
        "UNKNOWN_PARAMETER",
        "unknown parameter 'cursor'; this API version requires 'page_token'",
    ),
    "rate_limit": (
        "TransientError",
        "RATE_LIMITED",
        "upstream rate limit exceeded; retry after 1 second",
    ),
    "timeout": ("TransientError", "TIMEOUT", "upstream request timed out"),
    "auth_expiry": (
        "AuthError",
        "TOKEN_EXPIRED",
        "the access token expired; refresh the token before retrying",
    ),
    "idempotency_key": (
        "ValidationError",
        "MISSING_IDEMPOTENCY_KEY",
        "request_id is required for this API version",
    ),
    "response_field": (
        "ResponseShapeError",
        "MISSING_RESPONSE_FIELD",
        "records are returned under 'data', not 'items'",
    ),
    "required_parameter": (
        "ValidationError",
        "MISSING_REQUIRED_PARAMETER",
        "account_id is required for this API version",
    ),
    "enum_change": (
        "ValidationError",
        "INVALID_ENUM_VALUE",
        "sort must be one of ['created_at']",
    ),
    "date_format": (
        "ValidationError",
        "INVALID_DATE_FORMAT",
        "since must be an ISO-8601 timestamp for this API version",
    ),
}

BASE_SCHEMA_PROPERTIES: dict[str, Any] = {
    "tenant": {"type": "string"},
    "limit": {"type": "integer", "minimum": 1, "maximum": 10},
}


def build_tool_schema(family: str, version: str) -> dict[str, Any]:
    """Return a versioned schema whose hash identifies the environment."""
    if family not in FAILURE_FAMILIES:
        raise ValueError(f"unknown failure family: {family}")
    properties = copy.deepcopy(BASE_SCHEMA_PROPERTIES)
    required = ["tenant"]
    if family in {"invalid_cursor", "rate_limit", "timeout", "response_field"}:
        properties["cursor"] = {
            "type": "string",
            "description": "Opaque cursor from a previous page",
        }
    elif family == "renamed_parameter":
        parameter = "cursor" if version == "1" else "page_token"
        properties[parameter] = {
            "type": "string",
            "description": "Opaque cursor" if version == "1" else "Opaque page token",
        }
    elif family == "auth_expiry":
        properties["auth_token"] = {
            "type": "string",
            "description": "Bearer token; expired tokens are rejected",
        }
        required.append("auth_token")
    elif family == "idempotency_key":
        key = "idempotency_key" if version == "1" else "request_id"
        properties[key] = {"type": "string", "description": "Idempotency key"}
        required.append(key)
    elif family == "required_parameter":
        parameter = "account" if version == "1" else "account_id"
        properties[parameter] = {"type": "string", "description": "Billing account id"}
        required.append(parameter)
    elif family == "enum_change":
        properties["sort"] = {
            "type": "string",
            "enum": ["created"] if version == "1" else ["created_at"],
        }
        required.append("sort")
    elif family == "date_format":
        properties["since"] = {
            "type": "string",
            "description": ("YYYY/MM/DD" if version == "1" else "ISO-8601 timestamp with timezone"),
        }
        required.append("since")
    return {
        "name": "list_records",
        "description": (
            f"List records using {family.replace('_', ' ')} semantics (API v{version})."
        ),
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
    }


# Backwards-compatible names used by older docs/tests.
V1_TOOL_SCHEMA = build_tool_schema("invalid_cursor", "1")
V2_TOOL_SCHEMA = build_tool_schema("invalid_cursor", "2")


class MockApiError(Exception):
    def __init__(self, error_type: str, error_code: str, message: str, status: int = 400) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.error_code = error_code
        self.message = message
        self.status = status

    def to_payload(self) -> dict[str, Any]:
        return {
            "error_type": self.error_type,
            "error_code": self.error_code,
            "error_message": self.message,
            "http_status": self.status,
        }


@dataclass
class ToolCallResult:
    ok: bool
    payload: dict[str, Any]
    error: dict[str, Any] | None = None


@dataclass
class PaginatedApiEnvironment:
    """Controllable, deterministic mock API with one failure family."""

    version: str = "1"
    failure_family: str = ""
    tenant: str = "acme"
    records: list[dict[str, Any]] = field(default_factory=list)
    page_size: int = 3
    inject_transient_failures: int = 0
    _calls: int = 0
    _fail_next: int = 0
    _fail_next_code: str = "RATE_LIMITED"
    _cursors: dict[str, int] = field(default_factory=dict)
    _cursor_counter: int = 0
    _expired_cursors: set[str] = field(default_factory=set)
    _access_log: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.failure_family:
            # Preserve the original demo contract: v1 used a cursor, v2 renamed
            # the parameter to page_token. New benchmark tasks always pass an
            # explicit family.
            self.failure_family = "invalid_cursor" if self.version == "1" else "renamed_parameter"
        if self.failure_family not in FAILURE_FAMILIES:
            raise ValueError(f"unknown failure family: {self.failure_family}")
        if not self.records:
            self.records = [
                {"id": f"rec-{index:03d}", "tenant": self.tenant, "value": index * 10}
                for index in range(1, 8)
            ]
        if self.inject_transient_failures:
            self._fail_next = self.inject_transient_failures
        elif self.failure_family == "rate_limit":
            self._fail_next = 1
            self._fail_next_code = "RATE_LIMITED"
        elif self.failure_family == "timeout":
            self._fail_next = 1
            self._fail_next_code = "TIMEOUT"

    # ----------------------------------------------------------- metadata
    @property
    def tool_schema(self) -> dict[str, Any]:
        return build_tool_schema(self.failure_family, self.version)

    def fingerprint(self) -> EnvironmentFingerprint:
        schema = self.tool_schema
        descriptor = ToolDescriptor(
            name="list_records",
            version=self.version,
            schema_hash=hash_tool_schema(schema),
            side_effect_class=str(SideEffectClass.READ_ONLY),
        )
        return build_fingerprint(
            environment_id=f"mock-api-{self.failure_family}-v{self.version}",
            environment_type="mock_api",
            environment_version=self.version,
            tools=[descriptor],
            api_schema=schema,
            task_suite_version="mock-api-suite-v2",
            feature_flags={"failure_family": self.failure_family},
        )

    def naive_parameters(self) -> dict[str, Any]:
        """The pre-memory request shape used by the reference agent.

        It is intentionally wrong for the selected family: the benchmark asks
        whether retrieval supplies the correction.
        """
        parameters: dict[str, Any] = {"tenant": self.tenant}
        if self.failure_family == "invalid_cursor":
            parameters["cursor"] = "start"
        elif self.failure_family == "renamed_parameter":
            # The agent assumes the wrong side of the rename.
            parameters["page_token" if self.version == "1" else "cursor"] = "start"
        elif self.failure_family == "auth_expiry":
            parameters["auth_token"] = "stale-token"
        elif self.failure_family == "enum_change":
            parameters["sort"] = "legacy"
        elif self.failure_family == "date_format":
            parameters["since"] = "2026-01-01"
        return parameters

    def naive_response_field(self) -> str:
        """Record field the pre-memory agent expects."""
        return "records" if self.failure_family == "response_field" else "items"

    def provoke_failure(self) -> ToolCallResult:
        """Replay the family's first failing call in a fresh sandbox."""
        call = self.list_records(**self.naive_parameters())
        if (
            call.ok
            and self.failure_family == "response_field"
            and call.payload.get(self.naive_response_field()) is None
        ):
            return ToolCallResult(
                ok=False,
                payload=call.payload,
                error=dict(family_error("response_field")),
            )
        return call

    # -------------------------------------------------------------- faults
    def expire_cursor(self, cursor: str) -> None:
        self._expired_cursors.add(cursor)

    def fail_next_calls(self, count: int = 1, error_code: str = "RATE_LIMITED") -> None:
        self._fail_next = count
        self._fail_next_code = error_code

    def access_log(self) -> list[dict[str, Any]]:
        return list(self._access_log)

    # --------------------------------------------------------------- tools
    def list_records(
        self,
        tenant: str,
        cursor: str | None = None,
        page_token: str | None = None,
        auth_token: str | None = None,
        idempotency_key: str | None = None,
        request_id: str | None = None,
        account: str | None = None,
        account_id: str | None = None,
        sort: str | None = None,
        since: str | None = None,
        limit: int | None = None,
    ) -> ToolCallResult:
        self._calls += 1
        self._access_log.append(
            {
                "tool": "list_records",
                "tenant": tenant,
                "cursor": cursor,
                "page_token": page_token,
                "auth_token": auth_token,
                "idempotency_key": idempotency_key,
                "request_id": request_id,
                "account": account,
                "account_id": account_id,
                "sort": sort,
                "since": since,
            }
        )
        if self._fail_next > 0:
            self._fail_next -= 1
            error = MockApiError(
                "TransientError",
                self._fail_next_code,
                "upstream request failed transiently",
                status=429 if self._fail_next_code == "RATE_LIMITED" else 504,
            )
            return ToolCallResult(ok=False, payload={}, error=error.to_payload())
        try:
            payload = self._list_records(
                tenant=tenant,
                cursor=cursor,
                page_token=page_token,
                auth_token=auth_token,
                idempotency_key=idempotency_key,
                request_id=request_id,
                account=account,
                account_id=account_id,
                sort=sort,
                since=since,
                limit=limit,
            )
            return ToolCallResult(ok=True, payload=payload)
        except MockApiError as exc:
            return ToolCallResult(ok=False, payload={}, error=exc.to_payload())

    def _list_records(
        self,
        *,
        tenant: str,
        cursor: str | None,
        page_token: str | None,
        auth_token: str | None,
        idempotency_key: str | None,
        request_id: str | None,
        account: str | None,
        account_id: str | None,
        sort: str | None,
        since: str | None,
        limit: int | None,
    ) -> dict[str, Any]:
        if tenant != self.tenant:
            raise MockApiError(
                "NotFoundError", "TENANT_NOT_FOUND", f"tenant {tenant!r} not found", 404
            )

        family = self.failure_family
        expected_token = f"fresh-token-v{self.version}"
        if family == "auth_expiry" and auth_token != expected_token:
            raise MockApiError(
                "AuthError",
                "TOKEN_EXPIRED",
                "the access token expired; refresh the token before retrying",
                401,
            )
        if family == "idempotency_key":
            key_name = "idempotency_key" if self.version == "1" else "request_id"
            key_value = idempotency_key if self.version == "1" else request_id
            if not key_value:
                raise MockApiError(
                    "ValidationError",
                    "MISSING_IDEMPOTENCY_KEY",
                    f"{key_name} is required for this API version",
                    400,
                )
        if family == "required_parameter":
            parameter = "account" if self.version == "1" else "account_id"
            value = account if self.version == "1" else account_id
            if not value:
                raise MockApiError(
                    "ValidationError",
                    "MISSING_REQUIRED_PARAMETER",
                    f"{parameter} is required for this API version",
                    400,
                )
        expected_sort = "created" if self.version == "1" else "created_at"
        if family == "enum_change" and sort != expected_sort:
            raise MockApiError(
                "ValidationError",
                "INVALID_ENUM_VALUE",
                f"sort must be one of ['{expected_sort}']",
                422,
            )
        if family == "date_format":
            valid_date = _is_slash_date(since) if self.version == "1" else _is_iso_datetime(since)
            if not valid_date:
                expected = "YYYY/MM/DD" if self.version == "1" else "ISO-8601 timestamp"
                raise MockApiError(
                    "ValidationError",
                    "INVALID_DATE_FORMAT",
                    f"since must be {expected} for this API version",
                    422,
                )

        # The renamed-parameter family rejects the legacy name; the other
        # families keep cursor pagination across both versions.
        if family == "renamed_parameter":
            if self.version == "1" and page_token is not None:
                raise MockApiError(
                    "ValidationError",
                    "UNKNOWN_PARAMETER",
                    "unknown parameter 'page_token'; this API version requires 'cursor'",
                    422,
                )
            if self.version == "2" and cursor is not None:
                raise MockApiError(
                    "ValidationError",
                    "UNKNOWN_PARAMETER",
                    "unknown parameter 'cursor'; this API version requires 'page_token'",
                    422,
                )
        token = page_token if family == "renamed_parameter" and self.version == "2" else cursor
        parameter_name = (
            "page_token" if family == "renamed_parameter" and self.version == "2" else "cursor"
        )
        offset = 0
        if token is not None:
            if token in self._expired_cursors:
                raise MockApiError(
                    "ValidationError",
                    "INVALID_CURSOR",
                    f"{parameter_name} {token!r} is invalid or expired",
                    422,
                )
            if token not in self._cursors:
                raise MockApiError(
                    "ValidationError",
                    "INVALID_CURSOR",
                    f"{parameter_name} {token!r} is invalid or expired",
                    422,
                )
            offset = self._cursors[token]
        size = min(limit or self.page_size, 10)
        page = self.records[offset : offset + size]
        next_offset = offset + size
        payload_key = "data" if family == "response_field" and self.version == "2" else "items"
        response: dict[str, Any] = {
            payload_key: page,
            "count": len(page),
            "total": len(self.records),
            "has_more": next_offset < len(self.records),
        }
        if next_offset < len(self.records):
            self._cursor_counter += 1
            new_token = f"cur_{self.version}_{self._cursor_counter:04d}"
            self._cursors[new_token] = next_offset
            response[parameter_name] = new_token
        return response

    def snapshot(self) -> dict[str, Any]:
        """Environment state used for postcondition checks."""
        return {
            "environment_id": f"mock-api-{self.failure_family}-v{self.version}",
            "environment_version": self.version,
            "environment_type": "mock_api",
            "failure_family": self.failure_family,
            "tool_names": ["list_records"],
            "record_count": len(self.records),
        }


def _is_slash_date(value: str | None) -> bool:
    if not value:
        return False
    return bool(re.match(r"^\d{4}/\d{2}/\d{2}$", value))


def _is_iso_datetime(value: str | None) -> bool:
    if not value:
        return False
    return bool(
        re.match(
            r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})$",
            value,
        )
    )


class MockApiSuite:
    """Registry of versioned environments for the demo/benchmark."""

    def __init__(self) -> None:
        self._environments: dict[tuple[str, str], PaginatedApiEnvironment] = {}

    def register(self, environment: PaginatedApiEnvironment) -> PaginatedApiEnvironment:
        self._environments[(environment.failure_family, environment.version)] = environment
        return environment

    def get(self, version: str = "1", failure_family: str | None = None) -> PaginatedApiEnvironment:
        family = failure_family or ("invalid_cursor" if version == "1" else "renamed_parameter")
        key = (family, version)
        if key not in self._environments:
            self.register(PaginatedApiEnvironment(version=version, failure_family=family))
        return self._environments[key]

    def reset(self) -> None:
        self._environments.clear()


def tool_registry(environment: PaginatedApiEnvironment) -> dict[str, Callable[..., ToolCallResult]]:
    return {"list_records": environment.list_records}


def family_error(failure_family: str) -> Mapping[str, str]:
    error_type, error_code, message = FAMILY_ERRORS[failure_family]
    return {"error_type": error_type, "error_code": error_code, "error_message": message}
