"""Deterministic mock paginated REST API environment.

Two versions exist so staleness is reproducible:
  v1: pagination uses ``cursor`` (opaque string from the previous page)
  v2: pagination uses ``page_token`` and rejects ``cursor`` with a schema error

All state is in-memory and scoped to one environment instance. No network,
no real credentials, no irreversible operations.
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from recap.domain.enums import SideEffectClass
from recap.domain.environments import (
    EnvironmentFingerprint,
    ToolDescriptor,
    build_fingerprint,
    hash_tool_schema,
)

V1_TOOL_SCHEMA = {
    "name": "list_records",
    "description": "List records for a tenant using cursor pagination.",
    "parameters": {
        "type": "object",
        "properties": {
            "tenant": {"type": "string"},
            "cursor": {"type": "string", "description": "Opaque cursor from a previous page"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 10},
        },
        "required": ["tenant"],
        "additionalProperties": False,
    },
}

V2_TOOL_SCHEMA = {
    "name": "list_records",
    "description": "List records for a tenant using page_token pagination.",
    "parameters": {
        "type": "object",
        "properties": {
            "tenant": {"type": "string"},
            "page_token": {"type": "string", "description": "Opaque page token"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 10},
        },
        "required": ["tenant"],
        "additionalProperties": False,
    },
}


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
    """Controllable, deterministic mock API."""

    version: str = "1"
    tenant: str = "acme"
    records: list[dict[str, Any]] = field(default_factory=list)
    page_size: int = 3
    inject_transient_failures: int = 0
    _calls: int = 0
    _fail_next: int = 0
    _cursors: dict[str, int] = field(default_factory=dict)
    _cursor_counter: int = 0
    _expired_cursors: set[str] = field(default_factory=set)
    _access_log: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.records:
            self.records = [
                {"id": f"rec-{index:03d}", "tenant": self.tenant, "value": index * 10}
                for index in range(1, 8)
            ]

    # ----------------------------------------------------------- metadata
    @property
    def tool_schema(self) -> dict[str, Any]:
        return copy.deepcopy(V1_TOOL_SCHEMA if self.version == "1" else V2_TOOL_SCHEMA)

    def fingerprint(self) -> EnvironmentFingerprint:
        schema = self.tool_schema
        descriptor = ToolDescriptor(
            name="list_records",
            version=self.version,
            schema_hash=hash_tool_schema(schema),
            side_effect_class=str(SideEffectClass.READ_ONLY),
        )
        return build_fingerprint(
            environment_id=f"mock-api-v{self.version}",
            environment_type="mock_api",
            environment_version=self.version,
            tools=[descriptor],
            api_schema=schema,
            task_suite_version="mock-api-suite-v1",
        )

    # -------------------------------------------------------------- faults
    def expire_cursor(self, cursor: str) -> None:
        self._expired_cursors.add(cursor)

    def fail_next_calls(self, count: int = 1) -> None:
        self._fail_next = count

    def access_log(self) -> list[dict[str, Any]]:
        return list(self._access_log)

    # --------------------------------------------------------------- tools
    def list_records(
        self,
        tenant: str,
        cursor: str | None = None,
        page_token: str | None = None,
        limit: int | None = None,
    ) -> ToolCallResult:
        self._calls += 1
        self._access_log.append(
            {"tool": "list_records", "tenant": tenant, "cursor": cursor, "page_token": page_token}
        )
        if self._fail_next > 0:
            self._fail_next -= 1
            error = MockApiError(
                "TransientError", "RATE_LIMITED", "upstream rate limit exceeded", status=429
            )
            return ToolCallResult(ok=False, payload={}, error=error.to_payload())
        try:
            payload = self._list_records(tenant, cursor, page_token, limit)
            return ToolCallResult(ok=True, payload=payload)
        except MockApiError as exc:
            return ToolCallResult(ok=False, payload={}, error=exc.to_payload())

    def _list_records(
        self, tenant: str, cursor: str | None, page_token: str | None, limit: int | None
    ) -> dict[str, Any]:
        if tenant != self.tenant:
            raise MockApiError(
                "NotFoundError", "TENANT_NOT_FOUND", f"tenant {tenant!r} not found", 404
            )
        # The declared schema sets additionalProperties=false, so reject the
        # parameter name that belongs to the other API version.
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
        size = min(limit or self.page_size, 10)
        token = cursor if self.version == "1" else page_token
        parameter_name = "cursor" if self.version == "1" else "page_token"
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
        page = self.records[offset : offset + size]
        next_offset = offset + size
        response: dict[str, Any] = {
            "items": page,
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
            "environment_id": f"mock-api-v{self.version}",
            "environment_version": self.version,
            "environment_type": "mock_api",
            "tool_names": ["list_records"],
            "record_count": len(self.records),
        }


class MockApiSuite:
    """Registry of versioned environments for the demo/benchmark."""

    def __init__(self) -> None:
        self._environments: dict[str, PaginatedApiEnvironment] = {}

    def register(self, environment: PaginatedApiEnvironment) -> PaginatedApiEnvironment:
        self._environments[environment.version] = environment
        return environment

    def get(self, version: str = "1") -> PaginatedApiEnvironment:
        if version not in self._environments:
            self.register(PaginatedApiEnvironment(version=version))
        return self._environments[version]

    def reset(self) -> None:
        self._environments.clear()


def tool_registry(environment: PaginatedApiEnvironment) -> dict[str, Callable[..., ToolCallResult]]:
    return {"list_records": environment.list_records}
