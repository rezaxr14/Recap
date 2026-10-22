"""Tiered v3 mock environments: opaque errors, silent drift, and hidden conventions.

This module is deliberately separate from ``mock_api`` so the v2 suite keeps its
exact behavior. The public surface matches ``PaginatedApiEnvironment``: the
reference agent and the LLM agent can use either without special cases.

Silent drift means the old request is accepted and returns HTTP-success-like
output, but the completion check fails. No error message explains the change.
Tier B uses an opaque error that can still be resolved by probing. Tier C relies
on a convention that cannot be derived from the schema or the error.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from recap.domain.enums import SideEffectClass
from recap.domain.environments import (
    EnvironmentFingerprint,
    ToolDescriptor,
    build_fingerprint,
    hash_tool_schema,
)
from recap.environments.mock_api import MockApiError, PaginatedApiEnvironment, ToolCallResult

V3_FAILURE_FAMILIES: tuple[str, ...] = (
    "self_explanatory",
    "opaque_required",
    "silent_cursor",
    "silent_field",
    "silent_enum",
    "silent_account",
    "novel_failure",
)

V3_FAMILY_ERRORS: dict[str, tuple[str, str, str]] = {
    "self_explanatory": ("None", "", ""),
    "opaque_required": (
        "ValidationError",
        "INVALID_REQUEST",
        "request rejected (reference id req-7a1c9e)",
    ),
    "silent_cursor": (
        "SilentBehaviorError",
        "COMPLETION_CHECK_FAILED",
        "request accepted but the completion check failed",
    ),
    "silent_field": (
        "SilentBehaviorError",
        "COMPLETION_CHECK_FAILED",
        "request accepted but the completion check failed",
    ),
    "silent_enum": (
        "SilentBehaviorError",
        "COMPLETION_CHECK_FAILED",
        "request accepted but the completion check failed",
    ),
    "silent_account": (
        "SilentBehaviorError",
        "COMPLETION_CHECK_FAILED",
        "request accepted but the completion check failed",
    ),
    "novel_failure": (
        "SilentBehaviorError",
        "COMPLETION_CHECK_FAILED",
        "request accepted but the completion check failed",
    ),
}

# Schema paths a recovery strategy actually depends on. Only these are compared
# by field-scoped compatibility; changes elsewhere are irrelevant drift.
V3_RELEVANT_PATHS: dict[str, tuple[str, ...]] = {
    "opaque_required": (
        "tools.list_records.parameters.properties.account_id",
        "tools.list_records.parameters.required",
    ),
    "silent_cursor": (
        "tools.list_records.parameters.properties.page_token",
        "tools.list_records.parameters.required",
    ),
    "silent_field": ("tools.list_records.parameters.required",),
    "silent_enum": (
        "tools.list_records.parameters.properties.sort",
        "tools.list_records.parameters.required",
    ),
    "silent_account": (
        "tools.list_records.parameters.properties.account_id",
        "tools.list_records.parameters.required",
    ),
}


def build_v3_tool_schema(family: str, version: str, *, variant: str = "") -> dict[str, Any]:
    """Return a versioned schema. ``variant`` changes only irrelevant fields."""
    if family not in V3_FAILURE_FAMILIES:
        raise ValueError(f"unknown v3 failure family: {family}")
    properties: dict[str, Any] = {
        "tenant": {
            "type": "string",
            "description": "Tenant slug",
        },
        "limit": {"type": "integer", "minimum": 1, "maximum": 10},
    }
    required = ["tenant"]
    if family in {"self_explanatory", "silent_field", "novel_failure"}:
        properties["cursor"] = {
            "type": "string",
            "description": "Opaque cursor from a previous page",
        }
    elif family == "opaque_required":
        properties["account_id"] = {
            "type": "string",
            "description": "Optional billing account identifier",
        }
    elif family == "silent_cursor":
        properties["cursor"] = {
            "type": "string",
            "description": "Legacy cursor parameter",
        }
        properties["page_token"] = {
            "type": "string",
            "description": "Opaque page token",
        }
    elif family == "silent_enum":
        properties["sort"] = {
            "type": "string",
            "enum": ["created"],
            "description": "Sort order",
        }
    elif family == "silent_account":
        properties["account_id"] = {
            "type": "string",
            "description": "Optional billing account identifier",
        }

    if variant == "irrelevant":
        properties["tenant"]["description"] = "Tenant slug; display label only"
    elif variant == "relevant":
        if "page_token" in properties:
            properties["page_token"]["type"] = "integer"
            properties["page_token"]["description"] = "Numeric page token"
        else:
            properties["page_token"] = {"type": "integer", "description": "Numeric page token"}

    return {
        "name": "list_records",
        "description": "List records from the tenant records service.",
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        },
    }


def v3_family_error(family: str) -> Mapping[str, str]:
    error_type, error_code, message = V3_FAMILY_ERRORS.get(
        family, ("SilentBehaviorError", "COMPLETION_CHECK_FAILED", "completion check failed")
    )
    return {"error_type": error_type, "error_code": error_code, "error_message": message}


def _json_path(value: Any, path: str) -> Any:
    current = value
    parts = path.split(".")
    # Relevant paths are written from the environment's perspective
    # (``tools.<tool>.<schema path>``), while ``build_v3_tool_schema`` returns
    # the tool schema itself.
    if len(parts) >= 3 and parts[0] == "tools":
        parts = parts[2:]
    for part in parts:
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def schema_path_hashes(schema: dict[str, Any], paths: tuple[str, ...]) -> dict[str, str]:
    from recap.domain.environments import prefixed_sha256

    return {path: prefixed_sha256(repr(_json_path(schema, path))) for path in paths}


@dataclass
class V3Environment(PaginatedApiEnvironment):
    """Mock environment for the v3 tiers.

    ``schema_variant`` only alters schema fields that are not in
    ``V3_RELEVANT_PATHS`` for the irrelevant case. Silent families return a
    successful-looking payload whose content is incomplete or in the wrong order.
    """

    version: str = "2"
    failure_family: str = "self_explanatory"
    tenant: str = "acme"
    records: list[dict[str, Any]] = field(default_factory=list)
    page_size: int = 3
    schema_variant: str = ""
    _calls: int = 0
    _access_log: list[dict[str, Any]] = field(default_factory=list)
    _cursors: dict[str, int] = field(default_factory=dict)
    _cursor_counter: int = 0

    def __post_init__(self) -> None:
        self._family_explicit = True
        if self.failure_family not in V3_FAILURE_FAMILIES:
            raise ValueError(f"unknown v3 failure family: {self.failure_family}")
        if not self.records:
            self.records = [
                {"id": f"rec-{index:03d}", "tenant": self.tenant, "value": index * 10}
                for index in range(1, 7)
            ]

    @property
    def family_explicit(self) -> bool:
        return True

    @property
    def is_v3(self) -> bool:
        return True

    @property
    def tool_schema(self) -> dict[str, Any]:
        return build_v3_tool_schema(self.failure_family, self.version, variant=self.schema_variant)

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
            task_suite_version="mock-api-suite-v3",
            feature_flags={
                "failure_family": self.failure_family,
                "schema_variant": self.schema_variant,
            },
        )

    def schema_path_hashes(self) -> dict[str, str]:
        paths = V3_RELEVANT_PATHS.get(self.failure_family, ())
        return schema_path_hashes(self.tool_schema, paths)

    def naive_parameters(self) -> dict[str, Any]:
        parameters: dict[str, Any] = {"tenant": self.tenant}
        family = self.failure_family
        if family in {"self_explanatory", "silent_field", "novel_failure"}:
            parameters["cursor"] = "start"
        elif family == "opaque_required":
            # The old client does not send the optional account id.
            pass
        elif family == "silent_cursor":
            # The legacy client sends ``cursor``. The v2 service accepts it and
            # ignores it, so the page never advances.
            parameters["cursor"] = "start"
        elif family == "silent_enum":
            parameters["sort"] = "created"
        elif family == "silent_account":
            pass
        return parameters

    def naive_response_field(self) -> str:
        return "items"

    def expected_records(self) -> list[dict[str, Any]]:
        return list(self.records)

    @property
    def requires_exact_order(self) -> bool:
        return self.failure_family == "silent_enum"

    def provoke_failure(self) -> ToolCallResult:
        return self.list_records(**self.naive_parameters())

    def list_records(  # type: ignore[override]
        self,
        tenant: str,
        cursor: str | None = None,
        page_token: str | None = None,
        account_id: str | None = None,
        sort: str | None = None,
        limit: int | None = None,
        **_ignored: Any,
    ) -> ToolCallResult:
        self._calls += 1
        self._access_log.append(
            {
                "tool": "list_records",
                "tenant": tenant,
                "cursor": cursor,
                "page_token": page_token,
                "account_id": account_id,
                "sort": sort,
            }
        )
        try:
            payload = self._list_records_v3(
                tenant=tenant,
                cursor=cursor,
                page_token=page_token,
                account_id=account_id,
                sort=sort,
                limit=limit,
            )
            return ToolCallResult(ok=True, payload=payload)
        except MockApiError as exc:
            return ToolCallResult(ok=False, payload={}, error=exc.to_payload())

    def _list_records_v3(
        self,
        *,
        tenant: str,
        cursor: str | None,
        page_token: str | None,
        account_id: str | None,
        sort: str | None,
        limit: int | None,
    ) -> dict[str, Any]:
        if tenant != self.tenant:
            raise MockApiError(
                "NotFoundError", "TENANT_NOT_FOUND", f"tenant {tenant!r} not found", 404
            )
        family = self.failure_family
        if family == "opaque_required" and not account_id:
            raise MockApiError(
                "ValidationError",
                "INVALID_REQUEST",
                "request rejected (reference id req-7a1c9e)",
                400,
            )
        if family == "silent_account" and account_id != "acct-north":
            # Accepted, no error. An incomplete result is a silent failure.
            return {"items": [], "count": 0, "total": len(self.records), "has_more": False}

        if family in {"silent_cursor", "novel_failure"} and self.version == "2":
            # ``cursor`` is accepted for compatibility but ignored. Only the
            # token returned in the response advances pagination.
            token = page_token if family == "silent_cursor" else None
            parameter_name = "page_token" if family == "silent_cursor" else "cursor"
        elif family == "silent_field":
            token = cursor
            parameter_name = "cursor"
        else:
            token = cursor
            parameter_name = "cursor"

        offset = 0
        if token not in {None, "start", ""}:
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
        if family == "silent_enum" and self.version == "2" and sort == "created":
            # Accepted value, different behavior: the service quietly falls back
            # to reverse insertion order.
            page = list(reversed(self.records))[offset : offset + size]
        next_offset = offset + size
        payload_key = "data" if family == "silent_field" and self.version == "2" else "items"
        response: dict[str, Any] = {
            payload_key: page,
            "count": len(page),
            "total": len(self.records),
            "has_more": next_offset < len(self.records),
        }
        if family == "silent_field" and self.version == "2":
            # A stale client reads ``items`` and sees an empty list.
            response["items"] = []
        if next_offset < len(self.records):
            self._cursor_counter += 1
            new_token = f"cur_{self.version}_{self._cursor_counter:04d}"
            self._cursors[new_token] = next_offset
            response[parameter_name] = new_token
        return response

    def snapshot(self) -> dict[str, Any]:
        return {
            "environment_id": f"mock-api-{self.failure_family}-v{self.version}",
            "environment_version": self.version,
            "environment_type": "mock_api",
            "failure_family": self.failure_family,
            "tool_names": ["list_records"],
            "record_count": len(self.records),
        }
