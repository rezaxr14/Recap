"""BFCL v3 function environments for the RECAP stack.

Each environment is one real BFCL function schema at a specific version. A call
is checked against the schema exactly as a strict provider would: unknown
arguments and missing required arguments are execution failures, not noise.

The drift used by the transfer/experiment conditions is the injected rename from
``recap.evals.bfcl.drift_schema``; schemas, queries, and gold arguments are real
BFCL data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from recap.domain.environments import (
    EnvironmentFingerprint,
    ToolDescriptor,
    build_fingerprint,
    canonical_json,
    hash_tool_schema,
    prefixed_sha256,
)
from recap.environments.mock_api import ToolCallResult


def tool_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """OpenAI-style tool document for one BFCL function."""
    return {
        "name": schema.get("name") or "",
        "description": schema.get("description") or "",
        "parameters": schema.get("parameters")
        or {"type": "object", "properties": {}, "required": []},
    }


@dataclass
class BFCLEnvironment:
    """One BFCL function at one schema version."""

    name: str
    schema: dict[str, Any]
    version: str = "2"
    environment_type: str = "bfcl_function"
    failing_args: dict[str, Any] = field(default_factory=dict)
    calls: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.name = self.name or str(self.schema.get("name") or "")
        if not self.name:
            raise ValueError("BFCLEnvironment requires a function name")

    # ---------------------------------------------------------------- metadata
    @property
    def tool_document(self) -> dict[str, Any]:
        return tool_schema(self.schema)

    @property
    def schema_hash(self) -> str:
        return hash_tool_schema(self.tool_document)

    def fingerprint(self) -> EnvironmentFingerprint:
        return build_fingerprint(
            environment_id=self.environment_id,
            environment_type=self.environment_type,
            environment_version=self.version,
            tools=[
                ToolDescriptor(
                    name=self.name,
                    version=self.version,
                    schema_hash=self.schema_hash,
                )
            ],
            api_schema=self.tool_document,
        )

    @property
    def environment_id(self) -> str:
        return f"bfcl:{self.name}:v{self.version}"

    def schema_path_hashes(self) -> dict[str, str]:
        properties = (self.tool_document.get("parameters") or {}).get("properties") or {}
        return {
            f"parameters.properties.{name}": prefixed_sha256(canonical_json(spec))
            for name, spec in properties.items()
        }

    def snapshot(self) -> dict[str, Any]:
        return {
            "environment_id": self.environment_id,
            "environment_version": self.version,
            "tool": self.tool_document,
            "schema_hash": self.schema_hash,
            "schema_path_hashes": self.schema_path_hashes(),
        }

    # ------------------------------------------------------------------- calls
    def call(self, args: dict[str, Any]) -> ToolCallResult:
        """Execute a call by validating it against the real schema."""
        self.calls.append(dict(args))
        params = self.tool_document.get("parameters") or {}
        properties = dict(params.get("properties") or {})
        required = list(params.get("required") or properties)
        unknown = sorted(set(args) - set(properties))
        if unknown:
            expected = sorted(required) or sorted(properties)
            message = f"unknown parameter '{unknown[0]}'; required " + ", ".join(
                f"'{name}'" for name in expected
            )
            return ToolCallResult(
                ok=False,
                payload={},
                error={
                    "error_type": "InvalidRequest",
                    "error_code": "UNKNOWN_PARAMETER",
                    "error_message": message,
                    "http_status": 400,
                },
            )
        missing = [name for name in required if name not in args]
        if missing:
            return ToolCallResult(
                ok=False,
                payload={},
                error={
                    "error_type": "InvalidRequest",
                    "error_code": "MISSING_REQUIRED_PARAMETER",
                    "error_message": "missing required parameter '" + missing[0] + "'",
                    "http_status": 400,
                },
            )
        return ToolCallResult(
            ok=True,
            payload={"function": self.name, "arguments": dict(args), "status": "ok"},
        )

    def replay_failure(self) -> ToolCallResult:
        return self.call(self.failing_args)


def tool_registry(environment: BFCLEnvironment) -> dict[str, Any]:
    return {environment.name: environment.call}
