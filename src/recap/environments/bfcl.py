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


# BFCL schemas use Python type names that are valid in the dataset but not in
# JSON Schema. Strict OpenAI-compatible servers reject ``"type": "dict"`` with
# ``Invalid discriminator value. Expected 'object'``.
_BFCL_TYPE_MAP = {
    "dict": "object",
    "float": "number",
    "tuple": "array",
}

_JSON_TYPES = ("string", "number", "integer", "boolean", "array", "object", "null")


def openai_parameters(parameters: dict[str, Any] | None) -> dict[str, Any]:
    """Convert BFCL parameter types to JSON Schema for live tool calls.

    The conversion is recursive and preserves every declared property, the
    required list, enums, defaults and descriptions. ``dict`` becomes
    ``object``, ``float`` becomes ``number`` and ``tuple`` becomes ``array``.
    ``any`` has no JSON Schema type, so it advertises the full JSON type set
    instead of silently narrowing the model's choice.

    This is deliberately separate from :func:`tool_schema`: fingerprints must
    keep hashing the dataset's declared schema, while the model needs a valid
    JSON Schema. Both describe the same parameter names and required set.
    """
    if not isinstance(parameters, dict):
        return {"type": "object", "properties": {}, "required": [], "additionalProperties": False}
    converted: dict[str, Any] = _convert(parameters)
    if converted.get("type") != "object":
        converted = {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        }
    return converted


def openai_tool_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """OpenAI tool document with JSON-Schema-valid parameter types."""
    document = tool_schema(schema)
    document["parameters"] = openai_parameters(document.get("parameters"))
    return document


def _convert(node: Any) -> Any:
    if isinstance(node, list):
        return [_convert(item) for item in node]
    if not isinstance(node, dict):
        return node

    converted: dict[str, Any] = {}
    for key, value in node.items():
        if key in {"properties", "patternProperties"} and isinstance(value, dict):
            converted[key] = {name: _convert(spec) for name, spec in value.items()}
        elif key in {"items", "additionalProperties", "not"}:
            converted[key] = _convert(value)
        elif key in {"anyOf", "oneOf", "allOf", "prefixItems"} and isinstance(value, list):
            converted[key] = [_convert(item) for item in value]
        elif key == "required" and isinstance(value, list):
            converted[key] = list(value)
        else:
            converted[key] = value

    declared = converted.get("type")
    if isinstance(declared, str):
        declared = _BFCL_TYPE_MAP.get(declared, declared)
        if declared == "any":
            converted["type"] = list(_JSON_TYPES)
        else:
            converted["type"] = declared
    elif isinstance(declared, list):
        converted["type"] = [
            _BFCL_TYPE_MAP.get(item, item) if isinstance(item, str) else item for item in declared
        ]

    if converted.get("type") == "object":
        properties = converted.get("properties")
        if properties is None:
            converted["properties"] = {}
        required = converted.get("required")
        if not isinstance(required, list):
            converted["required"] = []
        converted.setdefault("additionalProperties", False)
    if converted.get("type") == "array" and "items" not in converted:
        converted["items"] = {}
    return converted


def _matches_type(spec: dict[str, Any], value: Any) -> bool:
    declared = spec.get("type")
    if declared in (None, "any"):
        return True
    if isinstance(declared, list):
        return any(_matches_type({**spec, "type": item}, value) for item in declared)
    if declared == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if declared in {"float", "number"}:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if declared == "string":
        return isinstance(value, str)
    if declared == "boolean":
        return isinstance(value, bool)
    if declared in {"array", "tuple"}:
        return isinstance(value, list)
    if declared in {"dict", "object"}:
        return isinstance(value, dict)
    return True


def _type_name(spec: dict[str, Any]) -> str:
    declared = spec.get("type")
    if isinstance(declared, list):
        return " or ".join(str(item) for item in declared)
    return str(declared or "any")


def argument_errors(spec: dict[str, Any], value: Any, path: str) -> list[tuple[str, str]]:
    """Return ``(error_code, message)`` for a declared argument value.

    This is deliberately small and JSON-Schema-shaped: it checks the type
    BFCL declares, enums, and nested object/array items. It does not claim
    full JSON Schema conformance and never consults a gold answer.
    """
    if not _matches_type(spec, value):
        return [
            (
                "INVALID_ARGUMENT_TYPE",
                f"argument '{path}' must be {_type_name(spec)}, got {type(value).__name__}",
            )
        ]
    errors: list[tuple[str, str]] = []
    allowed = spec.get("enum")
    if isinstance(allowed, list) and value not in allowed:
        errors.append(
            (
                "INVALID_ENUM_VALUE",
                f"argument '{path}' must be one of {allowed}, got {value!r}",
            )
        )
    declared = spec.get("type")
    if declared in {"array", "tuple"}:
        items = spec.get("items")
        if isinstance(items, dict):
            for index, item in enumerate(value):
                errors.extend(argument_errors(items, item, f"{path}[{index}]"))
        elif isinstance(items, list):
            for index, (sub, item) in enumerate(zip(items, value, strict=False)):
                if isinstance(sub, dict):
                    errors.extend(argument_errors(sub, item, f"{path}[{index}]"))
    if declared in {"dict", "object"}:
        properties = spec.get("properties")
        if isinstance(properties, dict):
            for name, sub in properties.items():
                if name in value and isinstance(sub, dict):
                    errors.extend(argument_errors(sub, value[name], f"{path}.{name}"))
            if spec.get("additionalProperties") is False:
                for name in sorted(set(value) - set(properties)):
                    errors.append(
                        (
                            "UNKNOWN_PARAMETER",
                            f"unknown parameter '{name}' under '{path}'",
                        )
                    )
    required = spec.get("required")
    if declared in {"dict", "object"} and isinstance(required, list):
        for name in required:
            if name not in value:
                errors.append(
                    (
                        "MISSING_REQUIRED_PARAMETER",
                        f"missing required parameter '{path}.{name}'",
                    )
                )
    return errors


@dataclass
class BFCLEnvironment:
    """One BFCL function at one schema version."""

    name: str
    schema: dict[str, Any]
    version: str = "2"
    environment_type: str = "bfcl_function"
    failing_args: dict[str, Any] = field(default_factory=dict)
    calls: list[dict[str, Any]] = field(default_factory=list)
    # ``strict`` validates declared value types and enums in addition to
    # parameter names. The injected-rename experiment only needs names, and
    # keeping it off there preserves those reports byte-for-byte. Live model
    # runs turn it on: a string "5" where the schema declares integer is a
    # real execution failure, not a success.
    strict: bool = False

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
        if self.strict:
            for name in sorted(args):
                spec = properties.get(name)
                errors = argument_errors(spec, args[name], name) if isinstance(spec, dict) else []
                if errors:
                    code, message = errors[0]
                    return ToolCallResult(
                        ok=False,
                        payload={},
                        error={
                            "error_type": "InvalidRequest",
                            "error_code": code,
                            "error_message": message,
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
