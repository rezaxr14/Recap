"""Environment and tool fingerprints.

Fingerprints are built from an allowlisted, canonicalized metadata projection.
Secrets are never part of the projection. The canonical hash is stable across
key ordering and insignificant whitespace.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel, Field

FINGERPRINT_VERSION = "fp-v1"


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def prefixed_sha256(value: str) -> str:
    return f"sha256:{sha256_hex(value)}"


def hash_tool_schema(schema: dict[str, Any]) -> str:
    return prefixed_sha256(canonical_json(schema))


class ToolDescriptor(BaseModel):
    name: str
    version: str = "1.0.0"
    schema_hash: str = ""
    side_effect_class: str = "read_only"


class EnvironmentFingerprint(BaseModel):
    """Canonical, non-secret environment identity."""

    fingerprint_version: str = FINGERPRINT_VERSION
    environment_id: str
    environment_type: str = "mock_api"
    environment_version: str = "1.0.0"
    tool_versions: dict[str, str] = Field(default_factory=dict)
    tool_schema_hashes: dict[str, str] = Field(default_factory=dict)
    api_schema_hash: str = ""
    task_suite_version: str = ""
    feature_flags: dict[str, str] = Field(default_factory=dict)
    runtime_image_digest: str = ""

    def canonical_payload(self) -> dict[str, Any]:
        return {
            "fingerprint_version": self.fingerprint_version,
            "environment_id": self.environment_id,
            "environment_type": self.environment_type,
            "environment_version": self.environment_version,
            "tool_versions": dict(sorted(self.tool_versions.items())),
            "tool_schema_hashes": dict(sorted(self.tool_schema_hashes.items())),
            "api_schema_hash": self.api_schema_hash,
            "task_suite_version": self.task_suite_version,
            "feature_flags": dict(sorted(self.feature_flags.items())),
            "runtime_image_digest": self.runtime_image_digest,
        }

    def compute_hash(self) -> str:
        return prefixed_sha256(canonical_json(self.canonical_payload()))

    @property
    def fingerprint(self) -> str:
        return self.compute_hash()


def build_fingerprint(
    *,
    environment_id: str,
    environment_type: str,
    environment_version: str,
    tools: list[ToolDescriptor],
    api_schema: dict[str, Any] | None = None,
    task_suite_version: str = "",
    feature_flags: dict[str, str] | None = None,
) -> EnvironmentFingerprint:
    return EnvironmentFingerprint(
        environment_id=environment_id,
        environment_type=environment_type,
        environment_version=environment_version,
        tool_versions={tool.name: tool.version for tool in tools},
        tool_schema_hashes={tool.name: tool.schema_hash for tool in tools},
        api_schema_hash=prefixed_sha256(canonical_json(api_schema)) if api_schema else "",
        task_suite_version=task_suite_version,
        feature_flags=feature_flags or {},
    )
