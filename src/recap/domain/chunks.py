"""Memory chunk domain model.

A chunk is an immutable, traceable unit derived from a memory or from the run
events that back it. Postgres is authoritative; the chunk index is rebuildable.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

from recap.domain.enums import ChunkType
from recap.domain.environments import canonical_json
from recap.domain.runs import utc_now


class MemoryChunk(BaseModel):
    model_config = {"frozen": True}

    chunk_id: str
    workspace_id: str
    memory_id: str
    memory_version: int
    run_id: str | None = None
    chunk_type: ChunkType
    seq_start: int
    seq_end: int
    event_ids: list[str] = Field(default_factory=list)
    text: str
    text_hash: str
    token_estimate: int
    truncated: bool = False
    chunker_version: str = "1"
    redaction_rule_version: str = "redaction-v1"
    failure_code: str = ""
    tool_names: list[str] = Field(default_factory=list)
    tool_versions: list[str] = Field(default_factory=list)
    environment_id: str = ""
    environment_fingerprint: str = ""
    task_families: list[str] = Field(default_factory=list)
    exact_ids: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("event_ids", "exact_ids")
    @classmethod
    def _bounded_unique(cls, value: list[str]) -> list[str]:
        return list(dict.fromkeys(value))[:200]

    @property
    def memory_key(self) -> str:
        return f"{self.memory_id}:v{self.memory_version}"

    def to_search_document(self, *, active_for_retrieval: bool) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "memory_id": self.memory_id,
            "memory_version": self.memory_version,
            "workspace_id": self.workspace_id,
            "run_id": self.run_id or "",
            "chunk_type": str(self.chunk_type),
            "chunk_text": self.text,
            "seq_start": self.seq_start,
            "seq_end": self.seq_end,
            "token_estimate": self.token_estimate,
            "event_ids": self.event_ids,
            "failure_code": self.failure_code,
            "tool_names": self.tool_names,
            "tool_versions": self.tool_versions,
            "environment_id": self.environment_id,
            "environment_fingerprint": self.environment_fingerprint,
            "task_families": self.task_families,
            "chunker_version": self.chunker_version,
            "redaction_rule_version": self.redaction_rule_version,
            "exact_ids": self.exact_ids,
            "truncated": self.truncated,
            "active_for_retrieval": active_for_retrieval,
            "created_at": self.created_at.isoformat(),
        }


def chunk_id_for(
    *,
    workspace_id: str,
    memory_id: str,
    memory_version: int,
    chunk_type: str,
    seq_start: int,
    seq_end: int,
    chunker_version: str,
    text_hash: str,
    run_id: str | None = None,
) -> str:
    # ``run_id`` is an extension to the brief's formula. Without it, identical
    # event text at the same sequence numbers from two evidence runs would
    # collapse into one chunk and lose traceability.
    payload = canonical_json(
        {
            "workspace_id": workspace_id,
            "memory_id": memory_id,
            "memory_version": memory_version,
            "run_id": run_id or "",
            "chunk_type": chunk_type,
            "seq_start": seq_start,
            "seq_end": seq_end,
            "chunker_version": chunker_version,
            "text_hash": text_hash,
        }
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]
