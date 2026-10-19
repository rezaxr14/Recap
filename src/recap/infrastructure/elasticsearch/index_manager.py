"""Index/alias lifecycle for the memory and chunk search indices.

Idempotent initialization: create the versioned index, apply the write alias,
and atomically point the read alias at the new generation. Both indices are
rebuildable from Postgres, so generation changes are safe when the embedding
fingerprint changes.
"""

from __future__ import annotations

from typing import Any

from elasticsearch import Elasticsearch
from elasticsearch import exceptions as es_exceptions

from recap.config import ElasticsearchSettings
from recap.errors import SearchUnavailable
from recap.infrastructure.elasticsearch.client import TRANSPORT_ERRORS
from recap.infrastructure.elasticsearch.mappings import chunk_index_body, memory_index_body
from recap.logging import get_logger

logger = get_logger(__name__)


class IndexManager:
    def __init__(self, client: Elasticsearch, settings: ElasticsearchSettings) -> None:
        self.client = client
        self.settings = settings

    @property
    def read_alias(self) -> str:
        return self.settings.memories_read_alias

    @property
    def write_alias(self) -> str:
        return self.settings.memories_write_alias

    @property
    def write_index(self) -> str:
        return self.settings.write_index

    @property
    def chunks_read_alias(self) -> str:
        return self.settings.chunks_read_alias

    @property
    def chunks_write_alias(self) -> str:
        return self.settings.chunks_write_alias

    @property
    def chunks_write_index(self) -> str:
        return self.settings.chunks_write_index

    def index_exists(self, name: str | None = None) -> bool:
        try:
            return bool(self.client.indices.exists(index=name or self.write_index))
        except TRANSPORT_ERRORS as exc:
            raise SearchUnavailable("cannot connect to Elasticsearch") from exc

    def initialize(self) -> dict[str, Any]:
        """Create the memory index and aliases idempotently. Safe to repeat."""
        return self._initialize(
            index=self.write_index,
            body=memory_index_body(dimension=self.settings.vector_dimension),
            read_alias=self.read_alias,
            write_alias=self.write_alias,
        )

    def initialize_chunks(self) -> dict[str, Any]:
        """Create the chunk index and aliases idempotently. Safe to repeat."""
        return self._initialize(
            index=self.chunks_write_index,
            body=chunk_index_body(dimension=self.settings.vector_dimension),
            read_alias=self.chunks_read_alias,
            write_alias=self.chunks_write_alias,
        )

    def _initialize(
        self, *, index: str, body: dict[str, Any], read_alias: str, write_alias: str
    ) -> dict[str, Any]:
        created = False
        try:
            if not self.client.indices.exists(index=index):
                self.client.indices.create(index=index, **body)
                created = True
            self._ensure_alias(index, write_alias)
            self._ensure_read_alias(index, read_alias)
            self.client.indices.refresh(index=index)
        except TRANSPORT_ERRORS as exc:
            raise SearchUnavailable(
                "cannot connect to Elasticsearch during index initialization"
            ) from exc
        except es_exceptions.RequestError as exc:
            raise SearchUnavailable(
                "Elasticsearch rejected the index definition",
                details={"error": str(exc.info) if hasattr(exc, "info") else str(exc)},
            ) from exc
        return {
            "index": index,
            "created": created,
            "read_alias": read_alias,
            "write_alias": write_alias,
            "vector_dimension": self.settings.vector_dimension,
        }

    def _ensure_alias(self, index: str, alias: str) -> None:
        actions: list[dict[str, Any]] = [{"add": {"index": index, "alias": alias}}]
        try:
            self.client.indices.update_aliases(actions=actions)
        except es_exceptions.RequestError:
            # Alias already points at this index; re-adding is harmless but be explicit.
            logger.debug("alias update no-op", extra={"event": "alias_update"})

    def _ensure_read_alias(self, index: str, read_alias: str) -> None:
        """Atomically move the read alias to the current write index."""
        existing: dict[str, Any]
        try:
            existing = dict(self.client.indices.get_alias(name=read_alias))
        except es_exceptions.NotFoundError:
            existing = {}
        actions: list[dict[str, Any]] = []
        for existing_index in existing:
            if existing_index != index:
                actions.append({"remove": {"index": existing_index, "alias": read_alias}})
        actions.append({"add": {"index": index, "alias": read_alias}})
        self.client.indices.update_aliases(actions=actions)

    def delete_index(self, name: str | None = None) -> None:
        index = name or self.write_index
        try:
            self.client.indices.delete(index=index, ignore_unavailable=True)
        except TRANSPORT_ERRORS as exc:
            raise SearchUnavailable("cannot connect to Elasticsearch") from exc

    def health(self) -> dict[str, Any]:
        try:
            return {
                "index_exists": bool(self.client.indices.exists(index=self.write_index)),
                "read_alias_exists": bool(self.client.indices.exists_alias(name=self.read_alias)),
                "document_count": int(self.client.count(index=self.write_index)["count"]),
            }
        except TRANSPORT_ERRORS as exc:
            raise SearchUnavailable("cannot connect to Elasticsearch") from exc
        except es_exceptions.NotFoundError:
            return {"index_exists": False, "read_alias_exists": False, "document_count": 0}

    def chunks_health(self) -> dict[str, Any]:
        try:
            return {
                "index_exists": bool(self.client.indices.exists(index=self.chunks_write_index)),
                "read_alias_exists": bool(
                    self.client.indices.exists_alias(name=self.chunks_read_alias)
                ),
                "document_count": int(self.client.count(index=self.chunks_write_index)["count"]),
            }
        except TRANSPORT_ERRORS as exc:
            raise SearchUnavailable("cannot connect to Elasticsearch") from exc
        except es_exceptions.NotFoundError:
            return {"index_exists": False, "read_alias_exists": False, "document_count": 0}
