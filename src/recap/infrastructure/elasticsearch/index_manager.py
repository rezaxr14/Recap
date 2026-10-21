"""Index/alias lifecycle for the memory and chunk search indices.

Idempotent initialization: create the versioned index, apply the write alias,
and atomically point the read alias at the new generation. Both indices are
rebuildable from Postgres, so generation changes are safe when the embedding
fingerprint changes.

The chunk index records the embedding fingerprint, dimension, chunker version,
and index generation in its mapping ``_meta``. A mismatch with the running
configuration is a hard error: writing vectors from two fingerprints into one
dense_vector field silently corrupts ranking.
"""

from __future__ import annotations

from typing import Any

from elasticsearch import Elasticsearch
from elasticsearch import exceptions as es_exceptions

from recap.config import ElasticsearchSettings
from recap.errors import ConfigurationError, SearchUnavailable
from recap.infrastructure.elasticsearch.client import TRANSPORT_ERRORS
from recap.infrastructure.elasticsearch.mappings import (
    CHUNK_PROPERTIES,
    chunk_index_body,
    memory_index_body,
)
from recap.logging import get_logger

logger = get_logger(__name__)


class IndexManager:
    def __init__(
        self,
        client: Elasticsearch,
        settings: ElasticsearchSettings,
        *,
        embedding_fingerprint: str = "",
        chunker_version: str = "",
    ) -> None:
        self.client = client
        self.settings = settings
        self.embedding_fingerprint = embedding_fingerprint
        self.chunker_version = chunker_version
        self._active_index: str | None = None

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
    def write_target(self) -> str:
        """Concrete index that receives writes.

        After a successful reindex the write alias is authoritative; the
        configured generation only bootstraps a fresh deployment.
        """
        return self._active_index or self.write_index

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

    def memory_meta(self) -> dict[str, Any]:
        return {
            "embedding_fingerprint": self.embedding_fingerprint,
            "vector_dimension": self.settings.vector_dimension,
            "index_generation": self.settings.index_generation,
        }

    def active_index_name(self) -> str | None:
        """Return the index currently behind the write alias, if any."""
        try:
            aliases = dict(self.client.indices.get_alias(name=self.write_alias))
        except es_exceptions.NotFoundError:
            return None
        except TRANSPORT_ERRORS as exc:
            raise SearchUnavailable("cannot read the memory write alias") from exc
        if len(aliases) != 1:
            raise ConfigurationError(
                "memory write alias must point at exactly one index",
                details={"alias": self.write_alias, "indices": sorted(aliases)},
            )
        return str(next(iter(aliases)))

    def set_active_index(self, index: str) -> None:
        """Record an alias swap performed by the reindex service."""
        self._active_index = index

    def initialize(self) -> dict[str, Any]:
        """Create or discover the active memory index and aliases.

        Once a reindex has moved the aliases, the alias is authoritative. A
        process whose static ``index_generation`` is stale must not create a
        second writable index or re-point the write alias.
        """
        active = self.active_index_name()
        if active is not None:
            self._validate_memory_index(active)
            self._ensure_read_alias(active, self.read_alias)
            self.client.indices.refresh(index=active)
            self._active_index = active
            return {
                "index": active,
                "created": False,
                "read_alias": self.read_alias,
                "write_alias": self.write_alias,
                "vector_dimension": self.settings.vector_dimension,
                "index_generation": self.generation_from_name(active),
                "meta": self.memory_meta(),
            }
        if self.index_exists(self.write_index):
            self._validate_memory_index(self.write_index)
        result = self._initialize(
            index=self.write_index,
            body=memory_index_body(
                dimension=self.settings.vector_dimension,
                meta=self.memory_meta(),
            ),
            read_alias=self.read_alias,
            write_alias=self.write_alias,
        )
        self._active_index = self.write_index
        result["index_generation"] = self.settings.index_generation
        return result

    @staticmethod
    def generation_from_name(index: str) -> int | None:
        marker = "-v"
        if marker not in index:
            return None
        _, _, suffix = index.rpartition(marker)
        try:
            return int(suffix)
        except ValueError:
            return None

    def _validate_memory_index(self, index: str) -> None:
        try:
            mapping = self.client.indices.get_mapping(index=index)[index]["mappings"]
        except (KeyError, es_exceptions.NotFoundError):
            return
        properties = mapping.get("properties") or {}
        embedding = properties.get("content_embedding") or {}
        existing_dimension = embedding.get("dims")
        if existing_dimension is not None and int(existing_dimension) != int(
            self.settings.vector_dimension
        ):
            raise ConfigurationError(
                "memory index dimension does not match embeddings.dimension",
                details={
                    "index": index,
                    "index_dimension": existing_dimension,
                    "configured_dimension": self.settings.vector_dimension,
                    "action": (
                        "run `recap search reindex --to-generation vN` with a new generation "
                        "and the matching embedding dimension"
                    ),
                },
            )
        meta = mapping.get("_meta") or {}
        existing_fingerprint = str(meta.get("embedding_fingerprint") or "")
        if (
            self.embedding_fingerprint
            and existing_fingerprint
            and existing_fingerprint != self.embedding_fingerprint
        ):
            raise ConfigurationError(
                "memory index was built with a different embedding fingerprint",
                details={
                    "index": index,
                    "index_fingerprint": existing_fingerprint,
                    "configured_fingerprint": self.embedding_fingerprint,
                    "action": "run `recap search reindex --to-generation vN`",
                },
            )

    def initialize_chunks(self) -> dict[str, Any]:
        """Create the chunk index and aliases idempotently. Safe to repeat."""
        return self._initialize(
            index=self.chunks_write_index,
            body=chunk_index_body(
                dimension=self.settings.vector_dimension,
                meta=self.chunk_meta(),
            ),
            read_alias=self.chunks_read_alias,
            write_alias=self.chunks_write_alias,
            validate_chunk_meta=True,
        )

    def chunk_meta(self) -> dict[str, Any]:
        return {
            "embedding_fingerprint": self.embedding_fingerprint,
            "vector_dimension": self.settings.vector_dimension,
            "chunker_version": self.chunker_version,
            "chunks_index_generation": self.settings.chunks_index_generation,
        }

    def _initialize(
        self,
        *,
        index: str,
        body: dict[str, Any],
        read_alias: str,
        write_alias: str,
        validate_chunk_meta: bool = False,
    ) -> dict[str, Any]:
        created = False
        try:
            exists = bool(self.client.indices.exists(index=index))
            if not exists:
                self.client.indices.create(index=index, **body)
                created = True
            elif validate_chunk_meta:
                self._validate_chunk_index(index)
            self._ensure_alias(index, write_alias)
            self._ensure_read_alias(index, read_alias)
            self.client.indices.refresh(index=index)
        except ConfigurationError:
            raise
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
            **({"meta": self.chunk_meta()} if validate_chunk_meta else {}),
        }

    def _validate_chunk_index(self, index: str) -> None:
        """Fail fast when an existing chunk index does not match the settings."""
        try:
            mapping = self.client.indices.get_mapping(index=index)[index]["mappings"]
        except (KeyError, es_exceptions.NotFoundError):
            return
        properties = mapping.get("properties") or {}
        embedding = properties.get("content_embedding") or {}
        existing_dimension = embedding.get("dims")
        if existing_dimension is not None and int(existing_dimension) != int(
            self.settings.vector_dimension
        ):
            raise ConfigurationError(
                "chunk index dimension does not match embeddings.dimension",
                details={
                    "index": index,
                    "index_dimension": existing_dimension,
                    "configured_dimension": self.settings.vector_dimension,
                    "action": (
                        "bump elasticsearch.chunks_index_generation and run "
                        "`recap search reconcile --chunks`"
                    ),
                },
            )
        meta = mapping.get("_meta") or {}
        existing_fingerprint = str(meta.get("embedding_fingerprint") or "")
        if self.embedding_fingerprint and existing_fingerprint != self.embedding_fingerprint:
            raise ConfigurationError(
                "chunk index was built with a different embedding fingerprint",
                details={
                    "index": index,
                    "index_fingerprint": existing_fingerprint or "<missing>",
                    "configured_fingerprint": self.embedding_fingerprint,
                    "action": (
                        "bump elasticsearch.chunks_index_generation and run "
                        "`recap search reconcile --chunks`"
                    ),
                },
            )
        existing_generation = meta.get("chunks_index_generation")
        if existing_generation is not None and int(existing_generation) != int(
            self.settings.chunks_index_generation
        ):
            raise ConfigurationError(
                "chunk index generation metadata does not match the configured index",
                details={
                    "index": index,
                    "index_generation": existing_generation,
                    "configured_generation": self.settings.chunks_index_generation,
                },
            )
        expected_fields = set(CHUNK_PROPERTIES) | {"content_embedding"}
        missing = sorted(expected_fields - set(properties))
        if missing:
            raise ConfigurationError(
                "chunk index mapping is missing fields required by this version",
                details={
                    "index": index,
                    "missing_fields": missing,
                    "action": (
                        "bump elasticsearch.chunks_index_generation and run "
                        "`recap search reconcile --chunks`"
                    ),
                },
            )

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
            target = self.active_index_name() or self.write_index
            return {
                "index": target,
                "index_exists": bool(self.client.indices.exists(index=target)),
                "read_alias_exists": bool(self.client.indices.exists_alias(name=self.read_alias)),
                "document_count": int(self.client.count(index=target)["count"]),
                "generation": self.generation_from_name(target),
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
