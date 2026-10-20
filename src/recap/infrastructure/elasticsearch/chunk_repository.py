"""Chunk indexing and retrieval against the chunk search index.

Chunk documents are rebuildable from Postgres. Retrieval returns stable
application-level fields; grouping into memories happens in the recall engine so
that policy (applicability, compatibility, reliability) is still the only
selector in outcome-aware mode.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from elasticsearch import Elasticsearch
from elasticsearch import exceptions as es_exceptions

from recap.config import ElasticsearchSettings
from recap.domain.chunks import MemoryChunk
from recap.domain.retrieval import RetrievalContext
from recap.errors import SearchUnavailable
from recap.infrastructure.elasticsearch.client import TRANSPORT_ERRORS
from recap.infrastructure.elasticsearch.index_manager import IndexManager
from recap.infrastructure.elasticsearch.mappings import CHUNK_LEXICAL_FIELDS
from recap.logging import get_logger

logger = get_logger(__name__)

CHUNK_SOURCE_FIELDS = [
    "chunk_id",
    "text_hash",
    "memory_id",
    "memory_version",
    "workspace_id",
    "run_id",
    "chunk_type",
    "seq_start",
    "seq_end",
    "anchor_seq",
    "event_ids",
    "failure_code",
    "tool_names",
    "task_families",
    "environment_id",
    "environment_fingerprint",
    "chunker_version",
    "chunk_strategy",
    "embedding_fingerprint",
    "exact_ids",
    "truncated",
    "token_estimate",
]


def _filters(
    workspace_id: str,
    embedding_fingerprint: str = "",
    chunk_strategy: str = "",
    environment_fingerprint: str = "",
) -> list[dict[str, Any]]:
    filters: list[dict[str, Any]] = [
        {"term": {"workspace_id": workspace_id}},
        {"term": {"active_for_retrieval": True}},
    ]
    # Vectors from two fingerprints must never share a result set. The index
    # ``_meta`` check catches misconfiguration at startup; this filter keeps a
    # stale document from leaking into results if one ever slips in.
    if embedding_fingerprint:
        filters.append({"term": {"embedding_fingerprint": embedding_fingerprint}})
    if chunk_strategy:
        filters.append({"term": {"chunk_strategy": chunk_strategy}})
    # A chunk is only meaningful inside the environment version its parent
    # memory was compiled for. Unlike card-level retrieval (where an
    # incompatible memory must still be retrieved and explained), chunk
    # retrieval should never mix API versions.
    if environment_fingerprint:
        filters.append({"term": {"environment_fingerprint": environment_fingerprint}})
    return filters


def build_chunk_lexical_query(
    *,
    query: str,
    workspace_id: str,
    size: int = 30,
    embedding_fingerprint: str = "",
    chunk_strategy: str = "",
    environment_fingerprint: str = "",
) -> dict[str, Any]:
    should: list[dict[str, Any]] = []
    if query.strip():
        should.append(
            {
                "multi_match": {
                    "query": query,
                    "fields": [f"{field}^{boost}" for field, boost in CHUNK_LEXICAL_FIELDS.items()],
                    "type": "best_fields",
                    "operator": "or",
                }
            }
        )
    if not should:
        should.append({"match_all": {"boost": 0.1}})
    return {
        "size": size,
        "track_total_hits": False,
        "_source": CHUNK_SOURCE_FIELDS,
        "query": {
            "bool": {
                "should": should,
                "minimum_should_match": 1,
                "filter": _filters(
                    workspace_id,
                    embedding_fingerprint,
                    chunk_strategy,
                    environment_fingerprint,
                ),
            }
        },
        "sort": [{"_score": {"order": "desc"}}],
    }


def build_chunk_vector_query(
    *,
    embedding: list[float],
    workspace_id: str,
    size: int = 30,
    num_candidates: int = 200,
    embedding_fingerprint: str = "",
    chunk_strategy: str = "",
    environment_fingerprint: str = "",
) -> dict[str, Any]:
    return {
        "size": size,
        "track_total_hits": False,
        "_source": CHUNK_SOURCE_FIELDS,
        "knn": {
            "field": "content_embedding",
            "query_vector": embedding,
            "k": size,
            "num_candidates": max(num_candidates, size),
            "filter": {
                "bool": {
                    "filter": _filters(
                        workspace_id,
                        embedding_fingerprint,
                        chunk_strategy,
                        environment_fingerprint,
                    )
                }
            },
        },
    }


class ElasticsearchChunkRepository:
    def __init__(
        self,
        client: Elasticsearch,
        settings: ElasticsearchSettings,
        *,
        embedding_fingerprint: str = "",
        embedding_model_id: str = "",
    ) -> None:
        self.client = client
        self.settings = settings
        self.embedding_fingerprint = embedding_fingerprint
        self.embedding_model_id = embedding_model_id
        self.index_manager = IndexManager(
            client,
            settings,
            embedding_fingerprint=embedding_fingerprint,
        )

    def ensure_ready(self) -> None:
        self.index_manager.initialize_chunks()

    # ------------------------------------------------------------------ write
    def index_chunks(
        self,
        chunks: Sequence[MemoryChunk],
        embeddings: Sequence[Sequence[float]],
        *,
        active_for_retrieval: bool,
        refresh: bool = True,
    ) -> None:
        if len(chunks) != len(embeddings):
            raise SearchUnavailable("chunk/embedding count mismatch")
        if not chunks:
            return
        operations: list[dict[str, Any]] = []
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            if len(embedding) != self.settings.vector_dimension:
                raise SearchUnavailable(
                    "chunk embedding dimension does not match index mapping",
                    details={
                        "embedding": len(embedding),
                        "mapping": self.settings.vector_dimension,
                    },
                )
            document = chunk.to_search_document(active_for_retrieval=active_for_retrieval)
            document["content_embedding"] = list(embedding)
            document["embedding_model_id"] = self.embedding_model_id
            document["embedding_fingerprint"] = self.embedding_fingerprint
            operations.append(
                {
                    "index": {
                        "_index": self.index_manager.chunks_write_index,
                        "_id": chunk.chunk_id,
                    }
                }
            )
            operations.append(document)
        try:
            response = self.client.bulk(
                operations=operations, refresh="wait_for" if refresh else False
            )
        except TRANSPORT_ERRORS as exc:
            raise SearchUnavailable("cannot bulk index chunks: Elasticsearch unreachable") from exc
        if response.get("errors"):
            failures = [
                item for item in response.get("items", []) if item.get("index", {}).get("error")
            ]
            raise SearchUnavailable(
                "chunk bulk indexing reported failures", details={"failures": failures[:5]}
            )

    def set_active_for_memory(
        self, memory_id: str, memory_version: int, *, active: bool, refresh: bool = True
    ) -> int:
        try:
            response = self.client.update_by_query(
                index=self.index_manager.chunks_read_alias,
                query={
                    "bool": {
                        "filter": [
                            {"term": {"memory_id": memory_id}},
                            {"term": {"memory_version": memory_version}},
                        ]
                    }
                },
                script={
                    "source": "ctx._source.active_for_retrieval = params.active",
                    "lang": "painless",
                    "params": {"active": active},
                },
                refresh=refresh,
                conflicts="proceed",
            )
        except es_exceptions.NotFoundError:
            return 0
        except TRANSPORT_ERRORS as exc:
            raise SearchUnavailable(
                "cannot update chunk status: Elasticsearch unreachable"
            ) from exc
        return int(response.get("updated", 0))

    # ------------------------------------------------------------------- read
    def search(
        self,
        *,
        context: RetrievalContext,
        mode: str,
        candidate_k: int,
        embedding: Sequence[float] | None = None,
        chunk_strategy: str = "",
    ) -> dict[str, Any]:
        index = self.index_manager.chunks_read_alias
        warnings: list[str] = []
        hits: dict[str, dict[str, Any]] = {}
        branches: dict[str, dict[str, Any]] = {}

        def run(query: dict[str, Any], source: str) -> None:
            try:
                response = self.client.search(index=index, **query)
            except es_exceptions.NotFoundError:
                warnings.append(f"chunk index {index} does not exist")
                branches[source] = {"ids": [], "scores": {}}
                return
            except TRANSPORT_ERRORS as exc:
                raise SearchUnavailable("cannot search chunks: Elasticsearch unreachable") from exc
            ids: list[str] = []
            scores: dict[str, float] = {}
            for hit in response.get("hits", {}).get("hits", []):
                chunk_id = str(hit.get("_id"))
                ids.append(chunk_id)
                scores[chunk_id] = float(hit.get("_score") or 0.0)
                hits[chunk_id] = dict(hit.get("_source") or {})

            # Deterministic tie-break (see the memory repository): Lucene's
            # internal order is not stable across merges. ``chunk_id`` contains
            # a random memory id, so stable content fields come first.
            def _key(chunk_id: str) -> tuple[float, str, str, str, int, str]:
                source_doc = hits.get(chunk_id, {})
                # ``chunk_id`` embeds a random memory id, so it is the last
                # resort. Environment and text are stable across runs.
                return (
                    -scores[chunk_id],
                    str(source_doc.get("environment_fingerprint") or ""),
                    str(source_doc.get("text_hash") or ""),
                    str(source_doc.get("chunk_type") or ""),
                    int(source_doc.get("anchor_seq") or 0),
                    chunk_id,
                )

            ids = sorted(scores, key=_key)
            branches[source] = {"ids": ids, "scores": scores}

        lexical_mode = mode in {
            "bm25",
            "bm25_chunk",
            "hybrid_rrf",
            "hybrid_rrf_chunk",
            "hybrid_rerank",
            "outcome_aware",
            "outcome_aware_chunk",
        }
        vector_mode = mode in {
            "vector",
            "vector_chunk",
            "hybrid_rrf",
            "hybrid_rrf_chunk",
            "hybrid_rerank",
            "outcome_aware",
            "outcome_aware_chunk",
        }
        if lexical_mode:
            run(
                build_chunk_lexical_query(
                    query=context.query,
                    workspace_id=context.workspace_id,
                    size=candidate_k,
                    embedding_fingerprint=self.embedding_fingerprint,
                    chunk_strategy=chunk_strategy,
                    environment_fingerprint=context.environment_fingerprint,
                ),
                "lexical",
            )
        if vector_mode:
            if embedding is None:
                warnings.append("vector branch skipped: no embedding available")
            else:
                run(
                    build_chunk_vector_query(
                        embedding=list(embedding),
                        workspace_id=context.workspace_id,
                        size=candidate_k,
                        embedding_fingerprint=self.embedding_fingerprint,
                        chunk_strategy=chunk_strategy,
                        environment_fingerprint=context.environment_fingerprint,
                    ),
                    "vector",
                )
        if not hits:
            workspace_total = self.count(workspace_id=context.workspace_id)
            fingerprint_total = self.count(
                workspace_id=context.workspace_id,
                embedding_fingerprint=self.embedding_fingerprint,
            )
            environment_total = fingerprint_total
            if context.environment_fingerprint:
                environment_total = self.count(
                    workspace_id=context.workspace_id,
                    embedding_fingerprint=self.embedding_fingerprint,
                    environment_fingerprint=context.environment_fingerprint,
                )
            strategy_total = environment_total
            if chunk_strategy:
                strategy_total = self.count(
                    workspace_id=context.workspace_id,
                    embedding_fingerprint=self.embedding_fingerprint,
                    chunk_strategy=chunk_strategy,
                    environment_fingerprint=context.environment_fingerprint,
                )
            if workspace_total == 0:
                warnings.append(
                    "chunk_index_empty: no chunk documents for this workspace; "
                    "run `recap search reconcile --chunks`"
                )
            elif fingerprint_total == 0:
                warnings.append(
                    "no chunk documents match the configured embedding fingerprint "
                    f"{self.embedding_fingerprint or '<unset>'}; bump "
                    "chunks_index_generation and run `recap search reconcile --chunks`"
                )
            elif context.environment_fingerprint and environment_total == 0:
                warnings.append(
                    "chunk_environment_empty: no chunk documents for environment "
                    f"{context.environment_fingerprint}; rebuild chunks"
                )
            elif chunk_strategy and strategy_total == 0:
                warnings.append(
                    f"chunk_strategy_empty: no chunk documents for chunk_strategy="
                    f"{chunk_strategy} in this workspace; rebuild chunks"
                )
        degraded = bool(warnings) and not hits
        return {
            "hits": hits,
            "branches": branches,
            "degraded": degraded,
            "warnings": warnings,
        }

    def count(
        self,
        workspace_id: str | None = None,
        *,
        embedding_fingerprint: str = "",
        chunk_strategy: str = "",
        environment_fingerprint: str = "",
    ) -> int:
        try:
            filters: list[dict[str, Any]] = []
            if workspace_id:
                filters.append({"term": {"workspace_id": workspace_id}})
            if embedding_fingerprint:
                filters.append({"term": {"embedding_fingerprint": embedding_fingerprint}})
            if chunk_strategy:
                filters.append({"term": {"chunk_strategy": chunk_strategy}})
            if environment_fingerprint:
                filters.append({"term": {"environment_fingerprint": environment_fingerprint}})
            kwargs: dict[str, Any] = {"index": self.index_manager.chunks_read_alias}
            if filters:
                kwargs["query"] = {"bool": {"filter": filters}}
            return int(self.client.count(**kwargs)["count"])
        except TRANSPORT_ERRORS as exc:
            raise SearchUnavailable("cannot count chunks: Elasticsearch unreachable") from exc
        except es_exceptions.NotFoundError:
            return 0

    def list_chunk_ids(self, workspace_id: str, *, page_size: int = 1000) -> list[str]:
        """Return every chunk document id for one workspace (paginated)."""
        ids: list[str] = []
        search_after: list[Any] | None = None
        while True:
            body: dict[str, Any] = {
                "size": page_size,
                "track_total_hits": False,
                "_source": False,
                "query": {"term": {"workspace_id": workspace_id}},
                "sort": [{"chunk_id": {"order": "asc"}}],
            }
            if search_after is not None:
                body["search_after"] = search_after
            try:
                response = self.client.search(index=self.index_manager.chunks_read_alias, **body)
            except es_exceptions.NotFoundError:
                return ids
            except TRANSPORT_ERRORS as exc:
                raise SearchUnavailable("cannot list chunks: Elasticsearch unreachable") from exc
            hits = response.get("hits", {}).get("hits", [])
            if not hits:
                break
            for hit in hits:
                ids.append(str(hit.get("_id")))
            search_after = list(hits[-1].get("sort") or [])
            if len(hits) < page_size or not search_after:
                break
        return ids

    def delete_chunks(self, chunk_ids: Sequence[str], *, refresh: bool = True) -> int:
        if not chunk_ids:
            return 0
        operations: list[dict[str, Any]] = [
            {
                "delete": {
                    "_index": self.index_manager.chunks_write_index,
                    "_id": chunk_id,
                }
            }
            for chunk_id in chunk_ids
        ]
        try:
            response = self.client.bulk(
                operations=operations, refresh="wait_for" if refresh else False
            )
        except TRANSPORT_ERRORS as exc:
            raise SearchUnavailable("cannot delete chunks: Elasticsearch unreachable") from exc
        removed = 0
        for item in response.get("items", []):
            result = item.get("delete", {})
            if result.get("error"):
                continue
            if result.get("result") in {"deleted", "not_found"}:
                removed += 1
        return removed
