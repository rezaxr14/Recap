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
    "memory_id",
    "memory_version",
    "workspace_id",
    "run_id",
    "chunk_type",
    "seq_start",
    "seq_end",
    "event_ids",
    "failure_code",
    "tool_names",
    "task_families",
    "environment_id",
    "environment_fingerprint",
    "chunker_version",
    "embedding_fingerprint",
    "exact_ids",
    "truncated",
    "token_estimate",
]


def _filters(workspace_id: str) -> list[dict[str, Any]]:
    return [
        {"term": {"workspace_id": workspace_id}},
        {"term": {"active_for_retrieval": True}},
    ]


def build_chunk_lexical_query(*, query: str, workspace_id: str, size: int = 30) -> dict[str, Any]:
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
                "filter": _filters(workspace_id),
            }
        },
        "sort": [{"_score": {"order": "desc"}}],
    }


def build_chunk_vector_query(
    *, embedding: list[float], workspace_id: str, size: int = 30, num_candidates: int = 200
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
            "filter": {"bool": {"filter": _filters(workspace_id)}},
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
        self.index_manager = IndexManager(client, settings)

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
            branches[source] = {"ids": ids, "scores": scores}

        lexical_mode = mode in {
            "bm25",
            "bm25_chunk",
            "hybrid_rrf",
            "hybrid_rrf_chunk",
            "hybrid_rerank",
            "hybrid_rerank_chunk",
            "outcome_aware",
            "outcome_aware_chunk",
        }
        vector_mode = mode in {
            "vector",
            "vector_chunk",
            "hybrid_rrf",
            "hybrid_rrf_chunk",
            "hybrid_rerank",
            "hybrid_rerank_chunk",
            "outcome_aware",
            "outcome_aware_chunk",
        }
        if lexical_mode:
            run(
                build_chunk_lexical_query(
                    query=context.query, workspace_id=context.workspace_id, size=candidate_k
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
                    ),
                    "vector",
                )
        degraded = bool(warnings) and not hits
        return {
            "hits": hits,
            "branches": branches,
            "degraded": degraded,
            "warnings": warnings,
        }

    def count(self) -> int:
        try:
            return int(self.client.count(index=self.index_manager.chunks_read_alias)["count"])
        except TRANSPORT_ERRORS as exc:
            raise SearchUnavailable("cannot count chunks: Elasticsearch unreachable") from exc
        except es_exceptions.NotFoundError:
            return 0
