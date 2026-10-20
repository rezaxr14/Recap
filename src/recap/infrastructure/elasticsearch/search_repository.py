"""Search repository: indexing and retrieval against Elasticsearch.

Indexing is idempotent by ``memory_id:memory_version``. Retrieval returns stable
application-level fields, never raw ES response shapes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from elasticsearch import Elasticsearch
from elasticsearch import exceptions as es_exceptions

from recap.config import ElasticsearchSettings
from recap.domain.memories import ExperienceMemory, build_search_document
from recap.domain.retrieval import RetrievalContext
from recap.errors import SearchUnavailable
from recap.infrastructure.elasticsearch.client import TRANSPORT_ERRORS
from recap.infrastructure.elasticsearch.hybrid_queries import (
    build_lexical_query,
    build_vector_query,
)
from recap.infrastructure.elasticsearch.index_manager import IndexManager
from recap.logging import get_logger
from recap.recall.rrf import RankedList, reciprocal_rank_fusion

logger = get_logger(__name__)


def _allowed_statuses(context: RetrievalContext, allow_suspect: bool) -> list[str]:
    return ["active", "suspect"] if allow_suspect else ["active"]


class ElasticsearchMemoryRepository:
    def __init__(
        self,
        client: Elasticsearch,
        settings: ElasticsearchSettings,
        *,
        allow_suspect: bool = False,
    ) -> None:
        self.client = client
        self.settings = settings
        self.index_manager = IndexManager(client, settings)
        self.allow_suspect = allow_suspect

    def ensure_ready(self) -> None:
        self.index_manager.initialize()

    # ------------------------------------------------------------------ write
    def index_memory(
        self,
        memory: ExperienceMemory,
        embedding: Sequence[float],
        *,
        refresh: bool = True,
    ) -> None:
        if len(embedding) != self.settings.vector_dimension:
            raise SearchUnavailable(
                "embedding dimension does not match index mapping",
                details={
                    "embedding": len(embedding),
                    "mapping": self.settings.vector_dimension,
                },
            )
        document = build_search_document(memory)
        document["content_embedding"] = list(embedding)
        try:
            self.client.index(
                index=self.index_manager.write_index,
                id=f"{memory.id}:v{memory.memory_version}",
                document=document,
                refresh="wait_for" if refresh else False,
            )
        except TRANSPORT_ERRORS as exc:
            raise SearchUnavailable("cannot index memory: Elasticsearch unreachable") from exc
        except es_exceptions.RequestError as exc:
            raise SearchUnavailable(
                "Elasticsearch rejected memory document",
                details={"error": str(exc.info) if hasattr(exc, "info") else str(exc)},
            ) from exc

    def bulk_index(self, memories: Sequence[tuple[ExperienceMemory, Sequence[float]]]) -> None:
        if not memories:
            return
        operations: list[dict[str, Any]] = []
        for memory, embedding in memories:
            document = build_search_document(memory)
            document["content_embedding"] = list(embedding)
            operations.append(
                {
                    "index": {
                        "_index": self.index_manager.write_index,
                        "_id": f"{memory.id}:v{memory.memory_version}",
                    }
                }
            )
            operations.append(document)
        try:
            response = self.client.bulk(operations=operations, refresh="wait_for")
        except TRANSPORT_ERRORS as exc:
            raise SearchUnavailable(
                "cannot bulk index memories: Elasticsearch unreachable"
            ) from exc
        if response.get("errors"):
            failures = [
                item for item in response.get("items", []) if item.get("index", {}).get("error")
            ]
            raise SearchUnavailable(
                "bulk indexing reported failures", details={"failures": failures[:5]}
            )

    def delete_memory(self, memory_id: str, memory_version: int, *, refresh: bool = True) -> None:
        try:
            self.client.delete(
                index=self.index_manager.write_index,
                id=f"{memory_id}:v{memory_version}",
                refresh="wait_for" if refresh else False,
            )
        except es_exceptions.NotFoundError:
            return
        except TRANSPORT_ERRORS as exc:
            raise SearchUnavailable("cannot delete memory: Elasticsearch unreachable") from exc

    # ------------------------------------------------------------------- read
    def search(
        self,
        *,
        context: RetrievalContext,
        mode: str,
        candidate_k: int,
        embedding: Sequence[float] | None = None,
        index_generation: int | None = None,
    ) -> dict[str, Any]:
        """Return raw branch results.

        Shape:
          {
            "hits": {doc_id: source_dict},
            "branches": {source: {"ids": [...], "scores": {...}}},
            "degraded": bool,
            "warnings": [...],
          }
        """
        allowed = _allowed_statuses(context, self.allow_suspect)
        index = self.index_manager.read_alias
        warnings: list[str] = []
        degraded = False
        hits: dict[str, dict[str, Any]] = {}
        branches: dict[str, dict[str, Any]] = {}

        lexical_query = build_lexical_query(
            query=context.query,
            workspace_id=context.workspace_id,
            allowed_statuses=allowed,
            failure_signature=context.failure_signature,
            environment_fingerprint=context.environment_fingerprint or None,
            environment_id=context.environment_id or None,
            task_family=context.task_family or None,
            tool_names=context.tool_names or None,
            failure_code=context.error_code or None,
            size=candidate_k,
        )
        try:
            if mode == "vector":
                if embedding is None:
                    raise SearchUnavailable("vector mode requires an embedding")
                response = self.client.search(
                    index=index,
                    **build_vector_query(
                        embedding=list(embedding),
                        workspace_id=context.workspace_id,
                        allowed_statuses=allowed,
                        environment_fingerprint=context.environment_fingerprint or None,
                        environment_id=context.environment_id or None,
                        task_family=context.task_family or None,
                        tool_names=context.tool_names or None,
                        failure_code=context.error_code or None,
                        size=candidate_k,
                    ),
                )
                branch = _collect_branch(dict(response), "vector", hits)
                branches["vector"] = branch
            elif mode == "bm25":
                response = self.client.search(index=index, **lexical_query)
                branch = _collect_branch(dict(response), "bm25", hits)
                branches["bm25"] = branch
            elif mode == "hybrid_rrf":
                lexical_response = self.client.search(index=index, **lexical_query)
                lexical_branch = _collect_branch(dict(lexical_response), "bm25", hits)
                branches["bm25"] = lexical_branch
                if embedding is None:
                    degraded = True
                    warnings.append("embedding unavailable; returning BM25-only results")
                else:
                    vector_response = self.client.search(
                        index=index,
                        **build_vector_query(
                            embedding=list(embedding),
                            workspace_id=context.workspace_id,
                            allowed_statuses=allowed,
                            environment_fingerprint=context.environment_fingerprint or None,
                            environment_id=context.environment_id or None,
                            task_family=context.task_family or None,
                            tool_names=context.tool_names or None,
                            failure_code=context.error_code or None,
                            size=candidate_k,
                        ),
                    )
                    branches["vector"] = _collect_branch(dict(vector_response), "vector", hits)
            elif mode in {"hybrid_rerank", "outcome_aware"}:
                lexical_response = self.client.search(index=index, **lexical_query)
                branches["bm25"] = _collect_branch(dict(lexical_response), "bm25", hits)
                if embedding is None:
                    degraded = True
                    warnings.append("embedding unavailable; returning BM25-only results")
                else:
                    vector_response = self.client.search(
                        index=index,
                        **build_vector_query(
                            embedding=list(embedding),
                            workspace_id=context.workspace_id,
                            allowed_statuses=allowed,
                            environment_fingerprint=context.environment_fingerprint or None,
                            environment_id=context.environment_id or None,
                            task_family=context.task_family or None,
                            tool_names=context.tool_names or None,
                            failure_code=context.error_code or None,
                            size=candidate_k,
                        ),
                    )
                    branches["vector"] = _collect_branch(dict(vector_response), "vector", hits)
            else:
                raise SearchUnavailable(f"unsupported retrieval mode: {mode}")
        except TRANSPORT_ERRORS as exc:
            raise SearchUnavailable("Elasticsearch unavailable during search") from exc
        except es_exceptions.RequestError as exc:
            raise SearchUnavailable(
                "Elasticsearch rejected the query",
                details={"error": str(exc.info) if hasattr(exc, "info") else str(exc)},
            ) from exc

        return {
            "hits": hits,
            "branches": branches,
            "degraded": degraded,
            "warnings": warnings,
        }

    def fuse(self, branch_payload: dict[str, Any], *, limit: int) -> list[Any]:
        branches = [
            RankedList(
                source=name,
                ids=data["ids"],
                scores=data.get("scores", {}),
            )
            for name, data in branch_payload.get("branches", {}).items()
        ]
        hits = branch_payload.get("hits", {})
        fused = reciprocal_rank_fusion(branches, k=self.settings.rrf_k, limit=None)

        # RRF can produce exact score ties (for example, lexical ranks v1 first
        # while vector ranks v2 first). Resolve those with stable content keys,
        # never Lucene/doc ids, so reruns and merges cannot change the order.
        def _key(hit: Any) -> tuple[float, str, str, str, int, str]:
            source = hits.get(hit.doc_id, {})
            return (
                -hit.score,
                str(source.get("environment_fingerprint") or ""),
                str(source.get("content_hash") or source.get("text_hash") or ""),
                str(source.get("chunk_type") or ""),
                int(source.get("anchor_seq") or 0),
                hit.doc_id,
            )

        fused.sort(key=_key)
        fused = fused[:limit]
        for index, hit in enumerate(fused, start=1):
            hit.ranks["fused"] = index
        return fused

    def count(self) -> int:
        try:
            return int(self.client.count(index=self.index_manager.read_alias)["count"])
        except es_exceptions.NotFoundError:
            return 0
        except TRANSPORT_ERRORS as exc:
            raise SearchUnavailable("cannot count memories: Elasticsearch unreachable") from exc


def _collect_branch(
    response: Mapping[str, Any], source: str, hits: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    ids: list[str] = []
    scores: dict[str, float] = {}
    for hit in response.get("hits", {}).get("hits", []):
        doc_id = hit["_id"]
        ids.append(doc_id)
        scores[doc_id] = float(hit.get("_score") or 0.0)
        hits.setdefault(doc_id, hit.get("_source", {}))

    # ES tie order depends on segment/doc ids and can change after merges. Rank
    # identical scores by stable content keys, never by a random id, so a rerun
    # cannot flip a tie.
    def _key(doc_id: str) -> tuple[float, str, str, str]:
        source = hits.get(doc_id, {})
        return (
            -scores[doc_id],
            str(source.get("environment_fingerprint") or ""),
            str(source.get("content_hash") or ""),
            doc_id,
        )

    ids = sorted(scores, key=_key)
    return {"ids": ids, "scores": scores, "source": source}
