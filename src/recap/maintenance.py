"""Workspace maintenance: reproducible resets and chunk-index reconciliation.

The demo and the benchmark both need to start from a known-empty workspace. A
reset must clear *both* stores; leaving chunk documents (or ``memory_chunks``
rows) behind makes repeated runs non-reproducible because stale chunk documents
compete with the freshly built ones.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from elasticsearch import exceptions as es_exceptions
from sqlalchemy import delete, select

from recap.errors import SearchUnavailable
from recap.infrastructure.elasticsearch.client import TRANSPORT_ERRORS
from recap.logging import get_logger

if TYPE_CHECKING:
    from recap.container import Container

logger = get_logger(__name__)


def _rowcount(result: Any) -> int:
    return int(getattr(result, "rowcount", 0) or 0)


def reset_workspace(container: Container, workspace_ids: Iterable[str]) -> dict[str, Any]:
    """Delete every row and document owned by ``workspace_ids``.

    Postgres is cleared first (including ``memory_chunks``), then both search
    indices. Elasticsearch transport failures are raised rather than swallowed:
    a silent failure here is exactly what made benchmark runs drift.
    """
    from recap.infrastructure.db.models import (
        AuditEventRow,
        CompatibilityRow,
        MemoryChunkRow,
        MemoryRow,
        MemorySnapshotRow,
        OutboxRow,
        OutcomeRow,
        RepairCandidateRow,
        RetrievalDecisionRow,
        RevalidationJobRow,
        RunEventRow,
        RunRow,
        ValidationRow,
    )

    ids = sorted({workspace_id for workspace_id in workspace_ids if workspace_id})
    if not ids:
        return {"workspace_ids": [], "postgres": {}, "elasticsearch": {}}

    postgres: dict[str, int] = {}
    with container.database.session() as session:
        run_ids = list(session.scalars(select(RunRow.id).where(RunRow.workspace_id.in_(ids))).all())
        if run_ids:
            postgres["run_events"] = _rowcount(
                session.execute(delete(RunEventRow).where(RunEventRow.run_id.in_(run_ids)))
            )
        candidate_ids = list(
            session.scalars(
                select(RepairCandidateRow.id).where(RepairCandidateRow.workspace_id.in_(ids))
            ).all()
        )
        if candidate_ids:
            postgres["validations"] = _rowcount(
                session.execute(
                    delete(ValidationRow).where(ValidationRow.candidate_id.in_(candidate_ids))
                )
            )
        for model in (
            MemoryChunkRow,
            RetrievalDecisionRow,
            RepairCandidateRow,
            RevalidationJobRow,
            CompatibilityRow,
            OutcomeRow,
            MemorySnapshotRow,
            MemoryRow,
            AuditEventRow,
            OutboxRow,
            RunRow,
        ):
            postgres[model.__tablename__] = _rowcount(
                session.execute(delete(model).where(model.workspace_id.in_(ids)))
            )

    elasticsearch: dict[str, int] = {}
    for alias_name, index_name in (
        (container.index_manager.read_alias, container.index_manager.write_index),
        (container.index_manager.chunks_read_alias, container.index_manager.chunks_write_index),
    ):
        elasticsearch[index_name] = _delete_workspace_documents(
            container, alias=alias_name, index=index_name, workspace_ids=ids
        )

    logger.info(
        "workspace reset",
        extra={
            "event": "workspace_reset",
            "workspace_ids": ids,
            "postgres": postgres,
            "elasticsearch": elasticsearch,
        },
    )
    return {"workspace_ids": ids, "postgres": postgres, "elasticsearch": elasticsearch}


def _delete_workspace_documents(
    container: Container, *, alias: str, index: str, workspace_ids: list[str]
) -> int:
    query = {"terms": {"workspace_id": workspace_ids}}
    for target in (alias, index):
        try:
            response = container.es_client.delete_by_query(
                index=target,
                query=query,
                refresh=True,
                conflicts="proceed",
            )
        except es_exceptions.NotFoundError:
            # A clean checkout has no index yet; there is nothing to delete.
            continue
        except TRANSPORT_ERRORS as exc:
            raise SearchUnavailable(
                f"cannot reset workspace documents in {target}: Elasticsearch unreachable"
            ) from exc
        deleted = int(response.get("deleted", 0) or 0)
        ignored = int(response.get("version_conflicts", 0) or 0)
        if ignored:
            logger.warning(
                "workspace reset skipped document version conflicts",
                extra={
                    "event": "workspace_reset_conflicts",
                    "index": target,
                    "version_conflicts": ignored,
                },
            )
        return deleted
    logger.info(
        "workspace reset found no index yet",
        extra={"event": "workspace_reset_no_index", "index": index},
    )
    return 0


def prune_orphan_chunks(container: Container, workspace_id: str) -> dict[str, Any]:
    """Remove chunk documents whose chunk or parent memory no longer exists.

    An orphan is an Elasticsearch chunk document that has no ``memory_chunks``
    row, or whose parent memory is gone from Postgres. Chunks of retired
    memories are *not* orphans: lifecycle sync keeps them inactive but
    traceable.
    """
    from recap.infrastructure.db.models import MemoryRow

    chunk_search = container.chunk_search_repository
    if chunk_search is None:
        raise SearchUnavailable("chunk search repository is not configured")

    es_ids = set(chunk_search.list_chunk_ids(workspace_id))
    with container.services() as services:
        stored_chunks = services.chunk_store.list_for_workspace(workspace_id)
        stored_ids = {chunk.chunk_id for chunk in stored_chunks}
        chunk_parents = {chunk.chunk_id: chunk.memory_id for chunk in stored_chunks}
        memory_ids = set(
            services.session.scalars(
                select(MemoryRow.id).where(MemoryRow.workspace_id == workspace_id)
            ).all()
        )

    orphans = sorted(
        chunk_id
        for chunk_id in es_ids
        if chunk_id not in stored_ids or chunk_parents.get(chunk_id) not in memory_ids
    )
    removed = chunk_search.delete_chunks(orphans) if orphans else 0
    if orphans:
        logger.warning(
            "pruned orphan chunk documents",
            extra={
                "event": "chunk_orphans_pruned",
                "workspace_id": workspace_id,
                "orphans": len(orphans),
                "removed": removed,
            },
        )
    return {
        "workspace_id": workspace_id,
        "es_chunks": len(es_ids),
        "postgres_chunks": len(stored_ids),
        "orphans_removed": removed,
        "orphan_ids": orphans,
    }
