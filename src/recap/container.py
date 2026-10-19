"""Service container wiring concrete adapters to domain services.

One place constructs the object graph so API, CLI, SDK, and tests share exactly
the same semantics.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

from elasticsearch import Elasticsearch

from recap.config import Settings, get_settings
from recap.errors import SearchUnavailable
from recap.infrastructure.db.models import Base
from recap.infrastructure.db.repositories import (
    SqlAuditRepository,
    SqlCompatibilityRepository,
    SqlDecisionRepository,
    SqlEnvironmentRepository,
    SqlMemoryRepository,
    SqlOutcomeRepository,
    SqlOutboxRepository,
    SqlRepairRepository,
    SqlRevalidationRepository,
    SqlRunRepository,
)
from recap.infrastructure.db.session import Database, build_engine
from recap.infrastructure.elasticsearch.client import build_client, check_cluster
from recap.infrastructure.elasticsearch.index_manager import IndexManager
from recap.infrastructure.elasticsearch.search_repository import (
    ElasticsearchMemoryRepository,
)
from recap.infrastructure.embeddings import build_embedding_provider
from recap.memory_shift.service import MemoryShiftService
from recap.recall.engine import RecallEngine
from recap.recorder.recorder import RunRecorder
from recap.repair_compiler.compiler import RepairCompiler
from recap.repair_compiler.promotion import PromotionService
from recap.repair_compiler.validation import build_default_registry


@dataclass
class Container:
    settings: Settings
    database: Database
    es_client: Elasticsearch
    embeddings: object
    index_manager: IndexManager
    search_repository: ElasticsearchMemoryRepository
    validators: object

    @classmethod
    def build(cls, settings: Settings | None = None, *, es_client: Elasticsearch | None = None) -> Container:
        settings = settings or get_settings()
        engine = build_engine(settings.storage.postgres_url)
        database = Database(engine)
        client = es_client or build_client(settings.elasticsearch)
        embeddings = build_embedding_provider(settings.embeddings)
        index_manager = IndexManager(client, settings.elasticsearch)
        search_repository = ElasticsearchMemoryRepository(
            client,
            settings.elasticsearch,
            allow_suspect=settings.recall.allow_suspect_operational_memories,
        )
        return cls(
            settings=settings,
            database=database,
            es_client=client,
            embeddings=embeddings,
            index_manager=index_manager,
            search_repository=search_repository,
            validators=build_default_registry(),
        )

    def migrate(self) -> None:
        Base.metadata.create_all(self.database.engine)

    def init_search(self) -> dict[str, object]:
        return self.index_manager.initialize()

    def health(self) -> dict[str, object]:
        result: dict[str, object] = {"database": "unknown", "elasticsearch": "unknown"}
        try:
            with self.database.session() as session:
                session.execute(__import__("sqlalchemy").text("SELECT 1"))
            result["database"] = "ok"
        except Exception as exc:  # health check boundary
            result["database"] = f"error: {type(exc).__name__}"
        try:
            result["elasticsearch"] = check_cluster(self.es_client)
        except SearchUnavailable as exc:
            result["elasticsearch"] = {"status": "unavailable", "detail": exc.message}
        return result

    @contextmanager
    def unit_of_work(self) -> Iterator["Services"]:
        with self.database.session() as session:
            yield Services(self, session)

    def services(self) -> "ServicesContext":
        return ServicesContext(self)


class Services:
    """Services bound to one database session."""

    def __init__(self, container: Container, session) -> None:  # type: ignore[no-untyped-def]
        self.container = container
        self.session = session
        self.settings = container.settings
        self.runs = SqlRunRepository(session)
        self.memories = SqlMemoryRepository(session)
        self.outcomes = SqlOutcomeRepository(session)
        self.compatibility = SqlCompatibilityRepository(session)
        self.decisions = SqlDecisionRepository(session)
        self.audit = SqlAuditRepository(session)
        self.repairs = SqlRepairRepository(session)
        self.revalidation = SqlRevalidationRepository(session)
        self.environments = SqlEnvironmentRepository(session)
        self.outbox = SqlOutboxRepository(session)
        self.recorder = RunRecorder(self.runs)

        self.index_sync = IndexSynchronizer(
            container.search_repository,
            container.embeddings,
            self.memories,
            self.outbox,
            self.settings,
        )
        self.recall = RecallEngine(
            settings=self.settings,
            search_repository=container.search_repository,
            memory_repository=self.memories,
            compatibility_repository=self.compatibility,
            decision_repository=self.decisions,
            embedding_provider=container.embeddings,
        )
        self.memory_shift = MemoryShiftService(
            settings=self.settings,
            memory_repository=self.memories,
            outcome_repository=self.outcomes,
            compatibility_repository=self.compatibility,
            revalidation_repository=self.revalidation,
            audit_repository=self.audit,
            index_sync=self.index_sync,
        )
        self.repair_compiler = RepairCompiler(
            settings=self.settings,
            run_repository=self.runs,
            repair_repository=self.repairs,
            validators=container.validators,
        )
        self.promotion = PromotionService(
            settings=self.settings,
            memory_repository=self.memories,
            repair_repository=self.repairs,
            audit_repository=self.audit,
            index_sync=self.index_sync,
        )


class ServicesContext:
    def __init__(self, container: Container) -> None:
        self.container = container
        self._context = None

    def __enter__(self) -> Services:
        self._context = self.container.unit_of_work()
        return self._context.__enter__()

    def __exit__(self, exc_type, exc, tb):  # type: ignore[no-untyped-def]
        assert self._context is not None
        return self._context.__exit__(exc_type, exc, tb)


class IndexSynchronizer:
    """Index a memory and record the outbox entry atomically with the DB write.

    If Elasticsearch fails, the outbox row remains pending so a later reconcile
    run can repair the index without losing durable state.
    """

    def __init__(
        self,
        search_repository: ElasticsearchMemoryRepository,
        embeddings: object,
        memory_repository: SqlMemoryRepository,
        outbox: SqlOutboxRepository,
        settings: Settings,
    ) -> None:
        self.search = search_repository
        self.embeddings = embeddings
        self.memories = memory_repository
        self.outbox = outbox
        self.settings = settings

    def __call__(self, memory) -> bool:  # type: ignore[no-untyped-def]
        from recap.domain.memories import build_embedding_text

        text = build_embedding_text(memory)
        embedding = self.embeddings.embed_documents([text])[0]  # type: ignore[attr-defined]
        dedupe = f"index:{memory.id}:v{memory.memory_version}"
        self.outbox.enqueue(
            workspace_id=memory.workspace_id,
            entity_id=memory.id,
            operation="index",
            dedupe_key=dedupe,
        )
        try:
            self.search.index_memory(memory, embedding)
            self.memories.mark_indexed(
                memory.id, self.settings.elasticsearch.index_generation, success=True
            )
            return True
        except SearchUnavailable:
            self.memories.mark_indexed(memory.id, 0, success=False)
            return False
