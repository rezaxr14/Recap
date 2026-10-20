"""Service container wiring concrete adapters to domain services.

One place constructs the object graph so API, CLI, SDK, and tests share exactly
the same semantics.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from types import TracebackType

from elasticsearch import Elasticsearch
from sqlalchemy.orm import Session

from recap.chunking.service import ChunkIndexSynchronizer, ChunkService
from recap.config import Settings, get_settings
from recap.domain.enums import MemoryStatus
from recap.domain.memories import ExperienceMemory
from recap.errors import (
    EmbeddingUnavailable,
    ProviderAuthenticationError,
    ProviderResponseError,
    ProviderUnavailable,
    SearchUnavailable,
)
from recap.infrastructure.db.models import Base
from recap.infrastructure.db.repositories import (
    SqlAuditRepository,
    SqlCompatibilityRepository,
    SqlDecisionRepository,
    SqlEnvironmentRepository,
    SqlMemoryChunkRepository,
    SqlMemoryRepository,
    SqlMemorySnapshotRepository,
    SqlOutboxRepository,
    SqlOutcomeRepository,
    SqlRepairRepository,
    SqlRevalidationRepository,
    SqlRunRepository,
)
from recap.infrastructure.db.session import Database, build_engine
from recap.infrastructure.elasticsearch.chunk_repository import ElasticsearchChunkRepository
from recap.infrastructure.elasticsearch.client import build_client, check_cluster
from recap.infrastructure.elasticsearch.index_manager import IndexManager
from recap.infrastructure.elasticsearch.search_repository import (
    ElasticsearchMemoryRepository,
)
from recap.infrastructure.embeddings import build_embedding_provider
from recap.infrastructure.embeddings.cache import CachedEmbeddingProvider, SqlEmbeddingCache
from recap.logging import get_logger
from recap.memory_shift.service import MemoryShiftService
from recap.providers.base import ChatMessage, ChatProvider
from recap.providers.factory import build_chat_provider, provider_report
from recap.recall.engine import RecallEngine
from recap.recall.reranker import build_reranker
from recap.recall.snapshots import MemorySnapshotService
from recap.recorder.recorder import RunRecorder
from recap.repair_compiler.candidate_generation import LLMCandidateGenerator
from recap.repair_compiler.compiler import RepairCompiler
from recap.repair_compiler.promotion import PromotionService
from recap.repair_compiler.validation import ValidatorRegistry, build_default_registry

logger = get_logger(__name__)


@dataclass
class Container:
    settings: Settings
    database: Database
    es_client: Elasticsearch
    embeddings: object
    index_manager: IndexManager
    search_repository: ElasticsearchMemoryRepository
    validators: ValidatorRegistry
    chunk_search_repository: ElasticsearchChunkRepository | None = None
    llm: ChatProvider | None = None
    repair_llm: ChatProvider | None = None
    embedding_cache: SqlEmbeddingCache | None = None

    @classmethod
    def build(
        cls, settings: Settings | None = None, *, es_client: Elasticsearch | None = None
    ) -> Container:
        settings = settings or get_settings()
        engine = build_engine(settings.storage.postgres_url)
        database = Database(engine)
        client = es_client or build_client(settings.elasticsearch)
        base_embeddings = build_embedding_provider(settings.embeddings)
        embedding_cache = SqlEmbeddingCache(database) if settings.embeddings.cache_enabled else None
        embeddings: object = base_embeddings
        if embedding_cache is not None:
            embeddings = CachedEmbeddingProvider(base_embeddings, embedding_cache)
        llm = build_chat_provider("llm", settings.llm)
        repair_llm = build_chat_provider("repair_llm", settings.repair_llm)
        index_manager = IndexManager(
            client,
            settings.elasticsearch,
            embedding_fingerprint=getattr(embeddings, "embedding_fingerprint", ""),
            chunker_version=settings.chunking.version,
        )
        search_repository = ElasticsearchMemoryRepository(
            client,
            settings.elasticsearch,
            allow_suspect=settings.recall.allow_suspect_operational_memories,
        )
        chunk_search_repository = ElasticsearchChunkRepository(
            client,
            settings.elasticsearch,
            embedding_fingerprint=getattr(embeddings, "embedding_fingerprint", ""),
            embedding_model_id=getattr(embeddings, "model_id", settings.embeddings.model_id),
        )
        return cls(
            settings=settings,
            database=database,
            es_client=client,
            embeddings=embeddings,
            index_manager=index_manager,
            search_repository=search_repository,
            chunk_search_repository=chunk_search_repository,
            validators=build_default_registry(),
            llm=llm,
            repair_llm=repair_llm,
            embedding_cache=embedding_cache,
        )

    def migrate(self) -> None:
        Base.metadata.create_all(self.database.engine)

    def init_search(self) -> dict[str, object]:
        # Validate the embedding role against the configured dimension before
        # creating/moving the index. This is what prevents silently mixing
        # incompatible vector spaces.
        probe = self.embeddings.embed_query("recap embedding dimension probe")  # type: ignore[attr-defined]
        actual = len(probe)
        if actual != self.settings.embeddings.dimension:
            from recap.errors import ConfigurationError

            raise ConfigurationError(
                "embedding provider returned an unexpected dimension",
                details={
                    "expected": self.settings.embeddings.dimension,
                    "received": actual,
                    "provider": self.settings.embeddings.provider,
                    "model": self.settings.embeddings.model_id,
                },
            )
        return self.index_manager.initialize()

    def init_search_chunks(self) -> dict[str, object]:
        return self.index_manager.initialize_chunks()

    def provider_status(self) -> dict[str, object]:
        return provider_report(
            llm_settings=self.settings.llm,
            repair_llm_settings=self.settings.repair_llm,
            embedding_settings=self.settings.embeddings,
            llm=self.llm,
            repair_llm=self.repair_llm,
            embeddings=self.embeddings,  # type: ignore[arg-type]
        )

    def provider_health(self, *, live: bool = False) -> dict[str, object]:
        """Probe configured providers. Never prints or returns credentials."""
        results: dict[str, object] = {}
        for role, provider in (("agent_llm", self.llm), ("repair_llm", self.repair_llm)):
            role_result: dict[str, object] = {"status": "disabled"}
            if provider is not None:
                try:
                    role_result = dict(provider.health())
                except Exception as exc:  # health boundary
                    role_result = {
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                if live:
                    try:
                        reply = provider.chat([_probe_message()], max_tokens=16)
                        role_result["live_chat"] = {
                            "status": "ok",
                            "content_length": len(reply.content),
                            "tool_calls": len(reply.tool_calls),
                        }
                    except Exception as exc:  # health boundary
                        role_result["live_chat"] = {
                            "status": "error",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
            results[role] = role_result
        embedding_health = getattr(self.embeddings, "health", None)
        if callable(embedding_health):
            try:
                results["embeddings"] = embedding_health()
            except Exception as exc:  # health boundary
                results["embeddings"] = {
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
        else:
            results["embeddings"] = {"status": "unknown"}
        embedding_result = results["embeddings"]
        if not isinstance(embedding_result, dict):
            embedding_result = {"status": "unknown"}
            results["embeddings"] = embedding_result
        if live:
            try:
                vector = self.embeddings.embed_query("recap health probe")  # type: ignore[attr-defined]
                embedding_result["live_embedding"] = {
                    "status": "ok"
                    if len(vector) == self.settings.embeddings.dimension
                    else "error",
                    "dimension": len(vector),
                    "expected": self.settings.embeddings.dimension,
                }
            except Exception as exc:  # health boundary
                embedding_result["live_embedding"] = {
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                }
        return results

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
        result["providers"] = {
            "semantic_retrieval": bool(
                getattr(
                    getattr(self.embeddings, "capabilities", None), "semantic_embeddings", False
                )
            ),
            "agent_llm": self.settings.llm.provider,
            "repair_llm": self.settings.repair_llm.provider,
            "embeddings": self.settings.embeddings.provider,
        }
        return result

    @contextmanager
    def unit_of_work(self) -> Iterator[Services]:
        with self.database.session() as session:
            yield Services(self, session)

    def services(self) -> ServicesContext:
        return ServicesContext(self)


class Services:
    """Services bound to one database session."""

    def __init__(self, container: Container, session: Session) -> None:
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
        self.snapshots = SqlMemorySnapshotRepository(session)
        self.chunk_store = SqlMemoryChunkRepository(session)
        self.snapshot_service = MemorySnapshotService(
            memory_repository=self.memories,
            snapshot_repository=self.snapshots,
            settings=self.settings,
            embedding_provider=container.embeddings,
        )
        self.recorder = RunRecorder(
            self.runs,
            max_payload_bytes=self.settings.storage.max_event_payload_bytes,
            snapshot_service=self.snapshot_service,
        )

        self.index_sync = IndexSynchronizer(
            container.search_repository,
            container.embeddings,
            self.memories,
            self.outbox,
            self.settings,
            chunk_store=self.chunk_store,
            chunk_search=container.chunk_search_repository,
        )
        self.chunk_sync = ChunkIndexSynchronizer(
            container.chunk_search_repository,
            container.embeddings,
            self.chunk_store,
            self.outbox,
            self.settings,
        )
        self.chunk_service = ChunkService(
            settings=self.settings,
            memory_repository=self.memories,
            run_repository=self.runs,
            repair_repository=self.repairs,
            chunk_repository=self.chunk_store,
            chunk_index_sync=self.chunk_sync,
        )
        self.recall = RecallEngine(
            settings=self.settings,
            search_repository=container.search_repository,
            memory_repository=self.memories,
            compatibility_repository=self.compatibility,
            decision_repository=self.decisions,
            embedding_provider=container.embeddings,
            reranker=build_reranker(self.settings.recall.reranker),
            run_repository=self.runs,
            snapshot_repository=self.snapshots,
            chunk_search_repository=container.chunk_search_repository,
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
        generator = None
        if self.settings.repair_compiler.generator == "llm":
            if container.repair_llm is None:
                from recap.errors import ConfigurationError

                raise ConfigurationError(
                    "repair_compiler.generator=llm requires a configured repair_llm provider"
                )
            generator = LLMCandidateGenerator(container.repair_llm)
        self.repair_compiler = RepairCompiler(
            settings=self.settings,
            run_repository=self.runs,
            repair_repository=self.repairs,
            validators=container.validators,
            generator=generator,
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
        self._context: AbstractContextManager[Services] | None = None

    def __enter__(self) -> Services:
        self._context = self.container.unit_of_work()
        return self._context.__enter__()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool | None:
        assert self._context is not None
        return self._context.__exit__(exc_type, exc, tb)


class IndexSynchronizer:
    """Index a memory and record the outbox entry atomically with the DB write.

    The intent row is written before the embedding call so an embedding or
    Elasticsearch outage leaves retryable work behind instead of losing it. If
    Elasticsearch fails, the outbox row remains pending so a later worker run can
    repair the index without losing durable state.
    """

    def __init__(
        self,
        search_repository: ElasticsearchMemoryRepository,
        embeddings: object,
        memory_repository: SqlMemoryRepository,
        outbox: SqlOutboxRepository,
        settings: Settings,
        *,
        chunk_store: SqlMemoryChunkRepository | None = None,
        chunk_search: ElasticsearchChunkRepository | None = None,
    ) -> None:
        self.search = search_repository
        self.embeddings = embeddings
        self.memories = memory_repository
        self.outbox = outbox
        self.settings = settings
        self.chunk_store = chunk_store
        self.chunk_search = chunk_search
        self.last_error = ""

    def __call__(self, memory: ExperienceMemory) -> bool:
        from recap.domain.memories import build_embedding_text

        text = build_embedding_text(memory)
        dedupe = f"index:{memory.id}:v{memory.memory_version}"
        self.outbox.enqueue(
            workspace_id=memory.workspace_id,
            entity_id=memory.id,
            operation="index",
            dedupe_key=dedupe,
        )
        indexed = False
        try:
            embedding = self.embeddings.embed_documents([text])[0]  # type: ignore[attr-defined]
        except (
            EmbeddingUnavailable,
            ProviderUnavailable,
            ProviderAuthenticationError,
            ProviderResponseError,
        ) as exc:
            self.last_error = exc.message
            self.memories.mark_indexed(memory.id, 0, success=False)
        else:
            try:
                self.search.index_memory(memory, embedding)
                self.memories.mark_indexed(
                    memory.id, self.settings.elasticsearch.index_generation, success=True
                )
                self.outbox.mark_done_by_key(dedupe)
                self.last_error = ""
                indexed = True
            except SearchUnavailable as exc:
                self.last_error = exc.message
                self.memories.mark_indexed(memory.id, 0, success=False)
        # Chunk lifecycle follows the memory status in Postgres regardless of
        # whether the card index write succeeded. A card-index outage must not
        # leave stale chunk documents active, and a chunk-index outage leaves a
        # retryable outbox row rather than only a log line.
        self._sync_chunk_lifecycle(memory)
        return indexed

    def _sync_chunk_lifecycle(self, memory: ExperienceMemory) -> None:
        """Chunk lifecycle follows the memory, but never blocks memory indexing.

        Chunk documents are rebuildable, so a chunk-index outage is a retryable
        outbox problem, not a memory-index failure.
        """

        if self.chunk_store is None:
            return
        active = memory.status is MemoryStatus.ACTIVE
        self.chunk_store.set_active_for_memory(memory.id, memory.memory_version, active=active)
        if self.chunk_search is None:
            return
        dedupe = f"chunk_lifecycle:{memory.id}:v{memory.memory_version}"
        self.outbox.enqueue(
            workspace_id=memory.workspace_id,
            entity_id=memory.id,
            operation="lifecycle",
            dedupe_key=dedupe,
            entity_type="chunk_lifecycle",
        )
        try:
            self.chunk_search.set_active_for_memory(memory.id, memory.memory_version, active=active)
        except SearchUnavailable as exc:
            logger.warning(
                "chunk lifecycle update deferred to the outbox",
                extra={
                    "event": "chunk_lifecycle_deferred",
                    "memory_id": memory.id,
                    "error": exc.message,
                },
            )
            return
        self.outbox.mark_done_by_key(dedupe)


def _probe_message() -> ChatMessage:
    return ChatMessage(role="user", content="Reply with the single word: ok")
