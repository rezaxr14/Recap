"""Indexing must leave retryable outbox intent when the embedding role fails."""

from __future__ import annotations

from unittest.mock import Mock

from recap.config import Settings
from recap.container import IndexSynchronizer
from recap.domain.enums import ActionType, SideEffectClass
from recap.domain.memories import ExperienceMemory, MemoryContent
from recap.domain.strategies import RepairStrategy, StrategyAction
from recap.errors import ProviderUnavailable


def _memory() -> ExperienceMemory:
    return ExperienceMemory(
        workspace_id="ws-test",
        content=MemoryContent(
            title="Recover from INVALID_CURSOR",
            summary="Restart pagination from page one.",
            failure_code="INVALID_CURSOR",
            strategy=RepairStrategy(
                summary="Restart pagination.",
                actions=[
                    StrategyAction(
                        action_type=ActionType.REFRESH_PAGINATION_CURSOR,
                        tool_name="list_records",
                        side_effect_class=SideEffectClass.READ_ONLY,
                    )
                ],
            ),
        ),
    )


class FailingEmbeddings:
    def __init__(self, outbox: Mock) -> None:
        self.outbox = outbox

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # The durable intent must exist before the network call, so a crash or
        # outage cannot lose the indexing work.
        assert self.outbox.enqueue.called
        raise ProviderUnavailable("endpoint down", details={"attempts": 2})


def test_embedding_failure_keeps_outbox_intent_and_marks_failed() -> None:
    memory = _memory()
    search = Mock()
    memories = Mock()
    outbox = Mock()

    synchronizer = IndexSynchronizer(
        search_repository=search,
        embeddings=FailingEmbeddings(outbox),
        memory_repository=memories,
        outbox=outbox,
        settings=Settings(),
    )

    assert synchronizer(memory) is False

    outbox.enqueue.assert_called_once()
    assert outbox.enqueue.call_args.kwargs["dedupe_key"] == (
        f"index:{memory.id}:v{memory.memory_version}"
    )
    memories.mark_indexed.assert_called_once_with(memory.id, 0, success=False)
    search.index_memory.assert_not_called()
    assert "endpoint down" in synchronizer.last_error
