"""Memory versioning: family lookup and supersession candidates.

These exercise the SQL repository directly on SQLite. The full promotion path
(create new version -> retire old version) is covered end to end in
``tests/e2e/test_lifecycle.py``.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from recap.domain.enums import ActionType, MemoryStatus, SideEffectClass
from recap.domain.memories import ExperienceMemory, MemoryContent
from recap.domain.predicates import Predicate
from recap.domain.strategies import RepairStrategy, StrategyAction
from recap.infrastructure.db.repositories import SqlMemoryRepository
from recap.infrastructure.db.session import Database, build_engine

WORKSPACE = "ws-versions"


def _repository() -> tuple[SqlMemoryRepository, Database]:
    database = Database(build_engine("sqlite+pysqlite:///:memory:"))
    database.create_all()
    session = database.session_factory()
    return SqlMemoryRepository(session), database


def _memory(
    *,
    summary: str,
    failure_code: str = "INVALID_CURSOR",
    task_families: list[str] | None = None,
    tool_names: list[str] | None = None,
    activated_at: datetime | None = None,
) -> ExperienceMemory:
    families = task_families if task_families is not None else ["api_pagination"]
    tools = tool_names if tool_names is not None else ["list_records"]
    content = MemoryContent(
        title="Recovery",
        summary=summary,
        failure_signature=f"ValidationError|{failure_code}|list_records",
        failure_code=failure_code,
        task_families=families,
        tool_names=tools,
        environment_type="mock_api",
        preconditions=[
            Predicate(field="error_code", operator="in", value=[failure_code]),
        ],
        strategy=RepairStrategy(
            summary=summary,
            actions=[
                StrategyAction(
                    action_type=ActionType.REFRESH_PAGINATION_CURSOR,
                    tool_name="list_records",
                    side_effect_class=SideEffectClass.READ_ONLY,
                )
            ],
        ),
        postconditions=[Predicate(field="records_collected", operator="gte", value=1)],
    )
    return ExperienceMemory(
        workspace_id=WORKSPACE,
        content=content,
        status=MemoryStatus.ACTIVE,
        environment_id="mock-api-v1",
        environment_fingerprint="sha256:env-v1",
        activated_at=activated_at or datetime.now(UTC),
    )


def test_first_version_gets_its_own_family_id() -> None:
    repository, _ = _repository()
    created = repository.create_memory(_memory(summary="first"))
    assert created.family_id == created.id
    assert created.memory_version == 1
    assert created.supersedes_memory_id is None


def test_newer_content_for_the_same_recovery_is_a_supersession_target() -> None:
    repository, _ = _repository()
    older = repository.create_memory(
        _memory(summary="first", activated_at=datetime(2026, 1, 1, tzinfo=UTC))
    )
    newer = repository.create_memory(
        _memory(summary="second", activated_at=datetime(2026, 1, 2, tzinfo=UTC))
    )

    target = repository.find_supersession_target(
        WORKSPACE,
        task_families=["api_pagination"],
        tool_names=["list_records"],
        failure_code="INVALID_CURSOR",
        content_hash=newer.content_hash,
    )

    assert target is not None
    assert target.id == older.id


def test_exact_content_is_never_a_supersession_target() -> None:
    repository, _ = _repository()
    memory = repository.create_memory(_memory(summary="only"))
    target = repository.find_supersession_target(
        WORKSPACE,
        task_families=["api_pagination"],
        tool_names=["list_records"],
        failure_code="INVALID_CURSOR",
        content_hash=memory.content_hash,
    )
    assert target is None


def test_different_failure_code_or_scope_is_not_a_version() -> None:
    repository, _ = _repository()
    repository.create_memory(_memory(summary="first"))

    wrong_code = repository.find_supersession_target(
        WORKSPACE,
        task_families=["api_pagination"],
        tool_names=["list_records"],
        failure_code="RATE_LIMITED",
        content_hash="sha256:different",
    )
    wrong_family = repository.find_supersession_target(
        WORKSPACE,
        task_families=["auth_refresh"],
        tool_names=["list_records"],
        failure_code="INVALID_CURSOR",
        content_hash="sha256:different",
    )
    wrong_tool = repository.find_supersession_target(
        WORKSPACE,
        task_families=["api_pagination"],
        tool_names=["create_record"],
        failure_code="INVALID_CURSOR",
        content_hash="sha256:different",
    )

    assert wrong_code is None
    assert wrong_family is None
    assert wrong_tool is None


def test_retired_memories_are_not_superseded_again() -> None:
    repository, _ = _repository()
    first = repository.create_memory(_memory(summary="first"))
    second = repository.create_memory(_memory(summary="second"))
    retired = first.transition(MemoryStatus.RETIRED, "already replaced")
    repository.update_status(retired)

    target = repository.find_supersession_target(
        WORKSPACE,
        task_families=["api_pagination"],
        tool_names=["list_records"],
        failure_code="INVALID_CURSOR",
        content_hash="sha256:new",
    )

    assert target is not None
    assert target.id == second.id


def test_suspect_memories_remain_supersession_candidates() -> None:
    repository, _ = _repository()
    first = repository.create_memory(_memory(summary="first"))
    suspect = first.transition(MemoryStatus.SUSPECT, "environment drift")
    repository.update_status(suspect)

    target = repository.find_supersession_target(
        WORKSPACE,
        task_families=["api_pagination"],
        tool_names=["list_records"],
        failure_code="INVALID_CURSOR",
        content_hash="sha256:new",
    )

    assert target is not None
    assert target.id == first.id
    assert target.status is MemoryStatus.SUSPECT


def test_versions_are_ordered_and_share_one_family() -> None:
    repository, _ = _repository()
    first = repository.create_memory(_memory(summary="first"))
    second = _memory(summary="second")
    second.family_id = first.family_id
    second.memory_version = 2
    second.supersedes_memory_id = first.id
    repository.create_memory(second)

    versions = repository.list_versions(first.family_id, WORKSPACE)

    assert [item.memory_version for item in versions] == [1, 2]
    assert versions[1].supersedes_memory_id == first.id
    assert versions[0].id != versions[1].id
    # Timestamps must remain usable even when activation happened later.
    assert versions[0].activated_at is not None
    assert versions[1].activated_at is not None
    assert versions[1].activated_at >= versions[0].activated_at - timedelta(seconds=1)
