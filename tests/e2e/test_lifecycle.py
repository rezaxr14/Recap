"""Full vertical slice: record -> compile -> validate -> promote -> retrieve ->
apply -> outcome -> schema change -> stale rejection -> replacement.

This is the end-to-end acceptance test for the project's core claim. It uses
real PostgreSQL and real Elasticsearch, and every assertion is on persisted
state or a real retrieval decision.
"""

from __future__ import annotations

import uuid

import pytest
from tests.support import context_for, seed_promoted_memory

from recap.domain.enums import MemoryStatus, OutcomeStatus, RetrievalMode
from recap.domain.feedback import OutcomeEvidence
from recap.domain.retrieval import RetrievalRequest
from recap.environments.mock_api import PaginatedApiEnvironment
from recap.environments.reference_agent import ReferenceAgent
from recap.environments.run_recording import record_reference_run
from recap.infrastructure.db.repositories import ensure_workspace
from recap.recall.engine import _to_selected

pytestmark = pytest.mark.e2e


def _refresh(container) -> None:  # type: ignore[no-untyped-def]
    container.es_client.indices.refresh(index=container.index_manager.write_index)


def test_full_lifecycle_from_failure_to_supersession(container, workspace_id: str) -> None:
    # ---------------------------------------------------------- 1. failure run
    v1 = PaginatedApiEnvironment(version="1")
    v1_fingerprint = v1.fingerprint()
    run_id, agent_result = record_reference_run(
        container, v1, workspace_id=workspace_id, task_id="e2e-v1"
    )
    assert agent_result.success is False
    assert agent_result.final_error
    with container.services() as services:
        run = services.runs.get_run(run_id)
        assert run is not None and run.status.value == "failed"
        events = services.runs.list_events(run_id)
        assert any(event.event_type.value == "tool_call_failed" for event in events)
        services.environments.upsert(workspace_id, v1_fingerprint)

        # ------------------------------------------- 2. compile and validate
        candidates = services.repair_compiler.compile(run_id, environment=v1_fingerprint)
        assert len(candidates) == 1
        candidate_id = candidates[0].id
        static = services.repair_compiler.validate(
            candidate_id, environment=v1_fingerprint, validator_name="static"
        )
        assert static.status.value == "passed", static.checks
        sandbox = services.repair_compiler.validate(
            candidate_id, environment=v1_fingerprint, validator_name="mock_sandbox"
        )
        assert sandbox.status.value == "passed", sandbox.checks
        # The sandbox must actually reproduce the failure, not just claim to.
        reproduced = next(
            check for check in sandbox.checks if check["check"] == "failure_reproduced"
        )
        assert reproduced["passed"] is True

        # ----------------------------------------------------- 3. promotion
        refreshed = services.repairs.get(candidate_id)
        assert refreshed is not None
        decision = services.promotion.evaluate(refreshed)
        assert decision.approved, decision.reason
        memory = services.promotion.promote(refreshed)
        assert memory.status is MemoryStatus.ACTIVE
        assert memory.environment_fingerprint == v1_fingerprint.fingerprint
        assert memory.content.task_families == ["api_pagination"]
        memory_id = memory.id

        audit = services.audit.list_for_entity(memory_id)
        assert any(item.action == "promoted" for item in audit)
        stored = services.memories.get_memory(memory_id, workspace_id)
        assert stored is not None and stored.index_generation == 1

    _refresh(container)
    indexed = container.es_client.get(
        index=container.index_manager.read_alias, id=f"{memory_id}:v1"
    )
    assert indexed["_source"]["status"] == "active"
    assert indexed["_source"]["environment_fingerprint"] == v1_fingerprint.fingerprint

    # ----------------------------------------------- 4. retrieve and apply
    with container.services() as services:
        decision = services.recall.retrieve(
            RetrievalRequest(
                context=context_for(workspace_id=workspace_id, environment_version="1"),
                mode=RetrievalMode.OUTCOME_AWARE,
                final_k=3,
            )
        )
        assert decision.status.value == "selected"
        selected = decision.selected_memories[0]
        assert selected.memory_id == memory_id
        stored = services.memories.get_memory(memory_id, workspace_id)
        assert stored is not None
        selected_domain = _to_selected(stored, decision.candidates[0])

    applied = ReferenceAgent().run(PaginatedApiEnvironment(version="1"), memory=selected_domain)
    assert applied.success is True
    assert applied.used_memory_id == memory_id

    with container.services() as services:
        services.outcomes.append(
            OutcomeEvidence(
                workspace_id=workspace_id,
                memory_id=memory_id,
                memory_version=1,
                run_id=run_id,
                outcome=OutcomeStatus.VERIFIED_SUCCESS,
                applied=True,
                environment_fingerprint=v1_fingerprint.fingerprint,
            )
        )
        stats = services.outcomes.context_stats(memory_id, v1_fingerprint.fingerprint)
        assert stats["successes"] == 1

    # ------------------------------------- 5. schema change makes it stale
    v2 = PaginatedApiEnvironment(version="2")
    v2_fingerprint = v2.fingerprint()
    with container.services() as services:
        services.environments.upsert(workspace_id, v2_fingerprint)
        stale_decision = services.recall.retrieve(
            RetrievalRequest(
                context=context_for(
                    workspace_id=workspace_id,
                    environment_version="2",
                    error_code="UNKNOWN_PARAMETER",
                    query="API renamed cursor to page_token and rejected the cursor",
                ),
                mode=RetrievalMode.OUTCOME_AWARE,
                final_k=3,
            )
        )
    assert stale_decision.status.value == "no_safe_memory"
    stale_candidate = next(
        item for item in stale_decision.candidates if item.memory_id == memory_id
    )
    assert stale_candidate.selected is False
    assert any("tool_schema_mismatch" in reason for reason in stale_candidate.rejection_reasons)

    with container.services() as services:
        stored = services.memories.get_memory(memory_id, workspace_id)
        assert stored is not None
        compatibility = services.memory_shift.evaluate_compatibility(
            stored,
            environment_id=v2_fingerprint.environment_id,
            environment_fingerprint=v2_fingerprint.fingerprint,
            tool_schema_fingerprints=v2_fingerprint.tool_schema_hashes,
        )
        assert compatibility.status.value == "incompatible"
        services.memory_shift.on_incompatible_environment(
            stored,
            environment_id=v2_fingerprint.environment_id,
            environment_fingerprint=v2_fingerprint.fingerprint,
            tool_schema_fingerprints=v2_fingerprint.tool_schema_hashes,
        )
        requeued = services.memories.get_memory(memory_id, workspace_id)
        assert requeued is not None
        assert requeued.status is MemoryStatus.SUSPECT
        jobs = services.revalidation.list_queued(limit=50)
        assert any(job.memory_id == memory_id for job in jobs)

    # ------------------------------------------ 6. replacement for the new API
    v2_run_id, v2_result = record_reference_run(
        container, v2, workspace_id=workspace_id, task_id="e2e-v2"
    )
    assert v2_result.success is False
    candidate_id_v2, memory_v2 = seed_promoted_memory(
        container, workspace_id=workspace_id, environment_version="2"
    )
    assert candidate_id_v2
    assert memory_v2.environment_fingerprint == v2_fingerprint.fingerprint
    assert memory_v2.id != memory_id
    _refresh(container)

    with container.services() as services:
        fresh = services.recall.retrieve(
            RetrievalRequest(
                context=context_for(
                    workspace_id=workspace_id,
                    environment_version="2",
                    error_code="UNKNOWN_PARAMETER",
                    query="API renamed cursor to page_token and rejected the cursor",
                ),
                mode=RetrievalMode.OUTCOME_AWARE,
                final_k=3,
            )
        )
        assert fresh.status.value == "selected"
        assert [item.memory_id for item in fresh.selected_memories] == [memory_v2.id]
        assert memory_id not in [item.memory_id for item in fresh.selected_memories]

    v2_decision = fresh.selected_memories[0]
    applied_v2 = ReferenceAgent().run(PaginatedApiEnvironment(version="2"), memory=v2_decision)
    assert applied_v2.success is True
    assert applied_v2.used_memory_id == memory_v2.id


def test_memory_is_not_visible_in_another_workspace(container, workspace_id: str) -> None:
    seed_promoted_memory(container, workspace_id=workspace_id, environment_version="1")
    _refresh(container)
    other = f"other-ws-{uuid.uuid4().hex[:10]}"
    with container.services() as services:
        ensure_workspace(services.session, other)
        decision = services.recall.retrieve(
            RetrievalRequest(
                context=context_for(workspace_id=other, environment_version="1"),
                mode=RetrievalMode.OUTCOME_AWARE,
            )
        )
    assert decision.candidates == []


def test_promotion_creates_a_new_version_and_retires_the_old(container, workspace_id: str) -> None:
    """A new validated content version supersedes, never overwrites, the old one."""
    from tests.support import make_memory

    prior = make_memory(
        container,
        workspace_id=workspace_id,
        title="Recover from INVALID_CURSOR",
        summary="Restart pagination from the first page after the cursor expired.",
        failure_signature="ValidationError|INVALID_CURSOR|list_records",
        failure_code="INVALID_CURSOR",
        environment_version="1",
    )
    assert prior.family_id == prior.id
    _refresh(container)

    candidate_id, replacement = seed_promoted_memory(
        container, workspace_id=workspace_id, environment_version="1"
    )
    assert candidate_id
    assert replacement.id != prior.id
    assert replacement.family_id == prior.family_id
    assert replacement.memory_version == 2
    assert replacement.supersedes_memory_id == prior.id

    with container.services() as services:
        stored_prior = services.memories.get_memory(prior.id, workspace_id)
        assert stored_prior is not None
        assert stored_prior.status is MemoryStatus.RETIRED
        assert stored_prior.status_reason.startswith("superseded by memory")

        versions = services.memories.list_versions(prior.family_id, workspace_id)
        assert [version.memory_version for version in versions] == [1, 2]
        assert versions[1].id == replacement.id

        audit = services.audit.list_for_entity(prior.id)
        assert any(item.action == "superseded" for item in audit)

    _refresh(container)
    with container.services() as services:
        decision = services.recall.retrieve(
            RetrievalRequest(
                context=context_for(workspace_id=workspace_id, environment_version="1"),
                mode=RetrievalMode.OUTCOME_AWARE,
                final_k=3,
            )
        )
    selected_ids = [item.memory_id for item in decision.selected_memories]
    assert selected_ids == [replacement.id]
    assert prior.id not in selected_ids


def test_degradation_marks_memory_suspect_after_failures(container, workspace_id: str) -> None:
    from tests.support import make_memory

    memory = make_memory(
        container,
        workspace_id=workspace_id,
        title="Fragile recovery",
        summary="Restart pagination.",
        failure_signature="ValidationError|INVALID_CURSOR|list_records",
        failure_code="INVALID_CURSOR",
        environment_version="1",
    )
    _refresh(container)
    with container.services() as services:
        stored = services.memories.get_memory(memory.id, workspace_id)
        assert stored is not None
        for _ in range(services.settings.reliability.consecutive_failures_for_suspect):
            stored = services.memories.get_memory(memory.id, workspace_id)
            assert stored is not None
            services.memory_shift.record_outcome(
                stored,
                outcome=OutcomeStatus.VERIFIED_FAILURE,
                applied=True,
                run_id=f"run-{uuid.uuid4().hex[:8]}",
                environment_id=stored.environment_id,
                environment_fingerprint=stored.environment_fingerprint,
            )
        stored = services.memories.get_memory(memory.id, workspace_id)
        assert stored is not None
        assert stored.status is MemoryStatus.SUSPECT
        assert stored.stats.consecutive_failures >= (
            services.settings.reliability.consecutive_failures_for_suspect
        )
    _refresh(container)
    with container.services() as services:
        decision = services.recall.retrieve(
            RetrievalRequest(
                context=context_for(workspace_id=workspace_id, environment_version="1"),
                mode=RetrievalMode.OUTCOME_AWARE,
            )
        )
    assert memory.id not in [item.memory_id for item in decision.selected_memories]
