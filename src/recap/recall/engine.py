"""RecallEngine: hybrid retrieval, hard filtering, utility selection, explanation.

Pipeline (see plan section 10.1):
  validate/redact -> build query -> retrieve branches -> fuse -> dedupe/cap ->
  hard filters -> rerank (optional) -> applicability/compatibility ->
  utility features -> select or no_safe_memory -> persist decision.
"""

from __future__ import annotations

import time
from typing import Any

from recap.config import Settings
from recap.domain.enums import (
    CompatibilityStatus,
    RetrievalMode,
    SelectionStatus,
)
from recap.domain.memories import ExperienceMemory, build_embedding_text
from recap.domain.retrieval import (
    CandidateEvaluation,
    RejectedCandidate,
    RetrievalDecision,
    RetrievalRequest,
    SelectedMemory,
)
from recap.domain.snapshots import MemorySnapshot
from recap.errors import NotFoundError, SearchUnavailable, ValidationError
from recap.infrastructure.elasticsearch.search_repository import (
    ElasticsearchMemoryRepository,
)
from recap.logging import get_logger
from recap.recall.applicability import (
    combine_compatibility,
    evaluate_applicability,
    is_operationally_applicable,
)
from recap.recall.eligibility import check_eligibility
from recap.recall.query_builder import build_query_text, redact_context
from recap.recall.utility_policy import (
    POLICY_VERSION,
    build_score_components,
    compute_features,
    should_select,
)

logger = get_logger(__name__)

_CHUNK_MODES = {
    RetrievalMode.BM25_CHUNK,
    RetrievalMode.VECTOR_CHUNK,
    RetrievalMode.HYBRID_RRF_CHUNK,
    RetrievalMode.OUTCOME_AWARE_CHUNK,
}
_VECTOR_MODES = {
    RetrievalMode.VECTOR,
    RetrievalMode.HYBRID_RRF,
    RetrievalMode.HYBRID_RERANK,
    RetrievalMode.OUTCOME_AWARE,
    RetrievalMode.VECTOR_CHUNK,
    RetrievalMode.HYBRID_RRF_CHUNK,
    RetrievalMode.OUTCOME_AWARE_CHUNK,
}


class RecallEngine:
    def __init__(
        self,
        *,
        settings: Settings,
        search_repository: ElasticsearchMemoryRepository,
        memory_repository: Any,
        compatibility_repository: Any,
        decision_repository: Any,
        embedding_provider: Any,
        reranker: Any | None = None,
        run_repository: Any | None = None,
        snapshot_repository: Any | None = None,
        chunk_search_repository: Any | None = None,
    ) -> None:
        self.settings = settings
        self.search = search_repository
        self.chunks = chunk_search_repository
        self.memories = memory_repository
        self.compatibility = compatibility_repository
        self.decisions = decision_repository
        self.embeddings = embedding_provider
        self.reranker = reranker
        self.runs = run_repository
        self.snapshots = snapshot_repository

    def retrieve(self, request: RetrievalRequest, *, persist: bool = True) -> RetrievalDecision:
        started = time.perf_counter()
        stage_latency: dict[str, float] = {}
        context = redact_context(request.context)
        warnings: list[str] = []
        degraded = False
        # Run-level pinning: when retrieval is attached to a run, the run's
        # snapshot bounds the candidate set. This happens before any branch so a
        # memory promoted mid-run cannot enter the candidate pool.
        snapshot, snapshot_id = self._resolve_snapshot(request, context, warnings)
        snapshot_entries = snapshot.entry_map() if snapshot is not None else None
        # Pure retrieval modes are explicit ablations: they rank by relevance only
        # and deliberately skip applicability, compatibility, and utility policy.
        # This is what makes them fair baselines for the outcome-aware mode.
        strict_policy = request.mode in {
            RetrievalMode.OUTCOME_AWARE,
            RetrievalMode.OUTCOME_AWARE_CHUNK,
        }
        chunk_mode = request.mode in _CHUNK_MODES

        # 1-2. Query construction
        t0 = time.perf_counter()
        query_text = build_query_text(context)
        stage_latency["query_build"] = _ms(t0)

        # 3. Retrieval branches
        t0 = time.perf_counter()
        embedding = None
        if request.mode in _VECTOR_MODES:
            try:
                embedding = self.embeddings.embed_query(query_text)
            except Exception as exc:  # provider boundary
                logger.warning("embedding failed: %s", exc, extra={"event": "embedding_failed"})
                warnings.append("embedding provider unavailable; retrieval degraded to BM25")
                degraded = True
                if request.mode in {RetrievalMode.VECTOR, RetrievalMode.VECTOR_CHUNK}:
                    decision = self._degraded_decision(request, query_text, warnings, snapshot_id)
                    self._persist(decision, persist)
                    return decision
        if chunk_mode:
            if self.chunks is None:
                raise SearchUnavailable(
                    "chunk retrieval requires the chunk search repository",
                    details={"mode": str(request.mode)},
                )
            branch_payload = self.chunks.search(
                context=context,
                mode=str(request.mode),
                candidate_k=request.candidate_k,
                embedding=embedding,
            )
        else:
            branch_payload = self.search.search(
                context=context,
                mode=str(request.mode),
                candidate_k=request.candidate_k,
                embedding=embedding,
            )
        degraded = degraded or branch_payload.get("degraded", False)
        warnings.extend(branch_payload.get("warnings", []))
        branches: dict[str, dict[str, Any]] = branch_payload["branches"]
        stage_latency["search"] = _ms(t0)

        # 4-5. Fuse, dedupe, cap
        t0 = time.perf_counter()
        if len(branches) > 1:
            fused = self.search.fuse(branches, limit=self.settings.recall.max_candidates)
            fused_order = [hit.doc_id for hit in fused]
            fused_scores = {hit.doc_id: hit.score for hit in fused}
            branch_ranks = {hit.doc_id: hit.ranks for hit in fused}
        else:
            only: dict[str, Any] = dict(next(iter(branches.values()), {"ids": [], "scores": {}}))
            fused_order = list(only.get("ids", []))[: self.settings.recall.max_candidates]
            fused_scores = {
                doc_id: only.get("scores", {}).get(doc_id, 0.0) for doc_id in fused_order
            }
            branch_ranks = {
                doc_id: {next(iter(branches), "bm25"): index}
                for index, doc_id in enumerate(fused_order, start=1)
            }
        stage_latency["fuse"] = _ms(t0)

        # 4b. Chunk grouping: fuse at chunk level, then group into memories.
        # The group score is the best supporting chunk; no chunk score can add
        # beyond what the policy already evaluates.
        supporting: dict[str, dict[str, list[str]]] = {}
        if chunk_mode:
            chunk_hits: dict[str, dict[str, Any]] = branch_payload["hits"]
            grouped: dict[str, dict[str, Any]] = {}
            for chunk_id in fused_order:
                source = chunk_hits.get(chunk_id, {})
                memory_id = str(source.get("memory_id") or "")
                if not memory_id:
                    continue
                version = int(source.get("memory_version") or 1)
                doc_id = f"{memory_id}:v{version}"
                bucket = grouped.setdefault(
                    doc_id,
                    {
                        "score": fused_scores.get(chunk_id, 0.0),
                        "ranks": {},
                        "chunks": [],
                        "events": [],
                    },
                )
                bucket["score"] = max(bucket["score"], fused_scores.get(chunk_id, 0.0))
                for branch, rank in branch_ranks.get(chunk_id, {}).items():
                    current = bucket["ranks"].get(branch)
                    bucket["ranks"][branch] = rank if current is None else min(current, rank)
                if chunk_id not in bucket["chunks"]:
                    bucket["chunks"].append(chunk_id)
                for event_id in source.get("event_ids") or []:
                    if event_id not in bucket["events"]:
                        bucket["events"].append(event_id)
            ordered = sorted(
                grouped.items(),
                key=lambda item: (
                    -item[1]["score"],
                    min(item[1]["ranks"].values()) if item[1]["ranks"] else 10**9,
                ),
            )
            fused_order = [doc_id for doc_id, _ in ordered]
            fused_scores = {doc_id: bucket["score"] for doc_id, bucket in ordered}
            branch_ranks = {doc_id: bucket["ranks"] for doc_id, bucket in ordered}
            supporting = {
                doc_id: {
                    "chunk_ids": bucket["chunks"][:20],
                    "event_ids": bucket["events"][:50],
                }
                for doc_id, bucket in ordered
            }

        # 5b. Optional reranking (HYBRID_RERANK only). Reranking changes order,
        # never safety: eligibility/applicability/compatibility run afterwards.
        rerank_applied = False
        relevance_by_doc: dict[str, float] = {}
        if (
            request.mode is RetrievalMode.HYBRID_RERANK
            and self.reranker is not None
            and fused_order
        ):
            t0 = time.perf_counter()
            rerank_ids: list[str] = []
            rerank_documents: list[str] = []
            for doc_id in fused_order:
                memory_id, _, _ = doc_id.partition(":v")
                memory = self.memories.get_memory(memory_id, context.workspace_id)
                rerank_ids.append(doc_id)
                rerank_documents.append(build_embedding_text(memory) if memory else "")
            try:
                order = self.reranker.rerank(query_text, rerank_documents, len(rerank_ids))
                fused_order = [rerank_ids[index] for index, _ in order]
                relevance_by_doc = {
                    rerank_ids[index]: max(0.0, min(1.0, score)) for index, score in order
                }
                for rank, doc_id in enumerate(fused_order, start=1):
                    branch_ranks.setdefault(doc_id, {})["rerank"] = rank
                rerank_applied = True
            except Exception as exc:  # reranker boundary: degrade to fusion order
                logger.warning("reranker failed: %s", exc, extra={"event": "reranker_failed"})
                warnings.append("reranker unavailable; using fusion order")
                degraded = True
            stage_latency["rerank"] = _ms(t0)

        # 6-9. Eligibility, applicability, compatibility, utility
        t0 = time.perf_counter()
        evaluations: list[CandidateEvaluation] = []
        for rank, doc_id in enumerate(fused_order, start=1):
            memory_id, _, version_text = doc_id.partition(":v")
            version = int(version_text or 1)
            entry = snapshot_entries.get(memory_id) if snapshot_entries is not None else None
            if snapshot_entries is not None:
                if entry is None:
                    evaluations.append(
                        CandidateEvaluation(
                            memory_id=memory_id,
                            memory_version=version,
                            title="",
                            status="unavailable",
                            fused_rank=rank,
                            hard_filter_passed=False,
                            rejection_reasons=["not_in_memory_snapshot"],
                        )
                    )
                    continue
                if entry.memory_version != version:
                    evaluations.append(
                        CandidateEvaluation(
                            memory_id=memory_id,
                            memory_version=version,
                            title="",
                            status="unavailable",
                            fused_rank=rank,
                            hard_filter_passed=False,
                            rejection_reasons=["snapshot_version_mismatch"],
                        )
                    )
                    continue
            memory = self.memories.get_memory(memory_id, context.workspace_id)
            if memory is None:
                evaluations.append(
                    CandidateEvaluation(
                        memory_id=memory_id,
                        memory_version=version,
                        title="",
                        status="missing",
                        fused_rank=rank,
                        hard_filter_passed=False,
                        rejection_reasons=["memory_not_found_in_source_of_truth"],
                    )
                )
                continue
            if entry is not None and entry.content_hash != memory.content_hash:
                evaluations.append(
                    CandidateEvaluation(
                        memory_id=memory_id,
                        memory_version=version,
                        title=memory.content.title,
                        status=str(memory.status),
                        fused_rank=rank,
                        hard_filter_passed=False,
                        rejection_reasons=["snapshot_content_mismatch"],
                    )
                )
                continue
            stored_compat = None
            stored = self.compatibility.get(memory.id, context.environment_fingerprint)
            if stored is not None:
                stored_compat = stored.status
            compatibility, compatibility_reason = combine_compatibility(
                memory, context, stored_compat
            )
            eligibility = check_eligibility(
                memory,
                context,
                settings=self.settings.recall,
                mode=request.mode,
                allow_unvalidated=request.allow_unvalidated,
                compatibility=compatibility if strict_policy else None,
                apply_compatibility_filters=strict_policy,
            )
            applicability = evaluate_applicability(memory, context)
            relevance = (
                relevance_by_doc.get(doc_id, 0.0)
                if rerank_applied
                else _normalize_fused(fused_scores.get(doc_id, 0.0))
            )
            features = compute_features(
                memory=memory,
                context=context,
                relevance=relevance,
                applicability=applicability.result,
                satisfaction_ratio=applicability.satisfaction_ratio,
                unknown_ratio=(
                    applicability.unknown_count / len(applicability.evaluations)
                    if applicability.evaluations
                    else 1.0
                ),
                compatibility=compatibility,
                reliability=self.settings.reliability,
            )
            reasons = list(eligibility.reasons)
            if strict_policy:
                if (
                    request.require_compatible_environment
                    and compatibility
                    in {CompatibilityStatus.INCOMPATIBLE, CompatibilityStatus.SUSPECT}
                    and "suspect_compatibility" not in reasons
                ):
                    reasons.append(f"compatibility_{compatibility}")
                if not is_operationally_applicable(applicability):
                    reasons.append(f"preconditions_{applicability.result}")

            if strict_policy:
                selected, select_reason = should_select(
                    features,
                    applicability=applicability.result,
                    compatibility=compatibility,
                    settings=self.settings.recall,
                )
                if not eligibility.eligible:
                    selected = False
                    select_reason = eligibility.reasons[0]
            else:
                # Retrieval-only baseline: rank by fused relevance, no policy.
                selected = eligibility.eligible
                select_reason = "selected" if selected else eligibility.reasons[0]
                features.total = relevance
            evaluations.append(
                CandidateEvaluation(
                    memory_id=memory.id,
                    memory_version=memory.memory_version,
                    title=memory.content.title,
                    status=str(memory.status),
                    branch_ranks=branch_ranks.get(doc_id, {}),
                    branch_scores={
                        source: data.get("scores", {}).get(doc_id, 0.0)
                        for source, data in branches.items()
                    },
                    fused_score=fused_scores.get(doc_id, 0.0),
                    fused_rank=rank,
                    hard_filter_passed=eligibility.eligible,
                    rejection_reasons=reasons,
                    applicability_result=applicability.result,
                    applicability_reason=applicability.reason,
                    compatibility_status=compatibility,
                    compatibility_reason=compatibility_reason,
                    features=features,
                    score_components=build_score_components(features),
                    final_score=features.total,
                    selected=selected and select_reason == "selected",
                    supporting_chunk_ids=supporting.get(doc_id, {}).get("chunk_ids", []),
                    supporting_event_ids=supporting.get(doc_id, {}).get("event_ids", []),
                    retrieval_granule="chunk" if chunk_mode else "memory_card",
                )
            )
        stage_latency["evaluate"] = _ms(t0)

        # 10. Select with diversity cap and final_k
        selected_memories: list[SelectedMemory] = []
        rejected: list[RejectedCandidate] = []
        strategy_counts: dict[str, int] = {}
        eligible_candidates = [item for item in evaluations if item.selected]
        eligible_candidates.sort(key=lambda item: (-item.final_score, item.fused_rank))
        for candidate in evaluations:
            if not candidate.selected:
                rejected.append(
                    RejectedCandidate(
                        memory_id=candidate.memory_id,
                        memory_version=candidate.memory_version,
                        reason=_primary_reason(candidate),
                        detail="; ".join(candidate.rejection_reasons),
                    )
                )
        for candidate in eligible_candidates:
            if len(selected_memories) >= request.final_k:
                rejected.append(
                    RejectedCandidate(
                        memory_id=candidate.memory_id,
                        memory_version=candidate.memory_version,
                        reason="final_k_exceeded",
                    )
                )
                continue
            memory = self.memories.get_memory(candidate.memory_id, context.workspace_id)
            if memory is None:  # pragma: no cover - raced deletion
                continue
            signature = memory.content.strategy.action_signature
            if strategy_counts.get(signature, 0) >= self.settings.recall.max_per_strategy:
                rejected.append(
                    RejectedCandidate(
                        memory_id=candidate.memory_id,
                        memory_version=candidate.memory_version,
                        reason="duplicate_strategy",
                    )
                )
                candidate.selected = False
                continue
            strategy_counts[signature] = strategy_counts.get(signature, 0) + 1
            selected_memories.append(_to_selected(memory, candidate))

        status = SelectionStatus.SELECTED if selected_memories else SelectionStatus.NO_SAFE_MEMORY
        reason = (
            f"selected {len(selected_memories)} memory(ies)"
            if selected_memories
            else _no_safe_reason(evaluations)
        )
        total_ms = _ms(started)
        stage_latency["total"] = total_ms
        decision = RetrievalDecision(
            request_id=request.request_id,
            run_id=request.run_id,
            memory_snapshot_id=snapshot_id,
            workspace_id=context.workspace_id,
            query=query_text,
            mode=request.mode,
            status=status,
            reason=reason,
            policy_version=POLICY_VERSION,
            index_generation=self.settings.elasticsearch.index_generation,
            embedding_model_id=self.embeddings.model_id,
            embedding_fingerprint=getattr(self.embeddings, "embedding_fingerprint", ""),
            retrieval_granule="chunk" if chunk_mode else "memory_card",
            candidates=evaluations,
            selected_memories=selected_memories,
            rejected_candidates=rejected,
            retrieval_degraded=degraded,
            warnings=warnings,
            stage_latency_ms=stage_latency,
            total_latency_ms=total_ms,
            config_snapshot={
                "candidate_k": request.candidate_k,
                "final_k": request.final_k,
                "mode": str(request.mode),
                "policy_version": POLICY_VERSION,
                "rrf_k": self.settings.elasticsearch.rrf_k,
                "embedding_model_id": self.embeddings.model_id,
                "no_safe_memory_threshold": self.settings.recall.no_safe_memory_threshold,
                "rerank_applied": rerank_applied,
                "reranker": getattr(self.reranker, "name", "none"),
                "memory_snapshot_id": snapshot_id,
                "memory_snapshot_entries": snapshot.entry_count if snapshot else None,
                "retrieval_granule": "chunk" if chunk_mode else "memory_card",
                "chunker_version": self.settings.chunking.version,
                "chunk_index_generation": self.settings.elasticsearch.chunks_index_generation,
                "embedding_fingerprint": getattr(self.embeddings, "embedding_fingerprint", ""),
            },
        )
        self._persist(decision, persist)
        return decision

    def _degraded_decision(
        self,
        request: RetrievalRequest,
        query_text: str,
        warnings: list[str],
        snapshot_id: str | None = None,
    ) -> RetrievalDecision:
        decision = RetrievalDecision(
            request_id=request.request_id,
            run_id=request.run_id,
            memory_snapshot_id=snapshot_id,
            workspace_id=request.context.workspace_id,
            query=query_text,
            mode=request.mode,
            status=SelectionStatus.DEGRADED,
            reason="required embedding provider unavailable for vector-only mode",
            retrieval_degraded=True,
            warnings=warnings,
        )
        return decision

    def _persist(self, decision: RetrievalDecision, persist: bool) -> None:
        if persist:
            self.decisions.save(decision)

    def _resolve_snapshot(
        self,
        request: RetrievalRequest,
        context: Any,
        warnings: list[str],
    ) -> tuple[MemorySnapshot | None, str | None]:
        if request.run_id is None:
            return None, None
        if self.runs is None or self.snapshots is None:
            raise ValidationError("run-scoped retrieval requires run and snapshot repositories")
        run = self.runs.get_run(request.run_id)
        if run is None:
            raise NotFoundError(f"run {request.run_id} not found")
        if run.workspace_id != context.workspace_id:
            raise ValidationError(f"run {request.run_id} belongs to a different workspace")
        if run.memory_snapshot_id is None:
            warnings.append("run has no memory snapshot; retrieval is not pinned to a frozen set")
            return None, None
        snapshot = self.snapshots.get(run.memory_snapshot_id, context.workspace_id)
        if snapshot is None:
            raise NotFoundError(
                f"memory snapshot {run.memory_snapshot_id} for run {run.id} not found"
            )
        return snapshot, snapshot.id

    def search_only(self, request: RetrievalRequest) -> RetrievalDecision:
        """Return ranked candidates without applying operational selection policy."""
        return self.retrieve(request, persist=True)


def _to_selected(memory: ExperienceMemory, candidate: CandidateEvaluation) -> SelectedMemory:
    content = memory.content
    return SelectedMemory(
        memory_id=memory.id,
        memory_version=memory.memory_version,
        title=content.title,
        summary=content.summary,
        strategy_summary=content.strategy.summary,
        actions=[action.model_dump(mode="json") for action in content.strategy.actions],
        preconditions=[predicate.model_dump(mode="json") for predicate in content.preconditions],
        postconditions=[predicate.model_dump(mode="json") for predicate in content.postconditions],
        limitations=content.known_limitations,
        evidence_run_ids=memory.evidence_run_ids,
        evidence_trace_ids=memory.evidence_trace_ids,
        score=candidate.final_score,
        features=candidate.features,
        compatibility_status=candidate.compatibility_status,
        content_hash=memory.content_hash,
        supporting_chunk_ids=candidate.supporting_chunk_ids,
        supporting_event_ids=candidate.supporting_event_ids,
        retrieval_granule=candidate.retrieval_granule,
    )


def _primary_reason(candidate: CandidateEvaluation) -> str:
    if candidate.rejection_reasons:
        return candidate.rejection_reasons[0]
    if not candidate.selected:
        return "not_selected"
    return "selected"


def _no_safe_reason(evaluations: list[CandidateEvaluation]) -> str:
    if not evaluations:
        return "no candidates matched the query"
    counts: dict[str, int] = {}
    for candidate in evaluations:
        for reason in candidate.rejection_reasons:
            counts[reason] = counts.get(reason, 0) + 1
    if not counts:
        return "no candidate cleared the selection policy"
    top = sorted(counts.items(), key=lambda item: -item[1])[:3]
    return "no_safe_memory: " + ", ".join(f"{reason}={count}" for reason, count in top)


def _normalize_fused(score: float) -> float:
    # RRF scores are small (typically < 0.05 for k=60). Scale for interpretability
    # without changing ordering.
    return min(1.0, score * 25.0)


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 3)


def search_error_message(exc: SearchUnavailable) -> str:
    return f"retrieval unavailable: {exc.message}"
