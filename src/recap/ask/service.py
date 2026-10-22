"""Grounded Q&A with programmatic citation verification.

Retrieval is hybrid for the memory index (BM25 + dense vectors fused with RRF)
and lexical over the bounded run/event history. Memories that are no longer in
the operational index (for example quarantined rows) are supplemented from
Postgres so lifecycle questions can still be answered.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from recap.ask.models import (
    AskAnswer,
    AskRequest,
    Citation,
    CitationVerification,
    RetrievalTrace,
    SourceType,
)
from recap.domain.memories import ExperienceMemory, build_embedding_text
from recap.domain.retrieval import RetrievalContext
from recap.domain.runs import AgentRun, RunEvent
from recap.errors import ConfigurationError
from recap.logging import get_logger
from recap.providers.base import ChatProvider, EmbeddingProvider

if TYPE_CHECKING:
    from recap.container import Services

logger = get_logger(__name__)

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "has",
    "have",
    "how",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}
_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_.:/-]*")
_ID_RE = re.compile(r"\b(?:mem|run|evt|dec|req)_[A-Za-z0-9]+\b")

_SYSTEM_PROMPT = """You answer questions about recorded agent runs and experience memories.
Use only the numbered evidence supplied by the user. Never invent an id, fact, or quote.

Return one JSON object with exactly these fields:
- answer: a short answer, or the literal string "insufficient_evidence".
- citations: an array of objects {source_id, quote}. source_id must be one of the
  supplied ids (for example "memory:mem_..."). quote must be copied verbatim from
  that source's evidence text; do not paraphrase, add ellipses, or join lines.
- confidence: a number from 0 to 1. It is not a probability and is not used as a
  substitute for citation verification.

If the evidence does not directly answer the question, return
"insufficient_evidence" with an empty citations array. Do not fill gaps with
general knowledge."""

_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source_id": {"type": "string"},
                    "quote": {"type": "string"},
                },
                "required": ["source_id", "quote"],
                "additionalProperties": False,
            },
        },
        "confidence": {"type": "number"},
    },
    "required": ["answer", "citations", "confidence"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class _Source:
    type: str
    id: str
    label: str
    text: str
    quote: str
    score: float
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.type}:{self.id}"


def _normalize_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _question_terms(question: str) -> list[str]:
    terms: list[str] = []
    for match in _TOKEN_RE.finditer(question.casefold()):
        term = match.group(0)
        if len(term) >= 3 and term not in _STOPWORDS and term not in terms:
            terms.append(term)
    return terms


def _overlap(text: str, terms: list[str]) -> int:
    lowered = text.casefold()
    return sum(1 for term in terms if term in lowered)


def _best_quote(text: str, terms: list[str], fallback: str = "") -> str:
    """Pick the line with the most query terms; the result is a real substring."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return fallback[:400]
    best = max(lines, key=lambda line: (_overlap(line, terms), len(line)))
    return best[:400]


def _memory_source(memory: ExperienceMemory, terms: list[str], score: float) -> _Source:
    status = memory.status.value
    text = "\n".join(
        [
            build_embedding_text(memory),
            f"Status: {status}",
            f"Status reason: {memory.status_reason}" if memory.status_reason else "",
            f"Memory version: {memory.memory_version}",
            f"Environment id: {memory.environment_id}" if memory.environment_id else "",
        ]
    )
    text = "\n".join(line for line in text.splitlines() if line.strip())
    quote = _best_quote(text, terms, fallback=memory.content.summary)
    return _Source(
        type="memory",
        id=memory.id,
        label=f'Memory "{memory.content.title}"',
        text=text,
        quote=quote,
        score=score,
        metadata={"status": status, "memory_version": str(memory.memory_version)},
    )


def _run_text(run: AgentRun) -> str:
    return "\n".join(
        [
            f"Run: {run.id}",
            f"Status: {run.status.value}",
            f"Failure type: {run.failure_type or 'none'}",
            f"Task: {run.task_id}",
            f"Task family: {run.task_family}",
            f"Environment: {run.environment_id}",
            f"Model: {run.model_provider}/{run.model_name}",
            f"Success criteria: {run.success_criteria}" if run.success_criteria else "",
            f"Tool calls: {run.tool_calls_count}",
        ]
    )


def _run_source(run: AgentRun, terms: list[str], score: float) -> _Source:
    text = "\n".join(line for line in _run_text(run).splitlines() if line.strip())
    quote = _best_quote(text, terms, fallback=f"Status: {run.status.value}")
    return _Source(
        type="run",
        id=run.id,
        label=f"Run {run.id}",
        text=text,
        quote=quote,
        score=score,
        metadata={"status": run.status.value, "failure_type": run.failure_type or "none"},
    )


def _event_text(event: RunEvent) -> str:
    payload = json.dumps(event.payload_redacted, sort_keys=True, ensure_ascii=False)
    lines = [
        f"Event: {event.id}",
        f"Run: {event.run_id}",
        f"Event type: {event.event_type.value}",
        f"Tool: {event.tool_name}" if event.tool_name else "",
        f"Payload: {payload}",
    ]
    return "\n".join(line for line in lines if line.strip())


def _event_source(event: RunEvent, terms: list[str], score: float) -> _Source:
    text = _event_text(event)
    quote = _best_quote(text, terms, fallback=f"Event type: {event.event_type.value}")
    return _Source(
        type="event",
        id=event.id,
        label=f"Event {event.id}",
        text=text,
        quote=quote,
        score=score,
        metadata={"event_type": event.event_type.value, "run_id": event.run_id},
    )


def verify_citations(
    citations: list[Citation], sources: list[_Source]
) -> tuple[list[Citation], CitationVerification]:
    """Require every cited source to exist and every quote to be a substring.

    Whitespace is normalized because JSON transport and source formatting can
    differ; case is preserved so the model cannot silently rewrite a quote.
    """
    by_key = {source.key: source for source in sources}
    checked = 0
    valid: list[Citation] = []
    rejected: list[dict[str, str]] = []
    for citation in citations:
        checked += 1
        key = f"{citation.type}:{citation.id}"
        source = by_key.get(key)
        if source is None:
            rejected.append(
                {"type": citation.type, "id": citation.id, "reason": "source_not_in_context"}
            )
            continue
        quote = _normalize_whitespace(citation.quote)
        if not quote:
            rejected.append({"type": citation.type, "id": citation.id, "reason": "empty_quote"})
            continue
        if quote not in _normalize_whitespace(source.text):
            rejected.append(
                {"type": citation.type, "id": citation.id, "reason": "quote_not_in_source"}
            )
            continue
        valid.append(citation)
    return valid, CitationVerification(
        checked=checked,
        valid=len(valid),
        invalid=len(rejected),
        rejected=rejected,
    )


class AskService:
    def __init__(self, services: Services, *, provider: ChatProvider | None = None) -> None:
        self.services = services
        self.container = services.container
        self.provider = provider if provider is not None else self.container.llm

    def answer(self, request: AskRequest) -> AskAnswer:
        sources, trace = self._retrieve(request)
        if not sources:
            return AskAnswer(
                status="insufficient_evidence",
                answer="insufficient_evidence",
                retrieval=trace,
                notes=["no evidence source matched the question"],
            )
        if request.mode == "llm":
            return self._answer_with_llm(request, sources, trace)
        return self._answer_extractively(request, sources, trace)

    # ------------------------------------------------------------- retrieval
    def _retrieve(self, request: AskRequest) -> tuple[list[_Source], RetrievalTrace]:
        terms = _question_terms(request.question)
        question_lower = request.question.casefold()
        identifiers = {value.casefold() for value in _ID_RE.findall(request.question)}
        trace = RetrievalTrace()

        memory_sources = self._memory_sources(
            request, terms=terms, identifiers=identifiers, trace=trace
        )
        run_sources = self._run_sources(
            request,
            terms=terms,
            question_lower=question_lower,
            identifiers=identifiers,
            trace=trace,
        )
        # Keep one best copy per (type, id) in case SQL and ES return the same memory.
        merged: dict[str, _Source] = {}
        for source in [*memory_sources, *run_sources]:
            current = merged.get(source.key)
            if current is None or source.score > current.score:
                merged[source.key] = source
        ordered = sorted(
            merged.values(),
            key=lambda source: (-source.score, source.type, source.id),
        )[: request.max_sources]
        trace.memory_hits = sum(1 for source in ordered if source.type == "memory")
        trace.run_hits = sum(1 for source in ordered if source.type == "run")
        trace.event_hits = sum(1 for source in ordered if source.type == "event")
        return ordered, trace

    def _memory_sources(
        self,
        request: AskRequest,
        *,
        terms: list[str],
        identifiers: set[str],
        trace: RetrievalTrace,
    ) -> list[_Source]:
        sources: dict[str, _Source] = {}
        embedding: list[float] | None = None
        try:
            embedder = cast(EmbeddingProvider, self.container.embeddings)
            embedding = list(embedder.embed_query(request.question))
        except Exception as exc:  # noqa: BLE001 - BM25-only degradation is explicit
            trace.degraded = True
            trace.warnings.append(
                f"embedding unavailable; memory retrieval was BM25-only: {type(exc).__name__}"
            )
            logger.warning("ask_embedding_unavailable: %s", type(exc).__name__)

        context = RetrievalContext(
            workspace_id=request.workspace_id,
            query=request.question,
            allow_unknown_state=True,
        )
        payload = self.container.search_repository.search(
            context=context,
            mode="hybrid_rrf",
            candidate_k=max(20, request.max_sources * 3),
            embedding=embedding,
        )
        if payload.get("degraded"):
            trace.degraded = True
        trace.warnings.extend(str(item) for item in payload.get("warnings", []))
        fused = self.container.search_repository.fuse(payload, limit=request.max_sources * 2)
        for rank, hit in enumerate(fused, start=1):
            source_doc = payload.get("hits", {}).get(hit.doc_id, {})
            memory_id = str(source_doc.get("memory_id") or hit.doc_id.split(":", 1)[0])
            memory = self.services.memories.get_memory(memory_id, request.workspace_id)
            if memory is None:
                continue
            score = float(_overlap(_memory_source(memory, terms, 0).text, terms))
            if memory.id.casefold() in identifiers:
                score += 20.0
            if score <= 0 and memory.id.casefold() not in identifiers:
                continue
            score += max(0.0, 1.0 - (rank - 1) / 20.0)
            sources[memory.id] = _memory_source(memory, terms, score)

        # Quarantined/retired memories are deliberately absent from the
        # operational index. Supplement the bounded recent history so lifecycle
        # questions can still be answered, and fetch an explicitly named memory.
        for memory in self.services.memories.list_memories(
            request.workspace_id, limit=max(20, request.max_sources * 4)
        ):
            candidate = _memory_source(memory, terms, 0.0)
            score = float(_overlap(candidate.text, terms))
            if score <= 0:
                continue
            existing = sources.get(memory.id)
            if existing is None or score > existing.score:
                sources[memory.id] = _memory_source(memory, terms, score)
        for identifier in identifiers:
            if not identifier.startswith("mem_"):
                continue
            memory = self.services.memories.get_memory(identifier, request.workspace_id)
            if memory is not None:
                sources[memory.id] = _memory_source(memory, terms, 20.0)
        return list(sources.values())

    def _run_sources(
        self,
        request: AskRequest,
        *,
        terms: list[str],
        question_lower: str,
        identifiers: set[str],
        trace: RetrievalTrace,
    ) -> list[_Source]:
        runs = self.services.runs.list_runs(request.workspace_id, limit=request.max_runs)
        # A run named explicitly in the question must not be lost to the row cap.
        for identifier in identifiers:
            if not identifier.startswith("run_"):
                continue
            run = self.services.runs.get_run(identifier)
            if (
                run is not None
                and run.workspace_id == request.workspace_id
                and all(existing.id != run.id for existing in runs)
            ):
                runs.append(run)

        scored: list[tuple[float, AgentRun, str]] = []
        for run in runs:
            text = _run_text(run)
            score = float(_overlap(text, terms))
            if run.id.casefold() in question_lower:
                score += 20.0
            if score <= 0:
                continue
            scored.append((score, run, text))
        scored.sort(key=lambda item: (-item[0], item[1].id))

        sources: list[_Source] = []
        # Only inspect events for runs that will contribute evidence, keeping the
        # number of history reads bounded.
        for score, run, _text in scored[: min(6, request.max_sources)]:
            sources.append(_run_source(run, terms, score))
            events = self.services.runs.list_events(run.id)
            for event in events:
                if not event.payload_redacted:
                    continue
                event_text = _event_text(event)
                event_score = float(_overlap(event_text, terms))
                if event_score <= 0:
                    continue
                # A run is the stable citation for "why did this run fail?";
                # its events are supporting detail and must not outrank it.
                sources.append(_event_source(event, terms, score * 0.5 + event_score / 10.0))
        return sources

    # ------------------------------------------------------------ answering
    def _answer_extractively(
        self, request: AskRequest, sources: list[_Source], trace: RetrievalTrace
    ) -> AskAnswer:
        top = sources[0]
        citations = [Citation(type=_cite_type(top), id=top.id, quote=top.quote)]
        parts: list[str] = []
        if top.type == "memory":
            parts.append(
                f"Memory {top.id} is recorded as {top.metadata.get('status', 'unknown')}. "
                f"{top.quote}"
            )
        elif top.type == "run":
            parts.append(
                f"Run {top.id} finished with status {top.metadata.get('status', 'unknown')} "
                f"and failure type {top.metadata.get('failure_type', 'none')}. {top.quote}"
            )
        else:
            parts.append(f"{top.label} records: {top.quote}")

        supporting = [source for source in sources[1:] if source.key != top.key][:2]
        for source in supporting:
            citations.append(Citation(type=_cite_type(source), id=source.id, quote=source.quote))
            parts.append(f"Additional evidence from {source.label}: {source.quote}")
        valid, verification = verify_citations(citations, sources)
        if verification.invalid or not valid:
            return AskAnswer(
                status="insufficient_evidence",
                answer="insufficient_evidence",
                answerer="extractive",
                verification=verification,
                retrieval=trace,
                notes=["extractive citations failed programmatic verification"],
            )
        return AskAnswer(
            status="answered",
            answer=" ".join(parts),
            citations=valid,
            answerer="extractive",
            verification=verification,
            retrieval=trace,
            notes=[
                "extractive answer: text is assembled from verified source quotes, not an LLM",
            ],
        )

    def _answer_with_llm(
        self, request: AskRequest, sources: list[_Source], trace: RetrievalTrace
    ) -> AskAnswer:
        if self.provider is None:
            raise ConfigurationError(
                "ask mode=llm requires a configured agent LLM provider",
                details={"setting": "RECAP_LLM_PROVIDER / LLM_BASE_URL / LLM_MODEL"},
            )
        user_prompt = self._llm_prompt(request.question, sources)
        result = self.provider.generate_structured(
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            schema=_SCHEMA,
        )
        raw_answer = str(result.get("answer") or "")
        raw_citations = result.get("citations") or []
        citations: list[Citation] = []
        if isinstance(raw_citations, list):
            for item in raw_citations:
                if not isinstance(item, dict):
                    continue
                source_id = str(item.get("source_id") or "")
                quote = str(item.get("quote") or "")
                source_type, separator, source_value = source_id.partition(":")
                if not separator or source_type not in {"memory", "run", "event"}:
                    citations.append(
                        Citation(type="memory", id=source_id, quote=quote)  # rejected below
                    )
                    continue
                citations.append(
                    Citation(type=_cite_type_value(source_type), id=source_value, quote=quote)
                )
        valid, verification = verify_citations(citations, sources)
        notes = ["LLM confidence is not calibrated; citations are verified programmatically"]
        if (
            not raw_answer.strip()
            or raw_answer.strip().casefold() in {"insufficient_evidence", "insufficient evidence"}
            or not valid
            or verification.invalid
        ):
            if verification.invalid:
                notes.append("at least one citation failed verification; downgraded")
            return AskAnswer(
                status="insufficient_evidence",
                answer="insufficient_evidence",
                confidence=0.0,
                answerer="llm",
                provider=self.provider.provider_type,
                model=self.provider.model,
                verification=verification,
                retrieval=trace,
                notes=notes,
            )
        confidence = _confidence(result.get("confidence"))
        return AskAnswer(
            status="answered",
            answer=raw_answer.strip(),
            citations=valid,
            confidence=confidence,
            answerer="llm",
            provider=self.provider.provider_type,
            model=self.provider.model,
            verification=verification,
            retrieval=trace,
            notes=notes,
        )

    @staticmethod
    def _llm_prompt(question: str, sources: list[_Source]) -> str:
        blocks = []
        for index, source in enumerate(sources, start=1):
            text = source.text[:1800]
            blocks.append(
                f"[{index}] source_id={source.key}\nlabel={source.label}\nevidence:\n{text}"
            )
        return (
            f"Question: {question}\n\n"
            "Evidence (only these sources may be cited):\n\n"
            + "\n\n".join(blocks)
            + "\n\nReturn the JSON object now."
        )


def _cite_type(source: _Source) -> SourceType:
    return cast(SourceType, source.type)


def _cite_type_value(value: str) -> SourceType:
    # Kept in one place so the Pydantic Literal and the verifier cannot drift.
    if value in {"memory", "run", "event"}:
        return cast(SourceType, value)
    raise ValueError(f"unsupported citation source type: {value}")


def _confidence(value: object) -> float:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    return min(1.0, max(0.0, number))
