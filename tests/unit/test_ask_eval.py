"""CPU-only checks for the Ask mini-eval fixtures and scoring.

The live evaluation itself is a script (it needs Postgres and Elasticsearch),
but the question shape and the metric arithmetic are pure and must not drift.
"""

from __future__ import annotations

from scripts.ask_eval import (
    ANSWERED,
    INSUFFICIENT,
    EvalCase,
    _category_summary,
    _render_markdown,
    aggregate,
    build_cases,
    evaluate_case,
    safe_command,
)

from recap.ask.models import AskAnswer, Citation, CitationVerification, RetrievalTrace


def _ids() -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    tokens = (
        "zephyr",
        "quartz",
        "nimbus",
        "tundra",
        "cobalt",
        "ember",
        "fjord",
        "garnet",
        "harbor",
        "indigo",
    )
    memory_ids = {token: f"mem_{token}" for token in tokens}
    run_ids = {token: f"run_{token}" for token in tokens}
    statuses = {
        **{token: "active" for token in tokens[:7]},
        "garnet": "suspect",
        "harbor": "quarantined",
        "indigo": "retired",
    }
    return memory_ids, run_ids, statuses


def _answer(*, status: str, answer: str, source: str | None, quote: str = "quote") -> AskAnswer:
    citations = []
    verification = CitationVerification()
    if source is not None:
        citations = [Citation(type="memory", id=source, quote=quote)]
        verification = CitationVerification(checked=1, valid=1, invalid=0)
    return AskAnswer(
        status=status,  # type: ignore[arg-type]
        answer=answer,
        citations=citations,
        verification=verification,
        retrieval=RetrievalTrace(),
    )


def test_report_command_never_contains_an_api_key() -> None:
    rendered = safe_command(["--mode", "llm", "--api-key", "secret-value"])
    assert "secret-value" not in rendered
    assert "--api-key ***" in rendered
    assert "secret-value" not in safe_command(["--api-key=secret-value"])


def test_mini_eval_has_thirty_answerable_and_ten_unanswerable_cases() -> None:
    memory_ids, run_ids, statuses = _ids()
    cases = build_cases(memory_ids, run_ids, statuses)

    assert len(cases) == 40
    assert sum(case.expected_status == ANSWERED for case in cases) == 30
    assert sum(case.category == "unanswerable" for case in cases) == 10
    assert len({case.case_id for case in cases}) == 40
    for case in cases:
        if case.expected_status == ANSWERED:
            assert case.gold_sources
            assert case.answer_tokens
        else:
            assert not case.gold_sources
            assert not case.answer_tokens


def test_evaluate_case_scores_citation_precision_and_recall() -> None:
    case = EvalCase(
        case_id="memory-zephyr",
        category="memory_content",
        question="zephyr recovery",
        expected_status=ANSWERED,
        gold_sources=("memory:mem_zephyr",),
        answer_tokens=("zephyr",),
    )
    good = evaluate_case(
        case,
        _answer(status=ANSWERED, answer="zephyr recovery", source="mem_zephyr"),
        latency_ms=10.0,
    )
    assert good["metrics"] == {
        "status_correct": True,
        "citation_precision": 1.0,
        "citation_recall": 1.0,
        "answer_correct": True,
        "quote_valid": True,
    }

    wrong = evaluate_case(
        case,
        _answer(status=ANSWERED, answer="zephyr recovery", source="mem_quartz"),
        latency_ms=10.0,
    )
    assert wrong["metrics"]["citation_precision"] == 0.0
    assert wrong["metrics"]["citation_recall"] == 0.0
    assert wrong["metrics"]["answer_correct"] is True


def test_unanswerable_control_is_not_reported_as_a_failure() -> None:
    case = EvalCase(
        case_id="unanswerable-01",
        category="unanswerable",
        question="orbital lobster migration",
        expected_status=INSUFFICIENT,
        gold_sources=(),
        answer_tokens=(),
    )
    row = evaluate_case(
        case,
        _answer(
            status=INSUFFICIENT,
            answer="insufficient_evidence",
            source=None,
        ),
        latency_ms=5.0,
    )
    assert row["metrics"]["status_correct"] is True
    summary = aggregate([row])
    assert summary["unanswerable_status_accuracy"] == 1.0
    assert summary["unanswerable_empty_citation_rate"] == 1.0

    report = {
        "generated_at": "2026-01-01T00:00:00+00:00",
        "evidence_kind": "deterministic_extractive",
        "config": {"mode": "extractive", "max_sources": 1},
        "provenance": {"answerer": "extractive"},
        "environment": {
            "platform": "test",
            "postgres": "postgresql+psycopg://***:***@localhost/test",
            "elasticsearch_url": "http://localhost:9200",
            "elasticsearch_version": "8.19.3",
            "license": {"type": "basic", "status": "active"},
            "embedding_provider": "hashing",
            "embedding_model": "test-hash",
            "embedding_dimension": 256,
        },
        "fixture": {"workspace_id": "test-workspace"},
        "summary": summary,
        "by_category": _category_summary([row]),
        "cases": [row],
    }
    markdown = _render_markdown(report)
    assert "No failures" in markdown
    failure_section = markdown.split("## Per-question results", 1)[0]
    assert "`unanswerable-01`" not in failure_section
