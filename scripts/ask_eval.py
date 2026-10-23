"""Run the Ask mini-eval over a seeded Postgres + Elasticsearch workspace.

The fixture is deliberately small and inspectable: ten experience memories
(active, suspect, quarantined, retired), ten failed agent runs with one event
each, and forty questions. Thirty questions are answerable from one recorded
source; ten are unanswerable controls with no matching evidence.

Two evidence modes are supported:

* ``extractive`` (default) is deterministic and uses the production extractive
  answerer. It is the reproducible headline; the script itself requires live
  Postgres and Elasticsearch and is not part of the CI test job.
* ``llm`` calls the configured OpenAI-compatible chat endpoint. This is live
  model evidence, not a deterministic benchmark; the exact model, endpoint, and
  timeouts are recorded in the report.

The script creates a unique workspace, seeds it through the production services,
runs every question, writes JSON plus Markdown, and deletes the workspace by
default. It never uses the v3 benchmark corpus or the operational demo workspace.

Example:

    python scripts/ask_eval.py --mode extractive
    python scripts/ask_eval.py --mode llm --base-url http://localhost:1234/v1 \
        --model qwen3.5-4b --timeout 120
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from elasticsearch import Elasticsearch

from recap.ask.models import AskAnswer, AskRequest
from recap.ask.service import AskService
from recap.config import LlmSettings, Settings, load_settings, redact_url
from recap.container import Container
from recap.domain.enums import (
    ActionType,
    EventType,
    MemoryStatus,
    RunStatus,
    SideEffectClass,
)
from recap.domain.memories import ExperienceMemory, MemoryContent
from recap.domain.predicates import Predicate
from recap.domain.runs import AgentRun, RunEvent, utc_now
from recap.domain.strategies import RepairStrategy, StrategyAction
from recap.environments.mock_api import PaginatedApiEnvironment
from recap.infrastructure.db.repositories import ensure_workspace
from recap.maintenance import reset_workspace
from recap.providers.base import ChatProvider
from recap.providers.factory import build_chat_provider

TOKENS = (
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
# A second unique marker per item keeps the gold source unambiguous even when
# several memories share a generic word such as "recovery" or "status".
MARKERS = {
    "zephyr": "alpha",
    "quartz": "beta",
    "nimbus": "gamma",
    "tundra": "delta",
    "cobalt": "epsilon",
    "ember": "zeta",
    "fjord": "eta",
    "garnet": "theta",
    "harbor": "iota",
    "indigo": "kappa",
}
# Seven active, one suspect (indexed), one quarantined and one retired (not in
# the operational index; Ask must supplement them from Postgres).
STATUSES = {
    **{token: MemoryStatus.ACTIVE for token in TOKENS[:7]},
    "garnet": MemoryStatus.SUSPECT,
    "harbor": MemoryStatus.QUARANTINED,
    "indigo": MemoryStatus.RETIRED,
}
UNANSWERABLE_QUESTIONS = (
    "orbital lobster migration",
    "lunar greenhouse protocol",
    "volcanic penguin orchestra",
    "quantum marmalade recipe",
    "submarine banjo parade",
    "glacier accordion festival",
    "meteorite bakery schedule",
    "holographic turnip museum",
    "desert whale census",
    "neon cactus arbitration",
)

ANSWERED = "answered"
INSUFFICIENT = "insufficient_evidence"


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    category: str
    question: str
    expected_status: str
    gold_sources: tuple[str, ...]
    answer_tokens: tuple[str, ...]


@dataclass(frozen=True)
class Fixture:
    workspace_id: str
    memory_ids: dict[str, str]
    run_ids: dict[str, str]
    statuses: dict[str, str]
    cases: tuple[EvalCase, ...]


def build_cases(
    memory_ids: dict[str, str],
    run_ids: dict[str, str],
    statuses: dict[str, str],
) -> tuple[EvalCase, ...]:
    """Return the fixed 40-question suite.

    Keeping this deterministic (and importable) lets a unit test assert that the
    suite still has at least 30 gold-labelled answerable questions and ten
    unanswerable controls without touching Postgres or Elasticsearch.
    """
    cases: list[EvalCase] = []
    for token in TOKENS:
        marker = MARKERS[token]
        memory_id = memory_ids[token]
        run_id = run_ids[token]
        status = statuses[token]
        cases.append(
            EvalCase(
                case_id=f"memory-{token}",
                category="memory_content",
                question=f"What recovery is recorded for {token} {marker}?",
                expected_status=ANSWERED,
                gold_sources=(f"memory:{memory_id}",),
                answer_tokens=(token,),
            )
        )
        cases.append(
            EvalCase(
                case_id=f"status-{token}",
                category="memory_lifecycle",
                question=f"What is the recorded status of {token} {marker}?",
                expected_status=ANSWERED,
                gold_sources=(f"memory:{memory_id}",),
                answer_tokens=(status,),
            )
        )
        cases.append(
            EvalCase(
                case_id=f"run-{token}",
                category="run_history",
                question=(
                    f"Which run recorded the {token} {marker} task failure and what was its status?"
                ),
                expected_status=ANSWERED,
                gold_sources=(f"run:{run_id}",),
                # The run id is the identity evidence; the answer only has to
                # state the requested status. Requiring the fixture token here
                # would penalise a correct answer that cites the run by id.
                answer_tokens=("failed",),
            )
        )
    for index, question in enumerate(UNANSWERABLE_QUESTIONS, start=1):
        cases.append(
            EvalCase(
                case_id=f"unanswerable-{index:02d}",
                category="unanswerable",
                question=question,
                expected_status=INSUFFICIENT,
                gold_sources=(),
                answer_tokens=(),
            )
        )
    return tuple(cases)


def seed_fixture(container: Container, workspace_id: str) -> Fixture:
    """Create the memories, runs, and events through production services."""
    environment = PaginatedApiEnvironment(version="2")
    fingerprint = environment.fingerprint()
    memory_ids: dict[str, str] = {}
    run_ids: dict[str, str] = {}
    statuses = {token: status.value for token, status in STATUSES.items()}

    with container.services() as services:
        # The script deletes its workspace at the end. Refuse to reuse a
        # non-empty workspace so a typo cannot wipe real local data.
        if services.memories.list_memories(workspace_id, limit=1) or services.runs.list_runs(
            workspace_id, limit=1
        ):
            raise RuntimeError(
                f"workspace {workspace_id} already contains data; refusing to seed or clean it"
            )
        ensure_workspace(services.session, workspace_id)
        for token in TOKENS:
            status = STATUSES[token]
            marker = MARKERS[token]
            failure_code = f"INVALID_{token.upper()}_{marker.upper()}"
            content = MemoryContent(
                title=f"{token.title()} {marker} pagination recovery",
                summary=(
                    f"The {token} {marker} pipeline requires a cursor refresh before "
                    "retrying the paginated request."
                ),
                failure_signature=f"ValidationError|{failure_code}|list_records",
                failure_code=failure_code,
                task_families=[f"{token}_{marker}_pagination"],
                tool_names=["list_records"],
                tool_schema_fingerprints={
                    "list_records": fingerprint.tool_schema_hashes["list_records"]
                },
                environment_type="mock_api",
                preconditions=[Predicate(field="error_code", operator="in", value=[failure_code])],
                strategy=RepairStrategy(
                    summary=(f"Refresh the {token} {marker} cursor and retry from the first page."),
                    actions=[
                        StrategyAction(
                            action_type=ActionType.REFRESH_PAGINATION_CURSOR,
                            tool_name="list_records",
                            description=f"Refresh the {token} {marker} pagination cursor.",
                            side_effect_class=SideEffectClass.READ_ONLY,
                        )
                    ],
                ),
                postconditions=[Predicate(field="records_collected", operator="gte", value=1)],
                known_limitations=[
                    f"The {token} {marker} endpoint may rate-limit repeated cursor refreshes."
                ],
            )
            memory = ExperienceMemory(
                workspace_id=workspace_id,
                content=content,
                status=status,
                status_reason=f"{status.value} ask mini-eval lifecycle fixture",
                environment_id=fingerprint.environment_id,
                environment_fingerprint=fingerprint.fingerprint,
            )
            created = services.memories.create_memory(memory)
            # Quarantined and retired memories stay out of the operational index
            # so the Ask service must recover them from Postgres.
            if status in {MemoryStatus.ACTIVE, MemoryStatus.SUSPECT} and not services.index_sync(
                created
            ):
                raise RuntimeError(f"failed to index memory {created.id}")
            memory_ids[token] = created.id

        for token in TOKENS:
            marker = MARKERS[token]
            now = utc_now()
            run = AgentRun(
                workspace_id=workspace_id,
                agent_id="ask-mini-eval-agent",
                task_id=f"task-{token}-{marker}",
                task_family=f"{token}_{marker}_pagination",
                environment_id=fingerprint.environment_id,
                environment_fingerprint=fingerprint.fingerprint,
                model_provider="deterministic",
                model_name="ask-mini-eval-fixture-v1",
                status=RunStatus.FAILED,
                failure_type=f"silent_{token}_{marker}_drift",
                success_criteria="collect all records",
                tool_calls_count=3,
                started_at=now,
                finished_at=now,
            )
            services.runs.create_run(run)
            services.runs.append_event(
                RunEvent(
                    run_id=run.id,
                    sequence_number=1,
                    event_type=EventType.TOOL_CALL_FAILED,
                    tool_name="list_records",
                    payload_redacted={
                        "token": token,
                        "marker": marker,
                        "error_code": f"INVALID_{token.upper()}_{marker.upper()}",
                        "detail": (
                            f"The {token} {marker} response omitted the final page and "
                            "returned an expired cursor."
                        ),
                    },
                )
            )
            run_ids[token] = run.id

    container.es_client.indices.refresh(index=container.index_manager.write_index)
    return Fixture(
        workspace_id=workspace_id,
        memory_ids=memory_ids,
        run_ids=run_ids,
        statuses=statuses,
        cases=build_cases(memory_ids, run_ids, statuses),
    )


def _mean(values: Sequence[float]) -> float:
    return round(statistics.fmean(values), 4) if values else 0.0


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(
        len(ordered) - 1,
        max(0, int(round((percentile / 100.0) * (len(ordered) - 1)))),
    )
    return round(ordered[index], 3)


def evaluate_case(
    case: EvalCase,
    answer: AskAnswer,
    latency_ms: float,
) -> dict[str, Any]:
    """Score one answer against its gold source and expected status."""
    cited = [f"{citation.type}:{citation.id}" for citation in answer.citations]
    cited_set = set(cited)
    gold = set(case.gold_sources)
    answerable = case.expected_status == ANSWERED
    status_correct = (
        answer.status == ANSWERED if answerable else answer.status == INSUFFICIENT and not cited
    )
    if answerable:
        precision = len(cited_set & gold) / len(cited_set) if cited_set else 0.0
        recall = len(cited_set & gold) / len(gold) if gold else 0.0
        answer_correct = all(
            token.casefold() in answer.answer.casefold() for token in case.answer_tokens
        )
    else:
        precision = 0.0
        recall = 0.0
        answer_correct = status_correct
    return {
        "case_id": case.case_id,
        "category": case.category,
        "question": case.question,
        "expected_status": case.expected_status,
        "expected_sources": list(case.gold_sources),
        "expected_answer_tokens": list(case.answer_tokens),
        "status": answer.status,
        "answer": answer.answer,
        "citations": [citation.model_dump() for citation in answer.citations],
        "cited_sources": cited,
        "verification": answer.verification.model_dump(),
        "retrieval": answer.retrieval.model_dump(),
        "latency_ms": round(latency_ms, 3),
        "metrics": {
            "status_correct": status_correct,
            "citation_precision": round(precision, 4),
            "citation_recall": round(recall, 4),
            "answer_correct": answer_correct,
            "quote_valid": answer.verification.invalid == 0,
        },
        "error": None,
    }


def error_row(case: EvalCase, exc: BaseException, latency_ms: float) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "category": case.category,
        "question": case.question,
        "expected_status": case.expected_status,
        "expected_sources": list(case.gold_sources),
        "expected_answer_tokens": list(case.answer_tokens),
        "status": "error",
        "answer": "",
        "citations": [],
        "cited_sources": [],
        "verification": {"checked": 0, "valid": 0, "invalid": 0, "rejected": []},
        "retrieval": {},
        "latency_ms": round(latency_ms, 3),
        "metrics": {
            "status_correct": False,
            "citation_precision": 0.0,
            "citation_recall": 0.0,
            "answer_correct": False,
            "quote_valid": False,
        },
        "error": f"{type(exc).__name__}: {exc}",
    }


def aggregate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    answerable = [row for row in rows if row["expected_status"] == ANSWERED]
    unanswerable = [row for row in rows if row["expected_status"] == INSUFFICIENT]
    latencies = [float(row["latency_ms"]) for row in rows]

    def mean_metric(items: Sequence[dict[str, Any]], key: str) -> float:
        return _mean([float(item["metrics"][key]) for item in items])

    return {
        "cases": len(rows),
        "answerable_cases": len(answerable),
        "unanswerable_cases": len(unanswerable),
        "errors": sum(1 for row in rows if row["error"] is not None),
        "status_accuracy": mean_metric(rows, "status_correct"),
        "answerable_status_accuracy": mean_metric(answerable, "status_correct"),
        "answerable_citation_precision": mean_metric(answerable, "citation_precision"),
        "answerable_citation_recall": mean_metric(answerable, "citation_recall"),
        "answerable_answer_token_accuracy": mean_metric(answerable, "answer_correct"),
        "quote_invalid_cases": sum(
            1 for row in rows if int(row["verification"].get("invalid", 0)) > 0
        ),
        "duplicate_citation_cases": sum(
            1 for row in rows if len(row["cited_sources"]) != len(set(row["cited_sources"]))
        ),
        "unanswerable_status_accuracy": mean_metric(unanswerable, "status_correct"),
        "unanswerable_empty_citation_rate": _mean(
            [
                1.0 if row["status"] == INSUFFICIENT and not row["citations"] else 0.0
                for row in unanswerable
            ]
        ),
        "latency_p50_ms": _percentile(latencies, 50),
        "latency_mean_ms": _mean(latencies),
    }


def _category_summary(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    categories = sorted({str(row["category"]) for row in rows})
    return {
        category: aggregate([row for row in rows if row["category"] == category])
        for category in categories
    }


def _license_info(client: Elasticsearch) -> dict[str, Any]:
    try:
        license_info = client.license.get()
        return {
            "type": license_info.get("license", {}).get("type"),
            "status": license_info.get("license", {}).get("status"),
        }
    except Exception as exc:  # noqa: BLE001 - reported, never hidden
        return {"error": f"{type(exc).__name__}: {exc}"}


def _settings() -> Settings:
    return load_settings().model_copy(deep=True)


def _make_provider(args: argparse.Namespace) -> tuple[ChatProvider, dict[str, Any]]:
    if not args.model.strip():
        raise SystemExit("--model is required for --mode llm")
    llm = LlmSettings(
        provider="openai_compatible",
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        timeout_seconds=args.timeout,
        read_timeout_seconds=args.timeout,
        write_timeout_seconds=args.timeout,
        pool_timeout_seconds=args.timeout,
        overall_timeout_seconds=args.timeout,
        no_progress_timeout_seconds=min(args.timeout, 60.0),
        max_retries=1,
        temperature=0.0,
        max_tokens=args.max_tokens,
        structured_mode=args.structured_mode,
        stream=False,
        max_concurrency=1,
    )
    provider = build_chat_provider("ask_eval", llm)
    if provider is None:
        raise SystemExit("could not construct the ask-eval chat provider")
    provenance = {
        "provider": provider.provider_type,
        "model": provider.model,
        "base_url": redact_url(args.base_url),
        "temperature": llm.temperature,
        "max_tokens": llm.max_tokens,
        "structured_mode": llm.structured_mode,
        "stream": llm.stream,
        "timeouts_seconds": {
            "connect": llm.connect_timeout_seconds,
            "read": llm.effective_read_timeout,
            "write": llm.write_timeout_seconds,
            "pool": llm.pool_timeout_seconds,
            "overall": llm.overall_timeout_seconds,
            "no_progress": llm.no_progress_timeout_seconds,
        },
        "max_retries": llm.max_retries,
    }
    return provider, provenance


def safe_command(argv: Sequence[str]) -> str:
    """Render the CLI for evidence without copying an API key into the report."""
    redacted: list[str] = []
    hide_next = False
    for argument in argv:
        if hide_next:
            redacted.append("***")
            hide_next = False
            continue
        if argument == "--api-key":
            redacted.append(argument)
            hide_next = True
            continue
        if argument.startswith("--api-key="):
            redacted.append("--api-key=***")
            continue
        redacted.append(argument)
    return " ".join([Path(sys.argv[0]).name, *redacted])


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("extractive", "llm"), default="extractive")
    parser.add_argument("--base-url", default="http://localhost:1234/v1")
    parser.add_argument("--model", default="")
    parser.add_argument("--api-key", default="lm-studio")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument(
        "--structured-mode",
        choices=("json_schema", "json_object"),
        default="json_schema",
    )
    parser.add_argument("--max-sources", type=int, default=1)
    parser.add_argument("--max-runs", type=int, default=40)
    parser.add_argument("--workspace", default="")
    parser.add_argument("--keep", action="store_true", help="leave the seeded workspace")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--markdown", type=Path, default=None)
    args = parser.parse_args()
    if args.out is None:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        args.out = Path("benchmark/reports") / f"ask_eval_{args.mode}_{timestamp}.json"
    if args.markdown is None:
        suffix = "-live" if args.mode == "llm" else ""
        args.markdown = Path("docs/evidence") / f"phase4-ask-mini-eval{suffix}.md"
    return args


def _render_markdown(report: dict[str, Any]) -> str:
    env = report["environment"]
    provenance = report["provenance"]
    summary = report["summary"]
    categories = report["by_category"]
    rows = report["cases"]

    def pct(value: float) -> str:
        return f"{value:.1%}"

    def cell(value: Any) -> str:
        return str(value).replace("|", "\\|").replace("\n", " ").strip()

    lines = [
        "# Phase 4 - Ask mini-eval",
        "",
        f"Generated by `scripts/ask_eval.py` at `{report['generated_at']}`.",
        f"Evidence kind: **{report['evidence_kind']}**.",
        "",
        "This is a seeded synthetic workspace, not real user traffic. Every number",
        "below was produced by this run. The per-question table is the committed",
        "evidence; the script also writes raw JSON under `benchmark/reports/`",
        "(ignored by default; regenerate with the command in the evidence notes).",
        "",
        "## Provenance",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Ask mode | `{report['config']['mode']}` |",
        f"| Answerer | `{provenance.get('answerer', 'extractive')}` |",
        f"| Embedding | `{env['embedding_provider']}` / "
        f"`{env['embedding_model']}` {env['embedding_dimension']}-d |",
        f"| Postgres | `{env['postgres']}` |",
        f"| Elasticsearch | `{env['elasticsearch_url']}` "
        f"(version {env['elasticsearch_version']}) |",
        f"| License | {env['license'].get('type', 'unknown')} "
        f"({env['license'].get('status', 'unknown')}) |",
        f"| Workspace | `{report['fixture']['workspace_id']}` |",
        f"| Source cap | {report['config']['max_sources']} per question |",
        f"| Questions | {summary['cases']} "
        f"({summary['answerable_cases']} answerable, "
        f"{summary['unanswerable_cases']} unanswerable) |",
        f"| Platform | {env['platform']} |",
        f"| Command | `{cell(report.get('command', 'scripts/ask_eval.py'))}` |",
    ]
    if report["config"]["mode"] == "llm":
        lines.extend(
            [
                f"| LLM provider | `{provenance.get('provider', '')}` |",
                f"| LLM model | `{provenance.get('model', '')}` |",
                f"| LLM endpoint | `{provenance.get('base_url', '')}` |",
                f"| Temperature | {provenance.get('temperature')} |",
                f"| Max tokens | {provenance.get('max_tokens')} |",
                f"| Structured mode | `{provenance.get('structured_mode', '')}` |",
                f"| Timeouts (s) | `{json.dumps(provenance.get('timeouts_seconds', {}), sort_keys=True)}` |",
            ]
        )
    lines.extend(
        [
            "",
            "## Headline",
            "",
            "| Slice | Cases | Status acc. | Citation P | Citation R | "
            "Answer-token acc. | Quote failures |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )

    def table_row(label: str, data: dict[str, Any]) -> str:
        if data["answerable_cases"]:
            precision = pct(data["answerable_citation_precision"])
            recall = pct(data["answerable_citation_recall"])
            answer_tokens = pct(data["answerable_answer_token_accuracy"])
        else:
            precision = recall = answer_tokens = "n/a"
        return (
            f"| {label} | {data['cases']} | {pct(data['status_accuracy'])} | "
            f"{precision} | {recall} | {answer_tokens} | "
            f"{data['quote_invalid_cases']} |"
        )

    lines.append(table_row("**all**", summary))
    for category, data in categories.items():
        lines.append(table_row(category, data))
    lines.extend(
        [
            "",
            "## Answerable and unanswerable handling",
            "",
            "| Slice | Cases | Correct status | Empty citations |",
            "|---|---:|---:|---:|",
            f"| Answerable | {summary['answerable_cases']} | "
            f"{pct(summary['answerable_status_accuracy'])} | n/a |",
            f"| Unanswerable | {summary['unanswerable_cases']} | "
            f"{pct(summary['unanswerable_status_accuracy'])} | "
            f"{pct(summary['unanswerable_empty_citation_rate'])} |",
            "",
            f"Errors: **{summary['errors']}**. Cases with duplicate citations: "
            f"**{summary['duplicate_citation_cases']}**. Median per-question "
            f"latency: **{summary['latency_p50_ms']} ms** (mean "
            f"{summary['latency_mean_ms']} ms).",
            "",
            "## Failures and rejections",
            "",
        ]
    )
    failures = [
        row
        for row in rows
        if row["error"]
        or not row["metrics"]["status_correct"]
        or (
            row["expected_status"] == ANSWERED
            and (
                row["metrics"]["citation_precision"] != 1.0
                or row["metrics"]["citation_recall"] != 1.0
                or not row["metrics"]["answer_correct"]
            )
        )
        or int(row["verification"].get("invalid", 0)) > 0
    ]
    if failures:
        lines.extend(
            [
                "| Case | Expected | Got | Citations | Answer |",
                "|---|---|---|---|---|",
            ]
        )
        for row in failures:
            got = str(row["status"])
            answer = cell(str(row["answer"])[:120])
            citations = cell(", ".join(row["cited_sources"]) or "-")
            lines.append(
                f"| `{row['case_id']}` | {row['expected_status']} | {got} | "
                f"{citations} | {answer} |"
            )
    else:
        lines.append("No failures: every case met its status, citation, and token checks.")
    lines.extend(
        [
            "",
            "## Per-question results",
            "",
            "| Case | Category | Question | Expected | Status | Citations | Latency ms | Answer |",
            "|---|---|---|---|---|---|---:|---|",
        ]
    )
    for row in rows:
        citations = ", ".join(row["cited_sources"]) or "-"
        lines.append(
            f"| `{row['case_id']}` | {row['category']} | {cell(row['question'])} | "
            f"{row['expected_status']} | {row['status']} | {cell(citations)} | "
            f"{row['latency_ms']} | {cell(str(row['answer'])[:180])} |"
        )
    lines.extend(
        [
            "",
            "## Method",
            "",
            "- Each memory and run is seeded through the production repositories and",
            "  the production index synchronizer; no rows are hand-written.",
            "- Seven memories are active, one is suspect, one is quarantined, and one",
            "  is retired. Quarantined and retired memories are absent from the",
            "  operational index, so lifecycle questions exercise the Postgres",
            "  supplement path.",
            "- Answerable questions name a unique token/marker pair; the gold",
            "  citation is exactly the memory or run that contains it. Unanswerable",
            "  controls use tokens that appear nowhere in the fixture.",
            "- The fixture is deliberately answerable by construction. It validates",
            "  the retrieval path and citation contract, not open-domain recall.",
            "- `citation_precision` / `citation_recall` are macro means over the 30",
            "  answerable questions. A missing answer counts as precision and recall",
            "  zero, not as an excluded case.",
            "- For run-history questions the answer-token check only requires the",
            "  requested status (`failed`); run identity is established by the gold",
            "  citation, not by requiring the fixture token in the answer.",
            "- `quote_invalid_cases` counts answers whose citations failed the strict",
            "  programmatic check (source in context and quote a case-sensitive,",
            "  normalized substring).",
            "- The extractive answerer is deterministic. LLM mode is live-model",
            "  evidence and can vary between runs; the exact model and timeouts are",
            "  recorded above.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = _parse_args()
    settings = _settings()
    workspace_id = args.workspace.strip() or f"ask-eval-{uuid.uuid4().hex[:12]}"
    provider: ChatProvider | None = None
    provider_provenance: dict[str, Any] = {"answerer": args.mode}
    started_at = datetime.now(UTC)
    report_path = args.out
    markdown_path = args.markdown
    cleanup: dict[str, Any] = {}

    container = Container.build(settings)
    try:
        container.migrate()
        container.init_search()
        container.init_search_chunks()
        if args.mode == "llm":
            provider, provider_provenance = _make_provider(args)
            provider_provenance["answerer"] = "llm"
        fixture = seed_fixture(container, workspace_id)
        rows: list[dict[str, Any]] = []
        with container.services() as services:
            ask = AskService(services, provider=provider)
            for case in fixture.cases:
                request = AskRequest(
                    workspace_id=workspace_id,
                    question=case.question,
                    mode=args.mode,
                    max_sources=args.max_sources,
                    max_runs=args.max_runs,
                )
                started = time.perf_counter()
                try:
                    answer = ask.answer(request)
                except Exception as exc:  # noqa: BLE001 - failures are evidence
                    rows.append(error_row(case, exc, (time.perf_counter() - started) * 1000.0))
                else:
                    rows.append(
                        evaluate_case(case, answer, (time.perf_counter() - started) * 1000.0)
                    )
        summary = aggregate(rows)
        report = {
            "generated_at": started_at.isoformat(),
            "command": safe_command(sys.argv[1:]),
            "evidence_kind": (
                "deterministic_extractive" if args.mode == "extractive" else "live_llm"
            ),
            "environment": {
                "platform": platform.platform(),
                "postgres": redact_url(str(settings.storage.postgres_url)),
                "elasticsearch_url": redact_url(settings.elasticsearch.url),
                "elasticsearch_version": container.es_client.info()["version"]["number"],
                "license": _license_info(container.es_client),
                "embedding_provider": settings.embeddings.provider,
                "embedding_model": settings.embeddings.model_id,
                "embedding_dimension": settings.embeddings.dimension,
            },
            "config": {
                "mode": args.mode,
                "max_sources": args.max_sources,
                "max_runs": args.max_runs,
                "workspace": workspace_id,
            },
            "provenance": provider_provenance,
            "fixture": {
                "workspace_id": workspace_id,
                "memories": len(fixture.memory_ids),
                "runs": len(fixture.run_ids),
                "questions": len(fixture.cases),
                "statuses": fixture.statuses,
            },
            "summary": summary,
            "by_category": _category_summary(rows),
            "cases": rows,
        }
        if not args.keep:
            cleanup = reset_workspace(container, [workspace_id])
        report["cleanup"] = cleanup
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(_render_markdown(report) + "\n", encoding="utf-8")
        print(json.dumps({"summary": summary, "report": str(report_path)}, indent=2))
        if cleanup:
            print(f"cleaned workspace {workspace_id}: {cleanup}", file=sys.stderr)
        else:
            print(f"kept workspace {workspace_id}", file=sys.stderr)
    finally:
        closer = getattr(provider, "close", None)
        if callable(closer):
            closer()
        container.es_client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
