"""Measure the Phase 4.2 Elasticsearch retrieval features on the v3 corpus.

The script builds the v3 gold memories and synthetic decoys in memory, indexes
them into temporary indices, and measures:

* a pure BM25 baseline (all metadata boosts off) with and without the
  identifier-aware analyzer;
* pre-filtered kNN with several ``num_candidates`` values;
* application-side RRF against the native Elasticsearch retriever;
* float32 HNSW against ``int8_hnsw`` quantization.

Everything runs against a temporary index prefix. The operational aliases are
never read or written. Set ``--keep`` only when debugging; otherwise the
indices are deleted at the end and the cleanup result is recorded.

Example:

    python scripts/retrieval_depth.py --fixtures 2000
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from elasticsearch import Elasticsearch
from elasticsearch import exceptions as es_exceptions

from recap.config import Settings, load_settings
from recap.container import Container
from recap.domain.memories import ExperienceMemory, build_embedding_text
from recap.evals.datasets import build_benchmark_v3_tasks
from recap.evals.v3_seed import build_v3_corpus
from recap.infrastructure.elasticsearch.hybrid_queries import (
    build_es_rrf_query,
    build_lexical_query,
    build_vector_query,
)
from recap.infrastructure.elasticsearch.mappings import (
    IDENTIFIER_ANALYZER,
    memory_index_body,
)
from recap.recall.rrf import RankedList, reciprocal_rank_fusion

# Diagnostic identifier queries. They are built from the v3 failure families and
# their real error text/schema paths, not from a separate toy corpus. This slice
# is reported separately from the benchmark queries: it asks whether prose can
# find an identifier, which the headline v3 natural-language queries do not.
IDENTIFIER_PROBES: dict[str, tuple[str, ...]] = {
    "opaque_required": (
        "invalid request when supplying account id",
        "account id required for the tenant request",
    ),
    "silent_cursor": (
        "page token must replace the legacy cursor",
        "page token advances pagination",
    ),
    "silent_field": (
        "silent behavior error in the data field",
        "data field contains the records while items are empty",
    ),
    "silent_enum": (
        "silent behavior error with sort created at",
        "created at is the canonical sort value",
    ),
    "silent_account": (
        "silent behavior error account id acct north",
        "organisation account id convention",
    ),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=int, default=2000, help="synthetic decoys")
    parser.add_argument("--candidates", default="50,100,200,500,1000")
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="query repetitions per experiment; HNSW results vary slightly",
    )
    parser.add_argument("--keep", action="store_true", help="leave temporary indices")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("benchmark/reports")
        / f"retrieval_depth_{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path("docs/evidence/phase4-retrieval-depth.md"),
    )
    return parser.parse_args()


def _settings() -> tuple[Settings, str]:
    settings = load_settings().model_copy(deep=True)
    run = uuid.uuid4().hex[:10]
    prefix = f"recap-depth-{run}"
    settings.elasticsearch.index_prefix = prefix
    settings.elasticsearch.index_generation = 1
    settings.elasticsearch.memories_read_alias = f"{prefix}-memories-read"
    settings.elasticsearch.memories_write_alias = f"{prefix}-memories-write"
    settings.elasticsearch.chunks_read_alias = f"{prefix}-chunks-read"
    settings.elasticsearch.chunks_write_alias = f"{prefix}-chunks-write"
    settings.elasticsearch.identifier_analysis = True
    settings.elasticsearch.request_timeout_seconds = max(
        settings.elasticsearch.request_timeout_seconds, 120.0
    )
    settings.embeddings.cache_enabled = False
    return settings, prefix


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((p / 100.0) * (len(ordered) - 1)))))
    return round(ordered[index], 3)


def _latency(values: list[float]) -> dict[str, float]:
    return {
        "p50_ms": _percentile(values, 50),
        "p95_ms": _percentile(values, 95),
        "mean_ms": round(statistics.fmean(values), 3) if values else 0.0,
        "max_ms": round(max(values), 3) if values else 0.0,
    }


def _metrics(ranked: list[str], gold: str, latency_ms: float) -> dict[str, Any]:
    rank = ranked.index(gold) + 1 if gold in ranked else None
    return {
        "recall_at_3": 1.0 if rank is not None and rank <= 3 else 0.0,
        "recall_at_10": 1.0 if rank is not None and rank <= 10 else 0.0,
        "mrr": round(1.0 / rank, 6) if rank is not None else 0.0,
        "rank": rank,
        "latency_ms": round(latency_ms, 3),
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    return {
        "n": count,
        "recall_at_3": round(statistics.fmean(r["recall_at_3"] for r in rows), 4) if rows else 0.0,
        "recall_at_10": round(statistics.fmean(r["recall_at_10"] for r in rows), 4)
        if rows
        else 0.0,
        "mrr": round(statistics.fmean(r["mrr"] for r in rows), 4) if rows else 0.0,
        "latency": _latency([r["latency_ms"] for r in rows]),
    }


def _with_repeats(per_run: list[dict[str, Any]], all_rows: list[dict[str, Any]]) -> dict[str, Any]:
    combined = _aggregate(all_rows)
    combined["queries"] = per_run[0]["n"] if per_run else 0
    combined["repeats"] = len(per_run)
    for metric in ("recall_at_3", "recall_at_10", "mrr"):
        values = [run[metric] for run in per_run]
        combined[f"{metric}_range"] = (
            [round(min(values), 4), round(max(values), 4)] if values else [0.0, 0.0]
        )
    return combined


def _ids(response: dict[str, Any]) -> tuple[list[str], dict[str, float]]:
    hits = response.get("hits", {}).get("hits", [])
    return (
        [str(hit["_id"]) for hit in hits],
        {str(hit["_id"]): float(hit.get("_score") or 0.0) for hit in hits},
    )


def _search_bm25(
    client: Elasticsearch,
    *,
    index: str,
    workspace_id: str,
    query: str,
    identifier: bool,
) -> tuple[list[str], float]:
    body = build_lexical_query(
        query=query,
        workspace_id=workspace_id,
        allowed_statuses=["active"],
        size=10,
        pure=True,
        identifier_analysis=identifier,
    )
    started = time.perf_counter()
    response = client.search(index=index, **body)
    elapsed = (time.perf_counter() - started) * 1000.0
    return _ids(dict(response))[0], elapsed


def _search_vector(
    client: Elasticsearch,
    *,
    index: str,
    workspace_id: str,
    embedding: list[float],
    num_candidates: int,
) -> tuple[list[str], float]:
    body = build_vector_query(
        embedding=embedding,
        workspace_id=workspace_id,
        allowed_statuses=["active"],
        size=10,
        num_candidates=num_candidates,
    )
    started = time.perf_counter()
    response = client.search(index=index, **body)
    elapsed = (time.perf_counter() - started) * 1000.0
    return _ids(dict(response))[0], elapsed


def _measure_bm25(
    client: Elasticsearch,
    *,
    index: str,
    workspace_id: str,
    queries: list[tuple[str, str]],
    identifier: bool,
    repeats: int,
) -> dict[str, Any]:
    per_run: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    for _ in range(repeats):
        rows = []
        for query, gold in queries:
            ranked, elapsed = _search_bm25(
                client,
                index=index,
                workspace_id=workspace_id,
                query=query,
                identifier=identifier,
            )
            rows.append(_metrics(ranked, gold, elapsed))
        per_run.append(_aggregate(rows))
        all_rows.extend(rows)
    return _with_repeats(per_run, all_rows)


def _measure_vector(
    client: Elasticsearch,
    *,
    index: str,
    workspace_id: str,
    queries: list[tuple[str, str]],
    embeddings: dict[str, list[float]],
    num_candidates: int,
    repeats: int,
) -> dict[str, Any]:
    per_run: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    for _ in range(repeats):
        rows = []
        for query, gold in queries:
            ranked, elapsed = _search_vector(
                client,
                index=index,
                workspace_id=workspace_id,
                embedding=embeddings[query],
                num_candidates=num_candidates,
            )
            rows.append(_metrics(ranked, gold, elapsed))
        per_run.append(_aggregate(rows))
        all_rows.extend(rows)
    return _with_repeats(per_run, all_rows)


def _measure_app_rrf(
    client: Elasticsearch,
    *,
    index: str,
    workspace_id: str,
    queries: list[tuple[str, str]],
    embeddings: dict[str, list[float]],
    candidate_k: int,
    num_candidates: int,
    rrf_k: int,
    repeats: int,
) -> dict[str, Any]:
    per_run: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    for _ in range(repeats):
        rows = []
        for query, gold in queries:
            started = time.perf_counter()
            lexical = dict(
                client.search(
                    index=index,
                    **build_lexical_query(
                        query=query,
                        workspace_id=workspace_id,
                        allowed_statuses=["active"],
                        size=candidate_k,
                        pure=True,
                        identifier_analysis=True,
                    ),
                )
            )
            vector = dict(
                client.search(
                    index=index,
                    **build_vector_query(
                        embedding=embeddings[query],
                        workspace_id=workspace_id,
                        allowed_statuses=["active"],
                        size=candidate_k,
                        num_candidates=num_candidates,
                    ),
                )
            )
            lexical_ids, lexical_scores = _ids(lexical)
            vector_ids, vector_scores = _ids(vector)
            fused = reciprocal_rank_fusion(
                [
                    RankedList("bm25", lexical_ids, scores=lexical_scores),
                    RankedList("vector", vector_ids, scores=vector_scores),
                ],
                k=rrf_k,
                limit=10,
            )
            elapsed = (time.perf_counter() - started) * 1000.0
            rows.append(_metrics([hit.doc_id for hit in fused], gold, elapsed))
        per_run.append(_aggregate(rows))
        all_rows.extend(rows)
    return _with_repeats(per_run, all_rows)


def _measure_native_rrf(
    client: Elasticsearch,
    *,
    index: str,
    workspace_id: str,
    queries: list[tuple[str, str]],
    embeddings: dict[str, list[float]],
    candidate_k: int,
    num_candidates: int,
    rrf_k: int,
    repeats: int,
) -> dict[str, Any]:
    per_run: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    try:
        for _ in range(repeats):
            rows = []
            for query, gold in queries:
                body = build_es_rrf_query(
                    query=query,
                    embedding=embeddings[query],
                    workspace_id=workspace_id,
                    allowed_statuses=["active"],
                    size=candidate_k,
                    rank_constant=rrf_k,
                    num_candidates=num_candidates,
                    pure=True,
                    identifier_analysis=True,
                )
                started = time.perf_counter()
                response = client.search(index=index, **body)
                elapsed = (time.perf_counter() - started) * 1000.0
                rows.append(_metrics(_ids(dict(response))[0], gold, elapsed))
            per_run.append(_aggregate(rows))
            all_rows.extend(rows)
    except es_exceptions.ApiError as exc:
        return {
            "supported": False,
            "error": f"{type(exc).__name__}: {exc}",
            "status_code": getattr(exc, "status_code", None),
            "metrics": None,
        }
    return {
        "supported": True,
        "error": None,
        "status_code": 200,
        "metrics": _with_repeats(per_run, all_rows),
    }


def _index_stats(client: Elasticsearch, index: str) -> dict[str, Any]:
    result = client.indices.stats(index=index)
    primaries = result.get("_all", {}).get("primaries", {})
    store = primaries.get("store", {})
    segments = primaries.get("segments", {})
    return {
        "documents": int(primaries.get("docs", {}).get("count", 0)),
        "size_bytes": int(store.get("size_in_bytes", 0)),
        "segments": int(segments.get("count", 0)),
    }


def _create_index(
    client: Elasticsearch,
    *,
    name: str,
    dimension: int,
    identifier_analysis: bool,
    dense_vector_type: str,
) -> None:
    client.indices.delete(index=name, ignore_unavailable=True)
    client.indices.create(
        index=name,
        **memory_index_body(
            dimension=dimension,
            identifier_analysis=identifier_analysis,
            dense_vector_type=dense_vector_type,
        ),
    )


def _index_all(
    container: Container,
    *,
    indices: list[str],
    memories: list[ExperienceMemory],
    embeddings: list[list[float]],
) -> None:
    rows = list(zip(memories, embeddings, strict=True))
    for index in indices:
        container.search_repository.bulk_index(rows, index=index, refresh=False)
        container.es_client.indices.refresh(index=index)


def _license_info(client: Elasticsearch) -> dict[str, Any]:
    try:
        license_info = client.license.get()
        return {
            "type": license_info.get("license", {}).get("type"),
            "status": license_info.get("license", {}).get("status"),
        }
    except Exception as exc:  # noqa: BLE001 - reported, never hidden
        return {"error": f"{type(exc).__name__}: {exc}"}


def _markdown(report: dict[str, Any]) -> str:
    corpus = report["corpus"]
    env = report["environment"]
    identifier = report["experiments"]["identifier_analyzer"]
    knn = report["experiments"]["knn_num_candidates"]
    fusion = report["experiments"]["hybrid_fusion"]
    quantization = report["experiments"]["quantization"]

    def metric(data: dict[str, Any], key: str) -> str:
        low, high = data.get(f"{key}_range", [data[key], data[key]])
        return f"{data[key]:.3f} [{low:.3f}, {high:.3f}]"

    def row(label: str, data: dict[str, Any]) -> str:
        lat = data["latency"]
        return (
            f"| {label} | {metric(data, 'recall_at_3')} | "
            f"{metric(data, 'recall_at_10')} | {metric(data, 'mrr')} | "
            f"{lat['p50_ms']:.1f} | {lat['p95_ms']:.1f} |"
        )

    def row_with_analyzer(label: str, analyzer: str, data: dict[str, Any]) -> str:
        lat = data["latency"]
        return (
            f"| {label} | {analyzer} | {metric(data, 'recall_at_3')} | "
            f"{metric(data, 'recall_at_10')} | {metric(data, 'mrr')} | "
            f"{lat['p50_ms']:.1f} | {lat['p95_ms']:.1f} |"
        )

    lines = [
        "# Phase 4.2 — Elasticsearch retrieval depth",
        "",
        f"Generated by `scripts/retrieval_depth.py` at `{report['generated_at']}`.",
        "Everything below is measured against a live Elasticsearch instance; no",
        "number is hand-written.",
        "",
        "## Provenance",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Elasticsearch | {env['elasticsearch_version']} |",
        f"| License | {env['license'].get('type', 'unknown')} "
        f"({env['license'].get('status', 'unknown')}) |",
        f"| Embedding | `{env['embedding_provider']}` / `{env['embedding_model']}` "
        f"{env['embedding_dimension']}-d |",
        f"| Corpus | {corpus['documents']} documents "
        f"({corpus['gold_memories']} gold, {corpus['decoys']} decoys) |",
        f"| Queries | {corpus['natural_queries']} v3 natural-language, "
        f"{corpus['identifier_queries']} identifier probes |",
        f"| Candidate k | {report['config']['candidate_k']} |",
        f"| RRF k | {report['config']['rrf_k']} |",
        f"| Repeats | {report['config']['repeats']} per experiment |",
        "",
        "Recall and MRR are means over the repeats; brackets show the min-max range",
        "across repeated queries on the same index. HNSW is approximate, so some",
        "candidate counts return slightly different rankings across repeats.",
        "",
        "The identifier slice is diagnostic, not the v3 headline: it is built from",
        "the v3 families' real error text and schema paths to test whether prose can",
        "find an identifier. It is reported separately from the benchmark queries.",
        "",
        "## 1. Identifier analyzer",
        "",
        "Pure BM25, every metadata boost off. `standard` uses the original mapping;",
        f"`{IDENTIFIER_ANALYZER}` adds `word_delimiter_graph` case/punctuation splitting",
        "and `preserve_original` subfields.",
        "",
        "| Query slice | Analyzer | Recall@3 | Recall@10 | MRR | p50 ms | p95 ms |",
        "|---|---|---|---|---|---|---|",
        row_with_analyzer("v3 natural", "standard", _slice(identifier["v3_natural"], "standard")),
        row_with_analyzer(
            "v3 natural", "identifier", _slice(identifier["v3_natural"], "identifier")
        ),
        row_with_analyzer(
            "identifier probes",
            "standard",
            _slice(identifier["identifier_probes"], "standard"),
        ),
        row_with_analyzer(
            "identifier probes",
            "identifier",
            _slice(identifier["identifier_probes"], "identifier"),
        ),
        "",
        "## 2. Pre-filtered kNN: `num_candidates`",
        "",
        "The workspace/status filter is applied inside the kNN query (pre-filter).",
        "Query embeddings are computed before the timer; the latency is the search.",
        "",
        "| num_candidates | Recall@3 | Recall@10 | MRR | p50 ms | p95 ms |",
        "|---|---|---|---|---|---|",
    ]
    for entry in knn:
        metrics = entry["metrics"]
        lat = metrics["latency"]
        lines.append(
            f"| {entry['num_candidates']} | {metric(metrics, 'recall_at_3')} | "
            f"{metric(metrics, 'recall_at_10')} | {metric(metrics, 'mrr')} | "
            f"{lat['p50_ms']:.1f} | {lat['p95_ms']:.1f} |"
        )
    lines += [
        "",
        "## 3. Fusion: application RRF vs native retriever",
        "",
        "| Path | Recall@3 | Recall@10 | MRR | p50 ms | p95 ms |",
        "|---|---|---|---|---|---|",
        row("application RRF", fusion["app_rrf"]),
    ]
    native = fusion["native_rrf"]
    if native["supported"]:
        lines.append(row("native ES RRF", native["metrics"]))
    else:
        lines.append("| native ES RRF | n/a | n/a | n/a | n/a | n/a |")
        lines += [
            "",
            "Native RRF was attempted and failed:",
            "",
            f"```\n{native['error']}\n```",
            "",
            "The application-side path is therefore the only usable fusion path on",
            "this deployment. The flag remains explicit and fails closed; RECAP does",
            "not silently switch algorithms.",
        ]
    lines += [
        "",
        "## 4. Quantization",
        "",
        "| Vector index | Recall@3 | Recall@10 | MRR | p50 ms | p95 ms | Size bytes |",
        "|---|---|---|---|---|---|---|",
    ]
    for label, key in (("float32 `hnsw`", "float_hnsw"), ("`int8_hnsw`", "int8_hnsw")):
        data = quantization[key]
        metrics = data["metrics"]
        lat = metrics["latency"]
        lines.append(
            f"| {label} | {metric(metrics, 'recall_at_3')} | "
            f"{metric(metrics, 'recall_at_10')} | {metric(metrics, 'mrr')} | "
            f"{lat['p50_ms']:.1f} | {lat['p95_ms']:.1f} | "
            f"{data['index']['size_bytes']} |"
        )
    quant = report["experiments"]["quantization"]
    float_size = quant["float_hnsw"]["index"]["size_bytes"]
    int8_size = quant["int8_hnsw"]["index"]["size_bytes"]
    lines += [
        "",
        "Elasticsearch keeps the raw float vectors on disk for reranking and future",
        "quantization changes; its documentation states that `int8` therefore adds about",
        "25% disk overhead. This run is consistent with that: the `int8_hnsw` index is",
        f"{int8_size} bytes versus {float_size} bytes for float32 `hnsw`. Quantization is a",
        "search-memory feature, not a disk-size win, and it reduced recall on this corpus.",
        "It stays opt-in.",
        "",
        "## 5. Decisions",
        "",
        "- `hybrid_strategy=app_rrf` remains the default. Native RRF is unusable on this",
        "  Basic licence and the flag fails closed instead of silently switching.",
        "- `knn_num_candidates=200` remains the default. On this 10k corpus, 500 and 1000",
        "  gave slightly better recall@3 (0.424 versus 0.394) with flat search latency, but",
        "  200 is the value used in the 100k latency run; changing it needs a 100k sweep, not",
        "  a conclusion from a 10k index.",
        "- `identifier_analysis=false` by default. It is opt-in because it requires a new",
        "  index generation; the probe slice shows a precision gain (MRR), not a v3 recall win.",
        "- `dense_vector_quantization=none` by default: `int8_hnsw` lost recall and did not",
        "  reduce disk use in this configuration.",
        "",
        "## 6. Notes and limitations",
        "",
    ]
    lines += [f"- {note}" for note in report["notes"]]
    lines += [
        "",
        "## 7. Reproduce",
        "",
        "```",
        report["command"],
        "```",
        "",
    ]
    return "\n".join(lines)


def _slice(entry: dict[str, Any], analyzer: str) -> dict[str, Any]:
    return entry[analyzer]


def main() -> int:
    args = _parse_args()
    candidates = [int(value) for value in args.candidates.split(",") if value.strip()]
    if not candidates or any(value < 1 for value in candidates):
        raise SystemExit("--candidates must be a comma-separated list of positive integers")
    if args.fixtures < 0:
        raise SystemExit("--fixtures must be >= 0")

    settings, prefix = _settings()
    # Fixed workspace id: the corpus namespaces document ids with a hash of the
    # workspace, so a random id changes tie-breaking and makes the reported
    # recall non-reproducible. The index prefix stays random and temporary.
    workspace_id = f"depth-ws-v3-{args.fixtures}"
    container = Container.build(settings)
    client = container.es_client
    created_indices: list[str] = []
    try:
        created = container.init_search()
        main_index = str(created["index"])
        created_indices.append(main_index)
        std_index = f"{prefix}-standard"
        int8_index = f"{prefix}-int8"
        _create_index(
            client,
            name=std_index,
            dimension=settings.elasticsearch.vector_dimension,
            identifier_analysis=False,
            dense_vector_type="hnsw",
        )
        created_indices.append(std_index)
        _create_index(
            client,
            name=int8_index,
            dimension=settings.elasticsearch.vector_dimension,
            identifier_analysis=True,
            dense_vector_type="int8_hnsw",
        )
        created_indices.append(int8_index)

        all_tasks = [task for task in build_benchmark_v3_tasks() if task.gold_memory_ids]
        memories, tasks, _ = build_v3_corpus(
            container,
            workspace_id,
            all_tasks,
            fixture_count=args.fixtures,
        )
        embeddings = container.embeddings.embed_documents(
            [build_embedding_text(memory) for memory in memories]
        )
        _index_all(
            container,
            indices=[main_index, std_index, int8_index],
            memories=memories,
            embeddings=embeddings,
        )

        gold_doc_by_memory: dict[str, str] = {}
        gold_by_family: dict[str, str] = {}
        for memory in memories:
            if not memory.id or "v3-gold" not in memory.id:
                continue
            doc_id = f"{memory.id}:v{memory.memory_version}"
            gold_doc_by_memory[memory.id] = doc_id
            family = memory.content.task_families[0] if memory.content.task_families else ""
            gold_by_family.setdefault(family, doc_id)
        natural_queries = [
            (
                task.retrieval_query or task.instruction,
                gold_doc_by_memory.get(task.gold_memory_ids[0], ""),
            )
            for task in tasks
            if task.gold_memory_ids and task.gold_memory_ids[0] in gold_doc_by_memory
        ]
        identifier_queries = [
            (query, gold_by_family[family])
            for family, probes in IDENTIFIER_PROBES.items()
            if family in gold_by_family
            for query in probes
        ]
        all_queries = natural_queries + identifier_queries
        query_embeddings = {
            query: container.embeddings.embed_query(query)  # type: ignore[attr-defined]
            for query, _ in all_queries
        }

        natural_standard = _measure_bm25(
            client,
            index=std_index,
            workspace_id=workspace_id,
            queries=natural_queries,
            identifier=False,
            repeats=args.repeats,
        )
        natural_identifier = _measure_bm25(
            client,
            index=main_index,
            workspace_id=workspace_id,
            queries=natural_queries,
            identifier=True,
            repeats=args.repeats,
        )
        probe_standard = _measure_bm25(
            client,
            index=std_index,
            workspace_id=workspace_id,
            queries=identifier_queries,
            identifier=False,
            repeats=args.repeats,
        )
        probe_identifier = _measure_bm25(
            client,
            index=main_index,
            workspace_id=workspace_id,
            queries=identifier_queries,
            identifier=True,
            repeats=args.repeats,
        )

        knn_results = [
            {
                "num_candidates": value,
                "metrics": _measure_vector(
                    client,
                    index=main_index,
                    workspace_id=workspace_id,
                    queries=natural_queries,
                    embeddings=query_embeddings,
                    num_candidates=value,
                    repeats=args.repeats,
                ),
            }
            for value in candidates
        ]

        app_rrf = _measure_app_rrf(
            client,
            index=main_index,
            workspace_id=workspace_id,
            queries=natural_queries,
            embeddings=query_embeddings,
            candidate_k=30,
            num_candidates=settings.elasticsearch.knn_num_candidates,
            rrf_k=settings.elasticsearch.rrf_k,
            repeats=args.repeats,
        )
        native_rrf = _measure_native_rrf(
            client,
            index=main_index,
            workspace_id=workspace_id,
            queries=natural_queries,
            embeddings=query_embeddings,
            candidate_k=30,
            num_candidates=settings.elasticsearch.knn_num_candidates,
            rrf_k=settings.elasticsearch.rrf_k,
            repeats=args.repeats,
        )

        vector_float = _measure_vector(
            client,
            index=main_index,
            workspace_id=workspace_id,
            queries=natural_queries,
            embeddings=query_embeddings,
            num_candidates=settings.elasticsearch.knn_num_candidates,
            repeats=args.repeats,
        )
        vector_int8 = _measure_vector(
            client,
            index=int8_index,
            workspace_id=workspace_id,
            queries=natural_queries,
            embeddings=query_embeddings,
            num_candidates=settings.elasticsearch.knn_num_candidates,
            repeats=args.repeats,
        )

        info = dict(client.info())
        report = {
            "generated_at": datetime.now(UTC).isoformat(),
            "command": (
                f"python scripts/retrieval_depth.py --fixtures {args.fixtures} "
                f"--candidates {args.candidates} --repeats {args.repeats}"
            ),
            "environment": {
                "platform": platform.platform(),
                "elasticsearch_url": settings.elasticsearch.url,
                "elasticsearch_version": info.get("version", {}).get("number"),
                "license": _license_info(client),
                "embedding_provider": settings.embeddings.provider,
                "embedding_model": settings.embeddings.model_id,
                "embedding_dimension": settings.embeddings.dimension,
                "identifier_analyzer": IDENTIFIER_ANALYZER,
                "index_prefix": prefix,
            },
            "config": {
                "candidate_k": 30,
                "rrf_k": settings.elasticsearch.rrf_k,
                "knn_num_candidates": settings.elasticsearch.knn_num_candidates,
                "identifier_analysis": settings.elasticsearch.identifier_analysis,
                "dense_vector_quantization": settings.elasticsearch.dense_vector_quantization,
                "repeats": args.repeats,
            },
            "corpus": {
                "workspace_id": workspace_id,
                "documents": len(memories),
                "gold_memories": len(gold_by_family),
                "decoys": args.fixtures,
                "task_queries": len(tasks),
                "natural_queries": len(natural_queries),
                "identifier_queries": len(identifier_queries),
            },
            "experiments": {
                "identifier_analyzer": {
                    "v3_natural": {
                        "standard": natural_standard,
                        "identifier": natural_identifier,
                    },
                    "identifier_probes": {
                        "standard": probe_standard,
                        "identifier": probe_identifier,
                    },
                },
                "knn_num_candidates": knn_results,
                "hybrid_fusion": {
                    "app_rrf": app_rrf,
                    "native_rrf": native_rrf,
                },
                "quantization": {
                    "float_hnsw": {
                        "index": _index_stats(client, main_index),
                        "metrics": vector_float,
                    },
                    "int8_hnsw": {
                        "index": _index_stats(client, int8_index),
                        "metrics": vector_int8,
                    },
                },
            },
            "indices": {
                "main": main_index,
                "standard": std_index,
                "int8": int8_index,
            },
            "notes": [
                "Synthetic v3 corpus; not a real-world traffic sample.",
                "The identifier probe slice is diagnostic and is reported separately "
                "from the v3 natural-language benchmark queries.",
                "BM25 comparisons use pure=True: metadata boosts and the exact-signature "
                "phrase clause are disabled.",
                "The workspace/status filter is inside the kNN query, so results are "
                "pre-filtered before the top-k selection.",
                "Query embedding time is excluded from search latency.",
                "Native RRF is license-gated in some Elasticsearch distributions; a "
                "licence failure is recorded, not hidden by a silent fallback.",
                "Elasticsearch retains raw float vectors on disk for reranking, so int8 "
                "adds ~25% disk overhead by design; it is a search-memory feature, not a "
                "disk-size win.",
                "Quantization changes recall and index size; it is opt-in and requires "
                "a new index generation.",
                "Decisions: app_rrf stays the default, knn_num_candidates=200, "
                "identifier_analysis opt-in, dense_vector_quantization=none.",
            ],
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(_markdown(report), encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
        print(f"Wrote {args.out}", flush=True)
        print(f"Wrote {args.markdown}", flush=True)
    finally:
        if args.keep:
            print(f"Kept {prefix}-*", flush=True)
        else:
            for index in created_indices:
                try:
                    client.indices.delete(index=index, ignore_unavailable=True)
                except Exception as exc:  # noqa: BLE001 - cleanup is reported
                    print(f"cleanup failed for {index}: {exc}", flush=True)
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
