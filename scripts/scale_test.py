"""Index synthetic memories and measure Elasticsearch retrieval latency.

The script creates a temporary index prefix, so it never touches the operational
aliases. It reports the configured embedding provider and dimension; it does not
pretend a hashing embedder is a semantic model.

Example (100k hashing documents, the default local configuration):

    python scripts/scale_test.py --count 100000

Example against a configured OpenAI-compatible embedding endpoint:

    set RECAP_EMBEDDING_PROVIDER=openai_compatible
    set RECAP_EMBEDDING_BASE_URL=http://localhost:1234/v1
    set RECAP_EMBEDDING_MODEL=text-embedding-nomic-embed-text-v1.5
    set RECAP_EMBEDDING_DIMENSION=768
    python scripts/scale_test.py --count 100000
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from recap.config import Settings, load_settings
from recap.container import Container
from recap.domain.enums import ActionType, MemoryStatus, SideEffectClass
from recap.domain.memories import ExperienceMemory, MemoryContent
from recap.domain.predicates import Predicate
from recap.domain.retrieval import RetrievalContext
from recap.domain.strategies import RepairStrategy, StrategyAction
from recap.environments.mock_api import PaginatedApiEnvironment


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=100_000, help="synthetic memories to index")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--queries", type=int, default=30, help="measured queries per mode")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("benchmark/reports") / f"scale_{datetime.now(UTC):%Y%m%dT%H%M%SZ}.json",
    )
    parser.add_argument("--keep", action="store_true", help="leave the temporary index in place")
    return parser.parse_args()


def _settings() -> tuple[Settings, str]:
    settings = load_settings().model_copy(deep=True)
    run = uuid.uuid4().hex[:10]
    prefix = f"recap-scale-{run}"
    settings.elasticsearch.index_prefix = prefix
    settings.elasticsearch.index_generation = 1
    settings.elasticsearch.memories_read_alias = f"{prefix}-memories-read"
    settings.elasticsearch.memories_write_alias = f"{prefix}-memories-write"
    settings.elasticsearch.chunks_read_alias = f"{prefix}-chunks-read"
    settings.elasticsearch.chunks_write_alias = f"{prefix}-chunks-write"
    # The scale test measures Elasticsearch, not the Postgres embedding cache.
    settings.embeddings.cache_enabled = False
    # A 768-d HNSW segment merge can exceed the production 5 s request timeout.
    # Give this isolated harness its own generous bulk/index timeout.
    settings.elasticsearch.request_timeout_seconds = max(
        settings.elasticsearch.request_timeout_seconds, 120.0
    )
    return settings, prefix


def _memory(
    index: int,
    *,
    workspace_id: str,
    environment: PaginatedApiEnvironment,
) -> ExperienceMemory:
    codes = (
        "INVALID_CURSOR",
        "INVALID_PAGE_TOKEN",
        "UNKNOWN_FIELD",
        "STALE_ACCOUNT_SCOPE",
        "RATE_LIMITED",
    )
    code = codes[index % len(codes)]
    fingerprint = environment.fingerprint()
    content = MemoryContent(
        title=f"Cursor recovery {index:07d}",
        summary=(f"Restart pagination from page one after {code}; record batch {index:07d}."),
        failure_signature=f"ValidationError|{code}|list_records",
        failure_code=code,
        task_families=["api_pagination", f"family_{index % 37}"],
        tool_names=["list_records"],
        tool_schema_fingerprints={"list_records": fingerprint.tool_schema_hashes["list_records"]},
        environment_type="mock_api",
        preconditions=[
            Predicate(field="error_code", operator="in", value=[code]),
        ],
        strategy=RepairStrategy(
            summary=f"Reset pagination state for {code}.",
            actions=[
                StrategyAction(
                    action_type=ActionType.REFRESH_PAGINATION_CURSOR,
                    tool_name="list_records",
                    side_effect_class=SideEffectClass.READ_ONLY,
                    description=f"Use page one for batch {index:07d}.",
                )
            ],
        ),
        postconditions=[Predicate(field="records_collected", operator="gte", value=1)],
    )
    return ExperienceMemory(
        id=f"mem_scale{index:07d}",
        workspace_id=workspace_id,
        content=content,
        status=MemoryStatus.ACTIVE,
        status_reason="synthetic scale fixture",
        environment_id=fingerprint.environment_id,
        environment_fingerprint=fingerprint.fingerprint,
    )


def _batches(total: int, size: int) -> Iterator[range]:
    for start in range(0, total, size):
        yield range(start, min(total, start + size))


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((p / 100.0) * (len(ordered) - 1)))))
    return round(ordered[index], 3)


def _measure(
    *,
    container: Container,
    workspace_id: str,
    queries: list[str],
    count: int,
    mode: str,
) -> dict[str, Any]:
    repository = container.search_repository
    embedder = container.embeddings
    latencies: list[float] = []
    # Warm up the query path and HNSW graph before recording.
    for index in range(min(3, count)):
        query = queries[index % len(queries)]
        context = RetrievalContext(workspace_id=workspace_id, query=query, allow_unknown_state=True)
        embedding = None if mode == "bm25" else embedder.embed_query(query)  # type: ignore[attr-defined]
        payload = repository.search(
            context=context,
            mode=mode,
            candidate_k=30,
            embedding=embedding,
        )
        repository.fuse(payload, limit=10)
    for index in range(count):
        query = queries[index % len(queries)]
        context = RetrievalContext(workspace_id=workspace_id, query=query, allow_unknown_state=True)
        embedding = None if mode == "bm25" else embedder.embed_query(query)  # type: ignore[attr-defined]
        start = time.perf_counter()
        payload = repository.search(
            context=context,
            mode=mode,
            candidate_k=30,
            embedding=embedding,
        )
        repository.fuse(payload, limit=10)
        latencies.append((time.perf_counter() - start) * 1000.0)
    return {
        "mode": mode,
        "n": len(latencies),
        "p50_ms": _percentile(latencies, 50),
        "p95_ms": _percentile(latencies, 95),
        "mean_ms": round(statistics.fmean(latencies), 3) if latencies else 0.0,
        "max_ms": round(max(latencies), 3) if latencies else 0.0,
    }


def _index_stats(client: Any, index: str) -> dict[str, Any]:
    result = client.indices.stats(index=index)
    primaries = result.get("_all", {}).get("primaries", {})
    store = primaries.get("store", {})
    return {
        "documents": int(primaries.get("docs", {}).get("count", 0)),
        "size_bytes": int(store.get("size_in_bytes", 0)),
    }


def _cleanup(container: Container, index: str) -> str:
    try:
        container.es_client.indices.delete(index=index, ignore_unavailable=True)
        return "deleted"
    except Exception as exc:  # noqa: BLE001 - cleanup failure is reported, not hidden
        return f"cleanup failed: {type(exc).__name__}: {exc}"


def main() -> int:
    args = _parse_args()
    if args.count < 1 or args.batch_size < 1 or args.queries < 1:
        raise SystemExit("count, batch-size, and queries must be positive")
    settings, prefix = _settings()
    workspace_id = f"scale-ws-{uuid.uuid4().hex[:10]}"
    environment = PaginatedApiEnvironment(version="1")
    queries = [
        "INVALID_CURSOR list_records cursor",
        "INVALID_PAGE_TOKEN page token",
        "UNKNOWN_FIELD page_token",
        "STALE_ACCOUNT_SCOPE account",
        "RATE_LIMITED retry backoff",
        "cursor recovery page one",
    ]

    container = Container.build(settings)
    started = time.perf_counter()
    created = container.init_search()
    index = str(created["index"])
    print(f"Temporary index: {index}", flush=True)
    print(
        f"Embedding: {settings.embeddings.provider} {settings.embeddings.model_id} "
        f"{settings.embeddings.dimension}-d",
        flush=True,
    )
    indexed = 0
    try:
        for batch_number, positions in enumerate(_batches(args.count, args.batch_size), start=1):
            memories = [
                _memory(position, workspace_id=workspace_id, environment=environment)
                for position in positions
            ]
            embeddings = container.embeddings.embed_documents(  # type: ignore[attr-defined]
                [memory.content.title + "\n" + memory.content.summary for memory in memories]
            )
            container.search_repository.bulk_index(
                list(zip(memories, embeddings, strict=True)), refresh=False
            )
            indexed += len(memories)
            if batch_number % 20 == 0 or indexed == args.count:
                elapsed = time.perf_counter() - started
                print(
                    f"indexed {indexed}/{args.count} ({indexed / max(elapsed, 1e-9):.0f}/s)",
                    flush=True,
                )
        container.es_client.indices.refresh(index=index)
        indexing_seconds = time.perf_counter() - started
        stats = _index_stats(container.es_client, index)
        results = [
            _measure(
                container=container,
                workspace_id=workspace_id,
                queries=queries,
                count=args.queries,
                mode=mode,
            )
            for mode in ("bm25", "vector", "hybrid_rrf")
        ]
        report = {
            "generated_at": datetime.now(UTC).isoformat(),
            "environment": {
                "platform": platform.platform(),
                "elasticsearch_url": settings.elasticsearch.url,
                "index": index,
                "embedding_provider": settings.embeddings.provider,
                "embedding_model": settings.embeddings.model_id,
                "embedding_dimension": settings.embeddings.dimension,
                "reranker": settings.recall.reranker,
            },
            "index": {
                "documents": indexed,
                **stats,
                "indexing_seconds": round(indexing_seconds, 3),
                "documents_per_second": round(indexed / max(indexing_seconds, 1e-9), 1),
            },
            "queries": results,
            "notes": [
                "synthetic memories; not a real-world corpus",
                "latency measures Elasticsearch search + fusion; query embedding is outside the timer",
                "temporary index prefix is deleted unless --keep is set",
            ],
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
        print(f"Wrote {args.out}", flush=True)
    finally:
        if args.keep:
            print(f"Kept {prefix}-*", flush=True)
        else:
            print(_cleanup(container, index), flush=True)
        container.es_client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
