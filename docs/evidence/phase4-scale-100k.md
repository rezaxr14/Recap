# Phase 4 scale test: 100,000 synthetic memories

This is a retrieval-index scale measurement, not an end-to-end benchmark. It measures
Elasticsearch bulk indexing and query latency for the memory index. It does **not** include
the reranker, the recall policy, repair compilation, or an LLM. Those stages are measured
separately, or not at all.

## Environment

| Item | Value |
|---|---|
| Date | 2026-10-22 |
| Host | Windows 11 (`Windows-11-10.0.26200-SP0`) |
| Elasticsearch | 8.19.3, Docker, single node, `http://localhost:9200` |
| Harness | `scripts/scale_test.py` |
| Documents | 100,000 synthetic memories |
| Index prefix | unique temporary prefix (`recap-scale-<run>-memories-v1`) |
| Cleanup | temporary index deleted after measurement |

## Deterministic run (hashing embeddings)

```powershell
.venv\Scripts\python.exe scripts\scale_test.py `
  --count 100000 --batch-size 1000 --queries 30 `
  --out .tmp\scale-100k.json
```

| Metric | Value |
|---|---|
| Embedding provider | `hashing` |
| Embedding model | `recap-hashing-embedding-v1` |
| Vector dimension | 256 |
| Indexed documents | 100,000 |
| Index size | 166,992,760 bytes (159.3 MiB) |
| Indexing time | 90.146 s |
| Indexing throughput | 1,109.3 documents/s |
| Queries measured per mode | 30 (after 3 warm-ups) |

| Mode | p50 | p95 | Mean | Max |
|---|---:|---:|---:|---:|
| BM25 | 62.170 ms | 102.652 ms | 64.301 ms | 108.499 ms |
| Vector (kNN) | 50.250 ms | 61.881 ms | 51.767 ms | 64.077 ms |
| Hybrid RRF | 110.983 ms | 126.854 ms | 113.515 ms | 130.110 ms |

The hybrid time is the sum of both retrieval branches plus application-side fusion, so it is
expected to be higher than either branch alone.

## What the numbers mean

- The corpus is synthetic and intentionally repetitive. It measures index and query latency,
  not retrieval quality.
- Ties and query terms repeat across six fixed queries; this is a latency sample, not a
  statistical estimate over a query distribution.
- The hashing embedder is deterministic and local. This run validates the harness and the
  index path without a model dependency; it is **not** evidence of semantic retrieval quality.
- `candidate_k=30` and fused output `limit=10` were held constant for all modes.
- Latency measures Elasticsearch search + application-side fusion only. The query embedding is
  computed before the timer, so remote embedding network/model time is **not** included. A
  remote embedding endpoint would add that time on top of the vector/hybrid numbers.
- The index was refreshed once after bulk loading, then queried. No concurrent writes were
  running.
- These are single-machine, single-node measurements. They are not a capacity plan.

Raw JSON is retained locally at `.tmp/scale-100k.json` (git-ignored). The command above can be
rerun against a different Elasticsearch endpoint by setting `ELASTICSEARCH_URL`.

## Live embedding run (nomic, 768-d)

```powershell
$env:EMBEDDING_PROVIDER='openai_compatible'
$env:EMBEDDING_BASE_URL='http://localhost:1234/v1'
$env:EMBEDDING_MODEL='text-embedding-nomic-embed-text-v1.5'
$env:EMBEDDING_DIMENSIONS='768'
$env:EMBEDDING_BATCH_SIZE='32'
$env:EMBEDDING_TIMEOUT_SECONDS='600'
$env:EMBEDDING_READ_TIMEOUT_SECONDS='600'
$env:EMBEDDING_OVERALL_TIMEOUT_SECONDS='1800'
$env:EMBEDDING_MAX_RETRIES='2'
.venv\Scripts\python.exe scripts\scale_test.py `
  --count 100000 --batch-size 256 --queries 30 `
  --out .tmp\scale-100k-nomic.json
```

`generated_at`: `2026-10-22T09:24:48Z`. LM Studio served
`text-embedding-nomic-embed-text-v1.5` at `http://localhost:1234/v1`; the harness used the
configured document/query embedding path, 32 texts per embedding request. This is a live
embedding-model run, **not** a live LLM answer-generation run.

| Metric | Value |
|---|---|
| Embedding provider | `openai_compatible` |
| Embedding model | `text-embedding-nomic-embed-text-v1.5` |
| Vector dimension | 768 |
| Indexed documents | 100,000 |
| Index size | 1,753,258,222 bytes (1,672.0 MiB) |
| Indexing time | 1,474.061 s (24.6 min) |
| Indexing throughput | 67.8 documents/s |
| Queries measured per mode | 30 (after 3 warm-ups) |

| Mode | p50 | p95 | Mean | Max |
|---|---:|---:|---:|---:|
| BM25 | 52.058 ms | 91.170 ms | 59.238 ms | 110.848 ms |
| Vector (kNN) | 66.564 ms | 75.005 ms | 66.013 ms | 82.878 ms |
| Hybrid RRF | 139.041 ms | 157.451 ms | 133.848 ms | 202.238 ms |

The embedding endpoint dominates indexing time: the deterministic hashing run indexed the same
100k documents at 1,109 documents/s, while the live 768-d path reached 67.8 documents/s. The
live run is slower and the index is larger, as expected; it is not a claim that nomic improves
ranking. Query latency is Elasticsearch search + application-side fusion only; the remote query
embedding call is outside the timer.

### Operational note

The first live attempt used the production 5 s Elasticsearch request timeout and failed during
bulk loading at roughly 30k documents. The harness now raises only the isolated scale-test
request timeout to 120 s; production defaults are unchanged. The successful run above used that
setting. No partial index was left behind.

Raw JSON is retained locally at `.tmp/scale-100k-nomic.json` (git-ignored).
