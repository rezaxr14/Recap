# Operations

## 1. Local stack

```bash
docker compose up -d postgres elasticsearch
docker compose ps
```

| Service | Host | Credentials |
|---|---|---|
| PostgreSQL 17 | `localhost:5433` | `recap` / `recap`, db `recap` |
| Elasticsearch 8.19.3 | `localhost:9200` | security disabled (local only) |
| API (optional profile) | `localhost:8000` | `docker compose --profile full up -d api` |

Container names: `recap-postgres`, `recap-elasticsearch`, `recap-api`.

The ES image is referenced as `elasticsearch:8.19.3`. If direct registry access is blocked,
pull from a mirror and tag it locally with that exact name before `compose up`.

## 2. First-time setup

```bash
python -m venv .venv
.venv/Scripts/activate
pip install -e ".[dev]"

recap db migrate
recap search init
recap health
python scripts/run_smoke_test.py
```

`recap db migrate` applies Alembic migrations. `recap search init` is idempotent: on a fresh
stack it creates `recap-memories-v1` and `recap-chunks-v2`, then points
`recap-memories-read` / `recap-memories-write` and `recap-chunks-read` /
`recap-chunks-write` at them. After a generation rebuild, the existing write alias is
authoritative: a process with a stale `index_generation` discovers the active index instead
of creating and re-pointing a second one. The chunk generation is a configuration value
(`elasticsearch.chunks_index_generation`, default `2`); bump it when the chunk mapping
changes so a new index can be built beside the old one.

## 3. Health

### Endpoints

| Endpoint | Meaning |
|---|---|
| `GET /health/live` | process is up; no dependency checks |
| `GET /health/ready` | `ready` only if Postgres answers and ES reports green/yellow |
| `GET /api/v1/diagnostics` | version, redacted config, embedding model, index counts |

`/health/ready` returns HTTP 200 with `"status": "degraded"` when a dependency is down, so a
load balancer can distinguish "process alive" from "can serve".

### CLI

```bash
recap health
recap providers
recap doctor [--live]
recap search health
```

`recap search health` reports `index_exists`, `read_alias_exists`, `document_count` for both the
memory and chunk indices.
`recap providers` is offline and prints every role, model, and capability without secrets.
`recap doctor` probes only configured endpoints; `--live` adds one small chat call and one
real embedding call. It exits non-zero when a configured dependency fails.

## 4. API surface

All application routes are under `/api/v1` except health and diagnostics.

```
POST /api/v1/runs
POST /api/v1/runs/{run_id}/events
POST /api/v1/runs/{run_id}/finish
GET  /api/v1/runs/{run_id}
GET  /api/v1/runs/{run_id}/events

POST /api/v1/memories
GET  /api/v1/memories
GET  /api/v1/memories/{memory_id}
POST /api/v1/memories/{memory_id}/transition
POST /api/v1/memories/{memory_id}/observe-outcome
GET  /api/v1/memories/{memory_id}/compatibility

POST /api/v1/retrieval/search
POST /api/v1/retrieval/select
GET  /api/v1/retrieval/decisions/{decision_id}

POST /api/v1/repairs/compile
GET  /api/v1/repairs/{candidate_id}
POST /api/v1/repairs/{candidate_id}/validate
GET  /api/v1/repairs/{candidate_id}/validation
POST /api/v1/repairs/{candidate_id}/promote

POST /api/v1/environments/fingerprints

GET  /health/live
GET  /health/ready
GET  /api/v1/diagnostics
```

`retrieval/search` returns ranked candidates without the operational policy;
`retrieval/select` applies it. Errors are structured:

```json
{"code": "validation_error", "message": "...", "details": {...}}
```

Creating a run (`POST /api/v1/runs`) also creates a memory snapshot in the same transaction.
Pass that run's `run_id` on retrieval to pin the candidate set: memories promoted after the
run started are excluded (`not_in_memory_snapshot`), and a changed version or content hash is
rejected. Live status checks still apply, so a memory quarantined after the snapshot is never
resurrected by it. A `run_id` that does not exist, or a referenced snapshot that is missing,
returns an error rather than silently falling back to unpinned retrieval.

Error codes map to HTTP statuses: `validation_error` 422, `not_found` 404, `conflict` 409,
`invalid_transition` 409, `promotion_rejected` 422, `safety_violation` 403,
`search_unavailable` 503, `embedding_unavailable` 503, `provider_unavailable` 503,
`provider_auth_error` 502, `provider_response_error` 502, `configuration_error` 500.

## 5. Common tasks

### Reindex everything from Postgres

```bash
recap search reconcile            # memories only
recap search reconcile --chunks   # memories and their chunks
```

Reindexes active memories in the default workspace, prints
`indexed=<n> failed=<n> chunks=<n> pending_outbox=<n>`. This is the recovery path when the
index is lost, corrupt, or was built with a different embedding model.

### Build or rebuild chunks

```bash
recap chunk build --workspace demo-workspace         # all active memories
recap chunk build --memory-id mem_... --workspace demo-workspace
recap chunk show mem_... --version 1
recap chunk count
```

Chunks live in Postgres (`memory_chunks`) as the source of truth and are indexed into
`recap-chunks-v2` through the same outbox. Rebuilding is idempotent: unchanged chunks keep
their id, and chunks no longer produced by the current chunker version are deleted. A memory
transition flips `active_for_retrieval` on its chunk documents; if the chunk index is
unreachable the memory index still succeeds (chunks are rebuildable), and
`recap search reconcile --chunks` is the recovery path.

### Inspect or clear the embedding cache

```bash
recap embedding-cache
recap embedding-cache --clear
recap providers                 # includes per-process cache hits/misses and provider usage
```

The cache is keyed by the exact embedding fingerprint plus a namespaced document/query text
hash. Clearing it is always safe; the next chunk rebuild or query re-embeds. Do not edit cache
rows by hand to change a vector: delete the cache and rebuild with the intended provider.
There is no automatic eviction yet; `last_accessed_at` and `hit_count` are recorded so an
operator can prune old entries, and `--clear` is the blunt but safe option.

### Drain the index outbox

```bash
recap search drain --batch-size 20 --max-batches 100
recap search outbox
```

Indexing intent is written transactionally with the database change. If Elasticsearch was
unavailable at write time, the row stays `pending` and is retried with exponential backoff.
`search drain` is the consumer:

- claims due rows (`FOR UPDATE SKIP LOCKED` on PostgreSQL, so multiple workers are safe)
- indexes or deletes the document
- marks the row `done` on success
- otherwise reschedules it (`attempts` + backoff), and moves it to `dead` after
  `max_attempts` (8)

`recap search outbox` prints counts per state and the number of due pending rows. The worker
is pull-based, not an always-on thread inside the API: run it from cron, a supervisor, or a
scheduled job. `dead` rows are visible and must be investigated, not silently dropped.

### Rebuild a memory-index generation without downtime

```bash
recap search reindex --to-generation v2
recap search reindex --to-generation v2 --workspace demo-workspace
recap search reindex --to-generation v2 --delete-old   # after verifying v2
```

`search reindex` is the supported path for an embedding-model or dimension change. It:

1. rejects a target generation that already exists, or a provider dimension that does not
   match both the configured mapping and the actual vector length;
2. creates `<prefix>-memories-v2` with a new `dense_vector` mapping and records the embedding
   fingerprint/model/generation in the mapping `_meta`;
3. re-embeds every active memory from Postgres (suspect memories are included by default
   because the read policy may explicitly allow them; use `--active-only` to exclude them);
4. refreshes the new index and checks that its document count equals the Postgres source
   count;
5. swaps the read and write aliases in one atomic `update_aliases` request;
6. updates each memory's `index_generation`/`index_status` in Postgres.

Readers keep using the old index until step 5. If any step before the swap fails, the aliases
are untouched and the partial target index is deleted. The old index is retained by default
as a rollback artifact; `--delete-old` removes it only after the swap and database update
succeed. Writes go through `recap-memories-write`, so normal processes immediately write to
the new generation without changing `index_generation`.

`recap search reconcile` remains the per-document repair tool for a corrupt or partially
missing active index. It is not a generation rebuild.

### Inspect a memory

```bash
recap memory show <memory_id> --workspace demo-workspace
recap memory compatibility <memory_id> --environment mock-api-v2
```

### Search manually

```bash
recap memory search "cursor expired" --mode outcome_aware --workspace demo-workspace
recap memory search "unknown parameter" --mode bm25 --error-code UNKNOWN_PARAMETER
```

The output is a table of candidates with rank, score, selected/rejected, and reasons, plus
the decision id.

### Compile, validate, promote

```bash
recap repair compile <run_id> --environment-version 1
recap repair validate <candidate_id>
recap repair promote <candidate_id>
```

Promotion fails with `promotion_rejected` and a list of failed checks if the candidate is
not eligible. That is the gate working, not a bug.

### Inspect model providers

```bash
recap providers
recap doctor
recap doctor --live
```

`providers` does not touch the network. `doctor` performs only configured endpoint probes and
never scans the network. See [`providers.md`](providers.md) for role configuration, the
no-download policy, and the live test gate.

### Run the configured LLM agent

```bash
recap agent run --environment-version 2 --workspace demo-workspace
recap agent run --environment-version 1 --memory-id mem_...
```

This requires a configured `llm` role and fails closed otherwise. The model chooses tool
calls; RECAP validates and executes only the declared tool and computes success from the
records actually returned. Provider failures are not converted into fake answers. Every run
is persisted with its model metadata, token usage, and tool trace.

### Demo

```bash
recap demo run
```

Writes `demo_report.md`. It resets only its own workspace (`demo-workspace`) rows and
documents, so it is safe to re-run.

### Benchmark

```bash
recap eval run --split dev
recap eval run --split test
```

Reports in `benchmark/reports/`. Gitignored; regenerate rather than commit.

## 6. Failure modes and responses

| Symptom | Likely cause | Response |
|---|---|---|
| `/health/ready` degraded, database error | Postgres down / wrong DSN | `docker compose ps`, check `DATABASE_URL` |
| `/health/ready` degraded, ES unavailable | ES starting or down | wait for healthcheck; `curl localhost:9200/_cluster/health` |
| `search_unavailable` on retrieval | ES unreachable | same; API returns 503 rather than empty results |
| `search init` fails with mapping error | vector dimension changed | bump `index_generation`, re-init, reconcile |
| Documents missing after a write | ES failure at index time | `recap search drain`; then `recap search reconcile` if needed |
| Retrieval returns `no_safe_memory` | policy rejected everything | inspect `decision_json`; reasons are recorded |
| Memory not selected after environment upgrade | `suspect`/`incompatible` compatibility | expected; compile a new validated memory |
| Promotion rejected | failed check | response lists each check and result |
| Duplicate event conflict | same sequence with different payload | client bug; inspect `details.sequence_number` |
| Outbox rows stuck pending | ES down or worker not scheduled | `recap search drain`; inspect `last_error`; check cron |
| `provider_auth_error` | wrong/missing API key | set the role's `*_API_KEY`; the key is never echoed |
| `provider_unavailable` | model endpoint down/rate-limited | `recap doctor`, check endpoint and retry/backoff settings |
| `provider_response_error` | malformed/truncated output, bad tool args, wrong embedding dimension | inspect provider logs; verify model and schema support; do not bypass validation |
| `search init` rejects embedding dimension | model/dimension changed | use a new `index_generation`; reconcile with the new provider |
| `recap doctor` says degraded | hashing/fake embeddings | configure a real embedding endpoint or explicitly accept degraded mode |

## 7. Database operations

### Migrations

```bash
recap db migrate
```

Alembic is configured from `alembic.ini` + `migrations/`; the CLI injects the configured
URL. Current head: `0008_embedding_cache`.

### Inspection

```sql
-- memory status distribution
SELECT status, count(*) FROM memories GROUP BY status ORDER BY 2 DESC;

-- unresolved revalidation jobs
SELECT state, count(*) FROM revalidation_jobs GROUP BY state;

-- pending index work
SELECT state, count(*) FROM index_outbox GROUP BY state;

-- chunk counts per type
SELECT chunk_type, count(*) FROM memory_chunks GROUP BY chunk_type ORDER BY 2 DESC;

-- embedding cache size and hit counters (the cache is optional)
SELECT count(*) AS entries, coalesce(sum(hit_count), 0) AS hits FROM embedding_cache;

-- recent degraded memories
SELECT entity_id, action, reason, created_at
FROM audit_events WHERE action LIKE 'transition:%'
ORDER BY created_at DESC LIMIT 20;
```

### Reset (development only)

```bash
docker compose down -v      # destroys volumes: all data
docker compose up -d postgres elasticsearch
recap db migrate
recap search init
```

Never run this against anything you care about.

## 8. Elasticsearch operations

### Aliases

```bash
curl -s localhost:9200/_cat/aliases/recap-memories-*
curl -s localhost:9200/_cat/indices/recap-memories-*
curl -s localhost:9200/_cat/aliases/recap-chunks-*
curl -s localhost:9200/_cat/indices/recap-chunks-*
```

Writes go through `recap-memories-write`; reads through `recap-memories-read`.
`recap search reindex` swaps both aliases in one atomic request; `search init` only
bootstraps them on a fresh stack or verifies the existing target.

### Document identity

Documents are `"<memory_id>:v<memory_version>"`. Reindexing the same version overwrites the
same document, so reconciliation is idempotent.

### Mapping safety

The mapping is `dynamic: strict`. Adding a field to `build_search_document` without adding it
to `mappings.py` will fail indexing loudly. That is intentional: silent mapping drift is how
search relevance quietly rots.

## 9. Security operations

- Elasticsearch security is disabled **for local development only**. For any shared
  deployment, enable TLS + auth and set `ELASTICSEARCH_API_KEY`.
- There is no API authentication. Do not expose the API beyond a trusted network.
- Redaction runs before persistence and before indexing. It is pattern-based and best-effort;
  treat stored data as potentially sensitive and protect the database accordingly.
- Provider credentials (`EMBEDDING_API_KEY`, `LLM_API_KEY`, `REPAIR_LLM_API_KEY`) live in
  environment/secrets only. They are excluded from `safe_dump`, diagnostics, logs, and CLI
  output.
- Sending run evidence to a remote model endpoint is a data-processing decision. Apply your
  own retention and provider policy.
- `.env` is gitignored. `.env.example` has no real credentials.

See [`threat-model.md`](threat-model.md).

## 10. Testing

```bash
pytest                          # unit only, no network
pytest --run-integration        # + live Postgres and Elasticsearch
pytest tests/integration tests/e2e -q --run-integration
```

Integration tests use:

- database `recap_test` (`RECAP_TEST_DATABASE_URL` to override)
- index prefix `recap-test` and aliases `recap-test-memories-*`

They never touch `recap` or `recap-memories-*`. A unique `workspace_id` is generated per
test.

Live model-endpoint tests are separately gated:

```bash
export RECAP_RUN_PROVIDER_INTEGRATION=1
pytest tests/integration/test_live_providers.py --run-integration
```

They skip when no real endpoint is configured and must never be replaced by a fake. CI does
not set the flag.

CI (`.github/workflows/ci.yml`) runs lint/typecheck, unit, then integration with real
service containers, followed by the demo and benchmark.

## 11. Capacity and performance notes

- ES request timeout defaults to 5 s (`elasticsearch.request_timeout_seconds`).
- Retrieval candidate cap is 100; per-branch candidate k is 30.
- Retrieval latency is recorded per stage (`query_build`, `search`, `fuse`, `rerank` when
  applicable, `evaluate`, `total`) in every decision.
- The outbox has a pull-based worker (`recap search drain`) with retry/backoff and a `dead`
  state after `max_attempts`. It is not an always-on thread; schedule it externally.
- No connection pooling tuning beyond SQLAlchemy defaults; no load testing has been done.

## 12. Upgrade procedure

1. Back up Postgres.
2. `recap db migrate`.
3. If the embedding model or dimension changed: configure the new provider and run
   `recap search reindex --to-generation vN`. The command validates dimensions, builds beside
   the old index, and swaps aliases atomically. Never point a new dimension at an existing
   index.
4. Run `python scripts/run_smoke_test.py`.
5. Run `recap eval run --split test` and compare against the previous report.
6. Deploy.

A dimension or fingerprint change without a new index generation fails at startup and tells
the operator to run `recap search reindex`; that is the guard against mixing incompatible
vectors in one index.

## 13. Known operational gaps

- no background scheduler for revalidation jobs (the outbox worker is pull-based)
- no authentication or rate limiting
- no retention/deletion policy
- no metrics endpoint (Prometheus/OpenTelemetry); structured logs and `/diagnostics` only
- no backup automation for Postgres or Elasticsearch
- single-node ES only; no shard/replica strategy, ILM, or snapshot repository
