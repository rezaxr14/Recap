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

`recap db migrate` applies Alembic migrations. `recap search init` is idempotent: it creates
`recap-memories-v1` if missing and points `recap-memories-read` / `recap-memories-write` at
it.

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
recap search health
```

`recap search health` reports `index_exists`, `read_alias_exists`, `document_count`.

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

Error codes map to HTTP statuses: `validation_error` 422, `not_found` 404, `conflict` 409,
`invalid_transition` 409, `promotion_rejected` 422, `safety_violation` 403,
`search_unavailable` 503, `embedding_unavailable` 503, `configuration_error` 500.

## 5. Common tasks

### Reindex everything from Postgres

```bash
recap search reconcile
```

Reindexes active memories in the default workspace, prints
`indexed=<n> failed=<n> pending_outbox=<n>`. This is the recovery path when the index is
lost, corrupt, or was built with a different embedding model.

### Rebuild the index from scratch

```bash
# 1. point at a new generation
RECAP_INDEX_GENERATION=2 recap search init
# 2. reindex into the new index (writes go to the new write alias)
recap search reconcile
# 3. the read alias has already been cut over atomically by `search init`
```

Changing `index_generation` creates `recap-memories-v2`; `IndexManager.initialize()` moves
the read alias atomically, so readers never see a partially-populated index.

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
| Documents missing after a write | ES failure at index time | `recap search reconcile`; check `pending_outbox` |
| Retrieval returns `no_safe_memory` | policy rejected everything | inspect `decision_json`; reasons are recorded |
| Memory not selected after environment upgrade | `suspect`/`incompatible` compatibility | expected; compile a new validated memory |
| Promotion rejected | failed check | response lists each check and result |
| Duplicate event conflict | same sequence with different payload | client bug; inspect `details.sequence_number` |
| Outbox rows stuck pending | no worker / ES down | reconcile manually; worker is not implemented |

## 7. Database operations

### Migrations

```bash
recap db migrate
```

Alembic is configured from `alembic.ini` + `migrations/`; the CLI injects the configured
URL. Current head: `0002_outbox_active_dedupe`.

### Inspection

```sql
-- memory status distribution
SELECT status, count(*) FROM memories GROUP BY status ORDER BY 2 DESC;

-- unresolved revalidation jobs
SELECT state, count(*) FROM revalidation_jobs GROUP BY state;

-- pending index work
SELECT state, count(*) FROM index_outbox GROUP BY state;

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
```

Writes go through `recap-memories-write`; reads through `recap-memories-read`. The read alias
is cut over atomically during `search init`.

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

CI (`.github/workflows/ci.yml`) runs lint/typecheck, unit, then integration with real
service containers, followed by the demo and benchmark.

## 11. Capacity and performance notes

- ES request timeout defaults to 5 s (`elasticsearch.request_timeout_seconds`).
- Retrieval candidate cap is 100; per-branch candidate k is 30.
- Retrieval latency is recorded per stage (`query_build`, `search`, `fuse`, `evaluate`,
  `total`) in every decision.
- The outbox has `max_attempts` and `next_attempt_at` columns, but **no consumer** yet;
  retry/backoff is not automated.
- No connection pooling tuning beyond SQLAlchemy defaults; no load testing has been done.

## 12. Upgrade procedure

1. Back up Postgres.
2. `recap db migrate`.
3. If the embedding model or dimension changed: bump `index_generation`, `recap search init`,
   `recap search reconcile`.
4. Run `python scripts/run_smoke_test.py`.
5. Run `recap eval run --split test` and compare against the previous report.
6. Deploy.

A dimension change without a new index generation will fail at `search init`; that is the
guard against mixing incompatible vectors in one index.

## 13. Known operational gaps

- no background worker for outbox or revalidation jobs
- no authentication or rate limiting
- no retention/deletion policy
- no metrics endpoint (Prometheus/OpenTelemetry); structured logs and `/diagnostics` only
- no backup automation for Postgres or Elasticsearch
- single-node ES only; no shard/replica strategy, ILM, or snapshot repository
