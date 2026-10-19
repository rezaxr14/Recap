# Data Model

## 1. Principles

1. **Postgres is the source of truth.** Elasticsearch holds a derived document per memory
   version and can be rebuilt with `recap search reconcile`.
2. **Append-only evidence, mutable state.** Runs, events, outcomes, validation results, and
   audit events are append-only. Memory status and stats are mutable but every mutation is
   audited.
3. **Content-addressed idempotency.** Memory and candidate content hashes make duplicate
   creation detectable.
4. **Explicit lifecycle.** Status changes go through a transition table, not ad-hoc updates.

## 2. Tables

### 2.1 `workspaces`

| Column | Type | Notes |
|---|---|---|
| `id` | varchar(64) PK | workspace identifier, also the tenant boundary |
| `name` | varchar(200) | display name |
| `created_at` | timestamptz | |

Every other table carries `workspace_id` and every query is scoped by it. This is a data
boundary, not an authentication boundary (see threat model).

### 2.2 `environments`

| Column | Type | Notes |
|---|---|---|
| `id` | varchar(128) PK | e.g. `mock-api-v1` |
| `workspace_id` | varchar(64) | |
| `environment_type` | varchar(64) | e.g. `mock_api` |
| `version` | varchar(64) | e.g. `1` |
| `fingerprint` | varchar(128) | canonical hash of version + tool schemas + API schema |
| `metadata_json` | json | tool descriptors, suite version |
| `created_at` | timestamptz | |

The fingerprint is what makes drift detection possible: a memory records the fingerprint it
was validated against, and a later run reports a different one.

### 2.3 `runs`

| Column | Type | Notes |
|---|---|---|
| `id` | varchar(64) PK | |
| `workspace_id` | varchar(64) | |
| `agent_id`, `task_id`, `task_family` | varchar | |
| `environment_id`, `environment_fingerprint` | varchar | |
| `toolset_fingerprint` | varchar | hash of the tool set |
| `model_provider`, `model_name`, `model_config_fingerprint` | varchar | provenance |
| `status` | varchar(32) | `running`, `succeeded`, `failed`, `partial`, `budget_exhausted`, `cancelled`, `invalid` |
| `failure_type` | varchar null | |
| `success_criteria` | text | |
| `tool_calls_count`, `retry_count`, `input_tokens`, `output_tokens` | int | budget accounting |
| `estimated_cost` | float | |
| `duration_ms` | int null | |
| `memory_snapshot_id` | varchar null | reserved for run-level snapshot pinning (not yet enforced) |
| `config_snapshot` | json | model/tool/policy configuration |
| `started_at`, `finished_at`, `created_at`, `updated_at` | timestamptz | |

### 2.4 `run_events`

| Column | Type | Notes |
|---|---|---|
| `id` | varchar(64) PK | |
| `run_id` | FK -> runs, cascade delete | |
| `sequence_number` | int | unique per run |
| `event_type` | varchar(64) | see enum below |
| `timestamp` | timestamptz | |
| `payload_schema_version` | varchar(32) | |
| `tool_name`, `tool_version`, `correlation_id` | varchar null | |
| `payload_redacted` | json | **only** redacted payloads are stored |
| `payload_hash` | varchar(80) | hash of the redacted payload |
| `redaction_applied` | bool | |
| `redaction_rule_version` | varchar(32) | |

Constraints: `unique(run_id, sequence_number)`, index on `(run_id, sequence_number)`.

Event types: `run_started`, `task_observation`, `tool_call_requested`, `tool_call_succeeded`,
`tool_call_failed`, `state_snapshot`, `memory_retrieval_requested`,
`memory_candidate_returned`, `memory_selected`, `memory_rejected`, `memory_applied`,
`repair_candidate_created`, `repair_validation_started`, `repair_validation_finished`,
`run_finished`.

### 2.5 `memories`

| Column | Type | Notes |
|---|---|---|
| `id` | varchar(64) PK | |
| `workspace_id` | varchar(64) | |
| `family_id` | varchar(64) | groups versions of the same lesson |
| `memory_version` | int | |
| `supersedes_memory_id` | varchar(64) null | |
| `schema_version` | varchar(32) | memory content schema |
| `status` | varchar(32) | lifecycle (below) |
| `status_reason` | text | required on transition |
| `created_from` | varchar(32) | `manual`, `repair_compiler`, `imported` |
| `risk_level` | varchar(16) | `low`, `medium`, `high` |
| `access_scope` | varchar(32) | `workspace` |
| `content_json` | json | the full `MemoryContent` |
| `content_hash` | varchar(80) | canonical hash of content |
| `environment_id`, `environment_fingerprint` | varchar | origin environment |
| `embedding_model_id`, `embedding_model_version`, `embedding_dimension` | | index provenance |
| `retrieval_document_id` | varchar(128) | ES doc id (`memory_id:vN`) |
| `index_generation` | int | which ES index generation |
| `indexed_at`, `index_status` | | `pending` / `indexed` / `failed` |
| `evidence_run_ids`, `evidence_trace_ids`, `validation_record_ids` | json arrays | provenance |
| `stats_json` | json | `OutcomeStats` (Beta-Binomial counters) |
| `created_at`, `updated_at`, `activated_at`, `last_validated_at` | timestamptz | |

Constraint: `unique(workspace_id, content_hash)` prevents duplicate content in a workspace.

Versioning: the first version of a logical recovery gets `family_id = memory_id`,
`memory_version = 1`, and no `supersedes_memory_id`. When the promotion gate accepts a new
content hash for the **same** workspace + failure code + overlapping task-family/tool scope,
it creates the next version (`family_id` inherited, version incremented, `supersedes_memory_id`
set) and retires the previous `active|suspect` version with a `superseded` audit event.
Content is never overwritten; `list_versions(family_id)` returns the full ordered history.

#### Memory lifecycle

```
draft ──> validating ──> active ──> suspect ──> quarantined ──> validating
              │             │           │            │
              ├──> rejected │           ├──> retired │
              └──> draft    └──> retired└──> active  └──> retired

retired, rejected: terminal
```

Enforced by `ALLOWED_TRANSITIONS` in `domain/memories.py`. Every transition requires a
non-empty reason and produces an audit event at the service layer.

`OPERATIONAL_STATUSES = {active, suspect, quarantined}` — but `suspect` and `quarantined`
are not selectable under default policy.

### 2.6 `outcomes`

Append-only evidence of what happened when a memory was used.

| Column | Type | Notes |
|---|---|---|
| `id` | varchar(64) PK | |
| `memory_id`, `memory_version` | | |
| `workspace_id` | | |
| `run_id` | varchar null | |
| `outcome` | varchar(32) | `verified_success`, `verified_failure`, `partial`, `unknown` |
| `applied` | bool | was it actually applied |
| `selected` | bool | was it selected |
| `environment_id`, `environment_fingerprint` | | context of use |
| `task_family` | varchar | |
| `severity` | varchar | |
| `notes` | text | |
| `evidence_json` | json | |
| `recorded_at` | timestamptz | |

Only `verified_success` / `verified_failure` update reliability counters. `partial` and
`unknown` are recorded but do not move the posterior.

### 2.7 `retrieval_decisions`

| Column | Type | Notes |
|---|---|---|
| `id` | varchar(64) PK | |
| `workspace_id` | | |
| `request_id` | varchar(64) | caller-supplied correlation |
| `run_id` | varchar null | |
| `mode` | varchar(32) | `bm25`, `vector`, `hybrid_rrf`, `hybrid_rerank`, `outcome_aware` |
| `status` | varchar(32) | `selected`, `no_safe_memory`, `degraded` |
| `query_redacted` | text | |
| `policy_version` | varchar(32) | e.g. `utility-v1` |
| `decision_json` | json | **full** decision: candidates, ranks, features, reasons |
| `total_latency_ms` | float | |
| `created_at` | timestamptz | |

`decision_json` is intentionally denormalized. An audit needs the decision *as evaluated*,
not a reconstruction from current memory state.

### 2.8 `repair_candidates`

| Column | Type | Notes |
|---|---|---|
| `id` | varchar(64) PK | |
| `workspace_id` | | |
| `source_run_id` | | |
| `status` | varchar(32) | `draft`, `validating`, `validated`, `rejected`, `promoted` |
| `validation_status` | varchar(32) | `not_started`, `running`, `passed`, `failed`, `inconclusive`, `blocked`, `requires_review` |
| `risk_level` | varchar(16) | |
| `content_hash` | varchar(80) | |
| `environment_id`, `environment_fingerprint` | | |
| `candidate_json` | json | diagnosis, failure window, strategy, predicates |
| `promoted_memory_id` | varchar null | |
| `created_at`, `updated_at` | timestamptz | |

### 2.9 `validation_results`

| Column | Type | Notes |
|---|---|---|
| `id` | varchar(64) PK | |
| `candidate_id` | | |
| `status` | varchar(32) | |
| `validator_name` | varchar(128) | `static` / `mock_sandbox` |
| `attempt` | int | |
| `environment_id`, `environment_fingerprint` | | what it was validated against |
| `candidate_hash` | varchar(80) | proves which candidate version was validated |
| `result_json` | json | checks, observed outcome, detail, error |
| `started_at`, `finished_at` | timestamptz | |

`candidate_hash` matters: a candidate edited after validation must not ride on the old
passing result. Promotion re-evaluates the candidate's current hash.

### 2.10 `compatibility`

| Column | Type | Notes |
|---|---|---|
| `id` | varchar(64) PK | |
| `memory_id`, `memory_version` | | |
| `workspace_id` | | |
| `environment_id`, `environment_fingerprint` | | |
| `status` | varchar(32) | `compatible`, `unknown`, `suspect`, `incompatible` |
| `reasons_json` | json array | human-readable reasons |
| `evidence_json` | json | declared vs observed schemas |
| `updated_at` | timestamptz | |

Constraint: `unique(memory_id, memory_version, environment_fingerprint)` — compatibility is
contextual, not global.

### 2.11 `revalidation_jobs`

| Column | Type | Notes |
|---|---|---|
| `id` | varchar(64) PK | |
| `memory_id`, `memory_version` | | |
| `workspace_id` | | |
| `reason` | varchar(32) | `version_change`, `environment_changed`, `reliability_drop`, `age_review`, `conflict`, `manual` |
| `target_environment_id`, `target_environment_fingerprint` | | |
| `validation_plan` | varchar(64) | |
| `priority` | int | |
| `state` | varchar(32) | `queued`, `running`, `passed`, `failed`, `inconclusive`, `blocked`, `cancelled` |
| `attempts`, `max_attempts` | int | |
| `dedupe_key` | varchar(200) unique | prevents duplicate jobs for the same memory+reason+target |
| `result_json`, `error` | | |
| `created_at`, `updated_at` | timestamptz | |

### 2.12 `audit_events`

| Column | Type | Notes |
|---|---|---|
| `id` | varchar(64) PK | |
| `workspace_id` | | |
| `entity_type`, `entity_id` | | |
| `action` | varchar(64) | e.g. `promoted`, `degraded`, `outcome_recorded_in_foreign_context` |
| `reason` | text | |
| `actor` | varchar(64) | |
| `before_json`, `after_json` | json | |
| `created_at` | timestamptz | |

### 2.13 `index_outbox`

| Column | Type | Notes |
|---|---|---|
| `id` | varchar(64) PK | |
| `workspace_id` | | |
| `entity_type`, `entity_id` | | |
| `operation` | varchar(16) | `index` / `delete` |
| `dedupe_key` | varchar(200) | `index:<memory_id>:v<n>` |
| `state` | varchar(16) | `pending`, `processing`, `done`, `dead` |
| `attempts`, `max_attempts` | int | incremented when the worker claims a row |
| `last_error` | text | last failure detail |
| `next_attempt_at` | timestamptz | backoff scheduling |
| `created_at`, `updated_at` | timestamptz | |

Partial unique index `uq_outbox_dedupe_active` on `dedupe_key` **where**
`state IN ('pending','processing')`. A completed row must not block a later reindex of the
same memory version. This was a real bug found by an integration test (see
`migrations/versions/0002_outbox_active_dedupe.py`).

Rows are written in the same transaction as the memory change and completed on the request
path when indexing succeeds. If Elasticsearch is unavailable the row stays `pending` and
`OutboxWorker` (`recap search drain`) retries it with exponential backoff; after
`max_attempts` it becomes `dead`, which is an operator-visible terminal state rather than a
silent drop.

## 3. Memory content schema

`MemoryContent` (stored in `memories.content_json`) is the contract between the compiler and
retrieval:

```json
{
  "title": "string",
  "summary": "string",
  "failure_signature": "ErrorType|CODE|tool",
  "failure_code": "INVALID_CURSOR",
  "task_families": ["api_pagination"],
  "tool_names": ["list_records"],
  "tool_versions": {"list_records": "1"},
  "tool_schema_fingerprints": {"list_records": "sha256:..."},
  "environment_constraints": {},
  "environment_type": "mock_api",
  "preconditions": [ { "field": "...", "operator": "eq", "value": "..." } ],
  "strategy": {
    "summary": "string",
    "actions": [ { "action_type": "refresh_pagination_cursor", "parameters": {...},
                   "side_effect_class": "read_only" } ],
    "action_signature": "refresh_pagination_cursor",
    "side_effect_class": "read_only",
    "limitations": ["string"]
  },
  "postconditions": [ { "field": "...", "operator": "eq", "value": "..." } ],
  "known_limitations": ["string"],
  "counterexamples": []
}
```

Preconditions/postconditions use a bounded predicate language (`eq`, `neq`, `in`, `exists`,
`gt`, `lt`, ...) evaluated against a state snapshot. There is no expression evaluation, no
attribute access, and no callable execution.

## 4. Index document

One Elasticsearch document per memory **version**, id `"<memory_id>:v<memory_version>"`.
Fields are a flattened projection of `MemoryContent` plus operational metadata (status,
counters, fingerprint, content hash, index generation). The mapping is `dynamic: strict`, so
an accidental new field fails loudly instead of silently creating a mapping.

## 5. Invariants and where they are enforced

| Invariant | Enforced by |
|---|---|
| Event order and idempotency | `RunRecorder` + `unique(run_id, sequence_number)` |
| No unredacted payload persisted | `redact_payload` before hash/store |
| Legal memory transitions only | `ALLOWED_TRANSITIONS` + `ExperienceMemory.transition` |
| One active memory per content hash per workspace | `unique(workspace_id, content_hash)` + promotion check |
| Promotion requires passed validation | `PromotionService.evaluate` |
| Compatibility is per environment fingerprint | `unique(memory_id, version, fingerprint)` |
| No duplicate revalidation jobs | `unique(dedupe_key)` |
| No duplicate active outbox rows | partial unique index |
| Search doc matches DB content | `build_search_document` + content hash |

## 6. Migrations

- `0001_initial` — all tables above.
- `0002_outbox_active_dedupe` — drops the global unique constraint on
  `index_outbox.dedupe_key` and adds the partial unique index.

Migrations are applied with `recap db migrate` (Alembic). `recap db init` is a development
shortcut that calls `Base.metadata.create_all`; it is not a substitute for migrations in a
shared environment.

## 7. Deletion and retention

There is no automatic retention policy. Deleting a workspace would require removing its
memories from the index and its rows from Postgres; the outbox pattern covers the index
side for individual memories but no cascade delete command exists yet. This is documented
as a gap rather than silently half-implemented.
