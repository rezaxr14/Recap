# Architecture

## 1. Problem framing

A tool-using agent that fails has produced evidence. RECAP's job is to turn that evidence
into a *durable, typed, validated* recovery memory and then serve it back to a later run
without letting a stale or incompatible memory cause a new failure.

Three subsystems sit around one contract:

| Subsystem | Question it answers |
|---|---|
| **Recorder** | What happened, in order, with secrets removed? |
| **RepairCompiler** | What is a safe, typed repair for this failure, and did it actually work? |
| **MemoryShift + Recall** | Is this memory still applicable *here*, *now*, and is it worth applying? |

Everything else (Postgres, Elasticsearch, the mock environment, the benchmark) exists to
make those three real and testable.

## 2. Component map

```
                         +-----------------------------+
   agent / host  ----->  |  FastAPI  /  CLI  /  SDK     |
                         +--------------+--------------+
                                        |
                         +--------------v--------------+
                         |        Container (DI)        |
                         |  one object graph, no globals|
                         +--------------+--------------+
                                        |
        +-------------------+-----------+-------------+------------------+
        |                   |                         |                  |
  +-----v------+     +------v-------+          +------v-------+   +------v------+
  |  Recorder  |     | RepairCompiler|         |  MemoryShift  |   |   Recall    |
  |  (events)  |     |  (repairs)    |         |  (drift)      |   |  (select)   |
  +-----+------+     +------+-------+          +------+-------+   +------+------+
        |                   |                         |                  |
        +-------------------+------------+------------+------------------+
                                         |
                          +--------------v---------------+
                          |  Postgres (source of truth)  |
                          |  runs, events, memories, ... |
                          +--------------+---------------+
                                         | transactional outbox
                          +------------------------------+
                          | Elasticsearch (rebuildable)  |
                          |  recap-memories-v1 + aliases |
                          |  recap-chunks-v2 + aliases   |
                          +------------------------------+
```

### 2.1 Domain layer (`src/recap/domain/`)

Pure Python/Pydantic. No I/O. Contains:

- enums for every state machine (`MemoryStatus`, `RunStatus`, `ValidationStatus`,
  `CompatibilityStatus`, `ActionType`, ...)
- `ExperienceMemory` with an explicit transition table and a mandatory reason
- the predicate mini-language (`predicates.py`) used for preconditions and postconditions
- `Strategy` / `StrategyAction` with allowlisted action types and a side-effect class
- `build_search_document` / `build_embedding_text`, the single place where a memory becomes
  an index document (and where free text is re-redacted)
- canonical JSON hashing for content-addressed idempotency

The domain layer never imports infrastructure. This is what makes the lifecycle and policy
tests fast and deterministic.

### 2.2 Recorder (`src/recap/recorder/`)

`RunRecorder` owns event ingestion:

- events must arrive with strictly increasing `sequence_number`
- a replay of the same `(sequence_number, payload_hash)` is a no-op returning the stored event
- a reuse of a sequence number with a different payload is a `ConflictError`
- jumps larger than 10,000 are rejected as a client bug
- every payload is redacted before hashing and persistence, so the hash describes what was
  actually stored

Redaction is deterministic and versioned (`redaction-v1`). It is defense in depth, not a
guarantee: callers must not send secrets in the first place.

### 2.3 Recall (`src/recap/recall/`)

The pipeline is deliberately linear and each stage is inspectable:

1. redact the retrieval context
2. build query text from failure evidence
3. run retrieval branches (BM25, vector, or both)
4. fuse with RRF, dedupe, cap
5. hard eligibility filters
6. applicability (preconditions against a bounded state snapshot)
7. compatibility (fingerprints + stored MemoryShift state)
8. utility features and weighted score
9. select, apply diversity cap, or return `no_safe_memory`
10. persist the full decision

The important design choice: **retrieval is not selection**. Elasticsearch hard-filters only
`workspace_id` and operational status. Task family, tool names, environment fingerprint, and
failure code are evaluated in the application. If they were ES filters, a mismatched memory
would simply vanish and the decision record could not explain why it was unsafe.

### 2.4 RepairCompiler (`src/recap/repair_compiler/`)

```
failed run events
      |
      v
extract_failure_window      (bounded, normalized signature, error code)
      |
      v
CandidateGenerator          (deterministic templates or configured repair LLM)
      |
      v
validate_draft              (schema, bounds, real event-ID linkage)
      |
      v
StaticValidator             (no execution)
      |
      v
MockSandboxValidator        (real execution in a disposable environment)
      |
      v
PromotionService            (the only path to `active`)
```

The deterministic generator is the default and keeps the demo reproducible. The LLM
generator passes only bounded, already-redacted events from the failure window and has no
capability to bypass validation, widen permissions, or promote itself. The compiler may
return no candidates. "No repair" is a valid, recorded outcome and is preferable to a
plausible-looking but unvalidated one.

### 2.5 MemoryShift (`src/recap/memory_shift/`)

Owns everything that can make a memory *less* trustworthy over time:

- compatibility evaluation and persistence
- Beta-Binomial reliability statistics
- degradation (`active -> suspect -> quarantined`) driven by verified outcomes
- revalidation job queue

It never rewrites memory content. A memory that no longer applies is marked and queued for
revalidation; the old version and its audit trail remain intact.

### 2.6 Infrastructure

- `infrastructure/db/`: SQLAlchemy models, repositories, session/engine. Postgres in
  production and dev, SQLite tolerated for fast tests.
- `infrastructure/elasticsearch/`: client, mappings, index manager, query DSL, hybrid
  fusion, search repository.
- `infrastructure/embeddings/`: embedding factory: hashing/fake/local adapters plus the
  OpenAI-compatible API adapter.
- `providers/`: `ChatProvider`/`EmbeddingProvider` ports, capability metadata,
  OpenAI-compatible HTTP client (retries, auth mapping, structured output, tool calls), and
  redacted role reporting. Model access never leaks into domain policy.

All remote model calls go through `providers/`. The adapters speak the OpenAI-compatible
wire format and require an already-running endpoint; RECAP does not download weights. There
is no silent fallback from a real provider to a fake one.

All Elasticsearch DSL is confined to `infrastructure/elasticsearch/`. Version-sensitive
syntax never leaks into business logic.

### 2.7 Environments

`environments/mock_api.py` is a deterministic, in-memory paginated API with two versions:

- v1: `cursor`
- v2: `page_token`, and passing `cursor` returns `UNKNOWN_PARAMETER`

Both schemas set `additionalProperties: false`, so the wrong parameter is rejected the way a
real strict API would reject it. Fault injection (rate limits, expired cursors) is explicit
and local. No network, no credentials, no irreversible operations.

`environments/reference_agent.py` is a deterministic agent that follows memory-mapped
parameter names. It does not inspect the environment version, so success genuinely depends
on receiving a correct memory rather than on privileged knowledge.

`environments/llm_agent.py` is the configured-model tool loop. It sends the environment's
declared tool schema to the `llm` role, validates every requested tool name and argument,
executes only `list_records`, and derives success from the records actually returned. Memory
is injected as bounded advisory JSON with an explicit untrusted-data preamble; the model
cannot execute code or widen the tool surface. `run_recording.py` persists both agent kinds
with identical event semantics and records the model provider, model name, prompt version,
token usage, and tool/retry counts.

### 2.8 Evaluation

`evals/` contains the task suite, metrics, and the benchmark runner. Baselines share the
same tasks, environment, agent, and tool budget. The safety metric
(`incompatible_selection_rate`) is computed **after** the run from recorded selections and
never feeds back into selection.

## 3. Request flows

### 3.1 Record a run

```
POST /api/v1/runs                  -> RunRecorder.start_run
POST /api/v1/runs/{id}/events      -> validate order -> redact -> persist
POST /api/v1/runs/{id}/finish      -> finalize status/counters
```

### 3.2 Retrieve a memory

```
POST /api/v1/retrieval/select
  -> RecallEngine.retrieve
     -> EmbeddingProvider.embed_query
     -> ElasticsearchMemoryRepository.search   (BM25 / kNN / both)
     -> fuse (RRF)
     -> for each candidate: eligibility, applicability, compatibility, features
     -> select or no_safe_memory
     -> RetrievalDecisionRepository.save
  <- RetrievalDecision (selected, rejected + reasons, latency, policy version)
```

### 3.3 Compile and promote a repair

```
POST /api/v1/repairs/compile
  -> RepairCompiler.compile(run_id, environment)
     -> extract_failure_window
     -> generate drafts -> validate drafts -> persist candidates (status=draft)

POST /api/v1/repairs/{id}/validate
  -> ValidatorRegistry.get(name).validate(candidate, environment)
     -> persist ValidationResult, update candidate status

POST /api/v1/repairs/{id}/promote
  -> PromotionService.promote
     -> re-check every promotion requirement
     -> create active memory -> audit -> index
```

### 3.4 Drift

```
MemoryShiftService.evaluate_compatibility(memory, environment)
  -> compatible | unknown | suspect | incompatible (persisted)

MemoryShiftService.on_incompatible_environment(memory, environment)
  -> active -> suspect
  -> enqueue revalidation job (deduplicated)
```

## 4. Key invariants

1. Postgres is authoritative. Any Elasticsearch document can be rebuilt from it.
2. A memory is `active` only via `PromotionService` (or an explicit manual API call that
   still respects the transition table and records an audit event).
3. `incompatible` memories are never operationally selectable; `suspect` memories are not
   selectable by default (`allow_suspect_operational_memories=false`).
4. Unknown preconditions are not satisfied. Operational selection fails closed.
5. Every retrieval decision records *why* each candidate was rejected.
6. Every state change on a memory writes an audit event with a reason.
7. Secrets are redacted before they reach the database, the index, or a log line.
8. Memory content is data. RECAP never executes it.

## 5. Why these boundaries

- **Domain without I/O** keeps policy testable without Docker and prevents accidental
  coupling between scoring and storage.
- **One container** means the API, CLI, demo, and tests share identical wiring. There is no
  second "test path" that behaves differently from production.
- **Typed actions instead of code** removes the entire class of "agent writes and executes a
  patch" risk. The cost is expressiveness; that cost is deliberate.
- **Two stores with one direction of truth** gives fast retrieval without letting the index
  become a second source of truth.

## 6. Extension points

| Port | Current implementation | How to extend |
|---|---|---|
| `EmbeddingProvider` | OpenAI-compatible API, hashing, fake, local sentence-transformers | implement `embed_documents`/`embed_query`, add capabilities, set `embeddings.provider` |
| `ChatProvider` | OpenAI-compatible API (agent and repair roles) | implement `chat`/`generate_structured`, set `llm.provider` or `repair_llm.provider` |
| `Validator` | static, mock sandbox | implement `validate()`, register in `build_default_registry()` |
| `CandidateGenerator` | deterministic; LLM adapter wired | implement `generate(events=...)`, inject into `RepairCompiler` |
| Reranker | `LexicalReranker` (`hybrid_rerank`) | implement the `Reranker` protocol and set `recall.reranker` |
| Environment | mock API v1/v2 | implement a fingerprint + tool registry |

## 7. Known architectural gaps

See the README capability table and `docs/evidence/phase1-summary.md` for the
evidence-backed list. The architecturally significant gaps are:

- no always-on scheduler: the outbox worker (`recap search drain`) is pull-based and
  revalidation jobs are processed on demand
- run-level snapshots are enforced only when retrieval carries a `run_id`; the benchmark
  currently relies on a read-only workspace rather than creating a run per task
- the deterministic reference agent still drives the reproducible demo and benchmark; the
  configured `llm` role now drives a separate tool-using agent (`recap agent run`), but the
  benchmark has not been run with it
- live-provider acceptance requires an operator-configured endpoint and is not exercised by CI
- benchmark reports currently use deterministic/hashing components, not real-model numbers
- no authentication layer; `workspace_id` is a data boundary, not an identity
