# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). The package version in
`pyproject.toml` is `0.1.0`; there is no tagged release yet.

## [Unreleased]

### Added

- Benchmark v3: memory-required tiers, silent drift (field, enum, account, cursor, novel
  failure), a 2,000-decoy synthetic store, field-scoped schema-path compatibility, a
  `no_memory` acceptance gate, and a non-discrimination guard. v2 is retained as the legacy
  ceiling/regression suite.
- v3 evidence artifacts: deterministic held-out matrix and a separate live-model run with the
  exact model, inference timeouts, nomic embedder, aggregate/selectable-task counts, and
  retained execution-failure cases under `docs/evidence/phase3b-v3-*.md`.
- No-build static replay web app under `web/` with a silent-drift contrast, acceptance gate,
  baseline matrix, rejected-candidate inspector, report-grounded Q&A, deterministic/live
  dataset selector, provenance labels, mobile layout, and GitHub Pages workflow. Export is
  handled by `scripts/export_replay.py`.
- `recap search worker` daemon for the index outbox, plus a packaged Compose `worker` service.
  The API never starts a hidden consumer thread.
- `scripts/scale_test.py`: temporary-prefix 100k synthetic index with bulk loading, BM25/vector/
  hybrid p50/p95 latency, deterministic hashing and configured-embedding modes, exact evidence
  in `docs/evidence/phase4-scale-100k.md`.
- Grounded Q&A at `POST /api/v1/ask`: hybrid memory retrieval plus bounded run/event history,
  an extractive default and an optional configured-LLM answerer, strict source-id and
  verbatim-quote verification, and explicit `insufficient_evidence` downgrade on any
  unverifiable citation.
- `scripts/ask_eval.py`: a 40-question Ask mini-eval over a seeded workspace (10 memories,
  10 failed runs, 30 answerable + 10 unanswerable), reporting citation precision/recall,
  quote failures, answer-token accuracy, status handling, duplicate citations, and latency.
  Deterministic extractive evidence and a separate live `qwen3.5-4b` LM Studio run are under
  `docs/evidence/phase4-ask-mini-eval*.md`, with full per-question tables and raw JSON under
  `benchmark/reports/` (ignored by default). The fixture is answerable by construction and is
  not an open-domain QA benchmark.
- Phase 4.2 Elasticsearch depth: `scripts/retrieval_depth.py`, opt-in identifier
  analyzer/subfields, pure-BM25 comparison mode, pre-filtered kNN `num_candidates`, native RRF
  retriever support with fail-closed license handling, `int8_hnsw` quantization, and
  `_source` exclusion for stored vectors. Measured on a 10,005-document v3 corpus with three
  repeats; evidence in `docs/evidence/phase4-retrieval-depth.md`.
- Unit coverage for the new mapping/query options in `tests/unit/test_es_queries.py`, plus
  integration coverage for the identifier analyzer and native-RRF fail-closed path.
- Optional shared `RECAP_API_KEY` and `RECAP_CORS_ORIGINS` settings. Health endpoints stay
  public; application routes accept `X-API-Key`; auth uses a constant-time comparison.
- API auth tests covering public health, missing/wrong key rejection, and local no-key mode.
- End-to-end run recording with idempotent events, redaction, and PostgreSQL persistence.
- Typed `RepairCompiler` that turns a failure window into a candidate, plus static and
  executing mock-sandbox validators.
- Promotion gate with provenance, validation, and no bypass path.
- Outcome-aware recall with compatibility, applicability, utility, and explicit rejection
  reasons; BM25, dense-vector, hybrid RRF, and deterministic lexical reranking.
- Event-aware and `naive_fixed` chunkers, chunk persistence, chunk-level retrieval, supporting
  evidence, and lifecycle propagation through the transactional outbox.
- Benchmark harness with card and chunk ablations, safety audit, JSON/CSV/Markdown reports,
  and a CI reproducibility check.
- Opt-in real-model benchmark runs (`--agent llm`) through the same bounded tool loop and
  persisted-run path, with append-only checkpoints keyed by `(task, baseline, seed)` and a
  signature that refuses mismatched model, prompt, embedding, or tool-budget resumes.
- Reports now record agent kind, exact model, prompt version, temperature, seed, tool budget,
  token usage, and provider failures, and `scripts/render_results.py` renders both deterministic
  and live-model reports.
- FastAPI health, diagnostics, recall, recording, and repair endpoints.
- CLI for migrations, index initialization, health, demo, benchmark, reconcile, outbox drain,
  providers, agent runs, and workspace maintenance.
- OpenAI-compatible chat and embedding adapters, a configured-LLM agent, and opt-in live
  provider tests.
- Documentation set covering architecture, data model, retrieval, drift detection, repair
  validation, providers, operations, threat model, evaluation methodology, and measured
  results.
- Repository polish: Apache-2.0 `LICENSE`, `CONTRIBUTING.md`, `SECURITY.md`,
  `CODE_OF_CONDUCT.md`, `CHANGELOG.md`, a documentation index, GitHub issue forms and a
  pull-request template, package metadata, GitHub About/topics, and a social-preview image.
- Phase 0 audit evidence: fresh-clone quickstart, live LM Studio doctor output, and the
  opt-in live-provider test result under `docs/evidence/`.
- Versioned memory-index rebuild (`recap search reindex --to-generation vN`) with dimension
  and fingerprint validation, Postgres re-embedding, atomic read/write alias swap, DB
  provenance, pre-swap rollback, and optional old-index deletion.
- Alias-authoritative index initialization: a process with a stale configured generation
  discovers the active write alias instead of creating and re-pointing a second index.
- Reranker provider port with deterministic lexical, disabled, and opt-in local
  CrossEncoder implementations; the CrossEncoder loads with `local_files_only=True`.
- Benchmark retrieval provenance (active reranker, model path, index) and per-stage p50/p95
  latency, including the rerank stage.
- Live gemma v1/v2 agent transcripts and a live repair-generator acceptance harness that
  reports schema success, retries, candidates, static validation, and sandbox validation.
- Phase 1 evidence summary distinguishing live model runs from CI and unit coverage.
- Phase 3 live-model evidence for the v2 dev and held-out test splits, with exact model,
  prompt, temperature, seed, tool budget, embedding fingerprint, token cost, and a documented
  ceiling effect: `no_memory` succeeds on every task in both splits.
- SSE streaming for OpenAI-compatible chat, including tool-call delta reassembly, first-token
  timing, optional usage, clean stream cancellation, and a non-streaming JSON fallback for
  compatible servers.
- Per-phase provider deadlines (connect, read, write, pool, overall, no-progress), bounded
  output, bounded format retries, and configurable local chat concurrency.
- Provider attempt/retry/TTFT accounting on `ChatResult`, agent runs, persisted run events, and
  benchmark rows/reports. Provider infrastructure outcomes are excluded from the benchmark
  success denominator and reported separately.
- Live repair-generation acceptance evidence at
  `docs/evidence/phase2-inference-reliability.md`: 4/4 schema-valid candidates, 4/4 static and
  mock-sandbox passes with the recorded gemma configuration.
- `ProviderTimeout` (`provider_timeout`, HTTP 504) distinguishes a stalled/timed-out endpoint
  from an unreachable or rejecting one.

### Changed

- Benchmark v3 is the current suite. Its deterministic headline uses a 256-d hashing embedder
  and synthetic decoys; the live run uses nomic 768-d and a dedicated `recap-v3-live` prefix
  and aliases, so the two vector spaces cannot be mixed.
- New indices exclude `content_embedding` from `_source`; the vector remains an index
  structure, not document content. Existing indices keep their old shape until reindexed.
- `hybrid_strategy=app_rrf` remains the production default. Native Elasticsearch RRF is
  unavailable on the Basic license measured here (HTTP 403) and the `es_rrf` option fails
  closed rather than silently falling back.
- `identifier_analysis`, `dense_vector_quantization`, and larger `knn_num_candidates` values
  are opt-in. On the measured 10k corpus, identifier subfields improved identifier-probe MRR
  but not natural-language recall; `int8_hnsw` lost recall and was larger on disk;
  `num_candidates=200` remains the default pending a 100k sweep. Enabling analyzer or
  quantization changes requires a new index generation.
- v3 memory ids are workspace-namespaced, preventing primary-key collisions when benchmarks
  share one Postgres database.
- Compose now packages `api` as a normal service and adds a separate `worker`; local services
  still use `restart: "no"` and are started explicitly.
- The README and documentation index now lead with v3 results, clearly separate deterministic
  evidence from live-model evidence, and document the static replay app.
- Chunker version bumped to `2`; chunk ids now include the chunk strategy so event-aware and
  naive chunks can coexist and be ablated.
- Chunk retrieval is scoped to the request's environment fingerprint. This prevents a chunk
  compiled for one API version from being served for another.
- Benchmark runs use a unique workspace per run, reset before and after execution, with
  stable tie-breaks and content-addressed chunk ids.
- Unchanged chunks with matching embedding fingerprint, generation, and index status are not
  re-embedded.
- Documentation now states the hashing-embedder and small-dataset limitations explicitly.
- README quickstart now includes PowerShell commands, and warns that fixed Compose container
  names/ports prevent two checkouts from running simultaneously.
- Stale embedding-cache and test-count claims were corrected; live chat/embedding/tool
  evidence is now distinguished from CI coverage.
- Local Compose services now use `restart: "no"` so they do not return at Docker/Windows
  startup; integration runs start them explicitly and stop them afterward.
- Chat requests stream by default and are serialized per local role. The default agent output
  cap is 512 tokens; the repair role is capped at 1024.
- Benchmark resume signatures now include inference settings, so a timeout, streaming, or
  output-cap change cannot silently reuse rows produced under different settings.
- Repair generation accepts the unambiguous model aliases `equals`/`not_equals` and normalizes
  them to `eq`/`neq`; the prompt also states the exact allowed operators, fields, and required
  schema-fingerprint precondition.

### Fixed

- Replay app duplicate DOM ids no longer let renderers replace their own controls; dataset
  switching, inspector, and architecture panels remain intact.
- Missing favicon no longer produces a console/network error in the replay app.
- Benchmark reproducibility gate now ignores checkpoint metadata JSON and compares only the
  rendered benchmark report.
- Removed stale workspace state that made chunk-baseline MRR drift between benchmark runs.
- Outbox worker now dispatches memory, chunk, and chunk-lifecycle rows, including deletes for
  chunks whose Postgres row is already gone.
- Chunk lifecycle changes are applied even when memory-card index synchronization fails.
- Chunk queries and startup validation now reject stale embedding fingerprints and dimension
  mismatches instead of silently mixing vectors.
- Reconcile is production-safe: one memory failure does not abort the run, orphan chunk
  documents are pruned, and the summary reports failures and pending outbox rows.
- Failure-window chunks preserve nearest preceding context when it fits, keep the failed
  result as the anchor, and do not copy the anchor into continuation packs.
- Long atomic units are truncated explicitly and propagate the truncation flag; header
  overhead is included in the token budget.
- UUID redaction covers v7 as well as v4.
- Unknown configuration keys now fail loudly instead of silently doing nothing.

### Security

- Fail-closed redaction for secrets and identifiers before persistence and indexing.
- Optional constant-time `RECAP_API_KEY` gate and explicit CORS allowlist; health endpoints
  remain public. This is a single shared key, not multi-tenant RBAC.
- Provider credentials are read from the environment and excluded from reports, diagnostics,
  and repository files.
- The local Compose stack is explicitly documented as development-only; it disables
  Elasticsearch security and must not be exposed to an untrusted network.

### Known limitations

- The default embedder is a deterministic hashing lexical embedder, not a semantic model.
- The dev split is intentionally small; the `naive_fixed` vs `event_aware` chunk ablation is
  degenerate until same-environment distractors and paraphrase queries are added.
- Live model-endpoint tests are opt-in and are not run in CI.
- Live repair-candidate generation with `gemma-4-e4b-uncensored-hauhaucs-aggressive` passed
  4/4 runs on one local machine after streaming and timeout fixes. This is a small acceptance
  run, not a multi-model reliability estimate; the deterministic generator is still the
  default.
- The streaming endpoint did not report completion-token usage in the doctor probes, so token
  throughput is reported as `null`; the recommended read timeout shown by `recap doctor
  --live` is a conservative heuristic.
- The CrossEncoder reranker is measured for latency, but the current two-task dev split cannot
  support a quality comparison.
- There is no multi-tenant authentication/RBAC layer; deploy behind your own gateway.
