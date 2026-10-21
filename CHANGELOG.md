# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). The package version in
`pyproject.toml` is `0.1.0`; there is no tagged release yet.

## [Unreleased]

### Added

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

### Changed

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

### Fixed

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
- Provider credentials are read from the environment and excluded from reports, diagnostics,
  and repository files.
- The local Compose stack is explicitly documented as development-only; it disables
  Elasticsearch security and must not be exposed to an untrusted network.

### Known limitations

- The default embedder is a deterministic hashing lexical embedder, not a semantic model.
- The dev split is intentionally small; the `naive_fixed` vs `event_aware` chunk ablation is
  degenerate until same-environment distractors and paraphrase queries are added.
- Live model-endpoint tests are opt-in and are not run in CI.
- Live repair-candidate generation with `gemma-4-e4b-uncensored-hauhaucs-aggressive` produced
  no schema-valid candidates (provider timeouts/rejections); the deterministic generator is
  still the default.
- The CrossEncoder reranker is measured for latency, but the current two-task dev split cannot
  support a quality comparison.
- There is no multi-tenant authentication/RBAC layer; deploy behind your own gateway.
