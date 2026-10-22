# Documentation index

The README is the front door; this page is the map. If a document and the code disagree,
the code and its tests are authoritative and the document should be fixed in the same change.

## Start here

| Document | What it answers |
|---|---|
| [`../README.md`](../README.md) | What RECAP is, what works, quickstart, measured results, known limits |
| [`benchmark-v3.md`](benchmark-v3.md) | Current memory-required suite: tiers, silent drift, field-scoped compatibility, acceptance gate, and held-out result |
| [`results-v3.md`](results-v3.md) | Deterministic headline matrix plus the separate live-model run, including selectable-task splits and execution-failure caveats |
| [`evidence/phase3b-v3-deterministic-test.md`](evidence/phase3b-v3-deterministic-test.md) | Exact deterministic command, environment, matrix, and retained failures |
| [`evidence/phase3b-v3-live-test.md`](evidence/phase3b-v3-live-test.md) | Exact live model, inference timeouts, nomic embedder, aggregate/selectable results, and execution-failure audit |
| [`../web/README.md`](../web/README.md) | Static recorder-replay app: export, serve, deploy, provenance, and honesty rules |
| [`evidence/phase1-summary.md`](evidence/phase1-summary.md) | Evidence-backed implementation state, live-run results, and known limits |
| [`benchmark-v2.md`](benchmark-v2.md) | Legacy ceiling/regression suite: ten families, hard negatives, fair baselines, leakage checks, and limitations |
| [`evidence/phase2-summary.md`](evidence/phase2-summary.md) | Measured benchmark v2 dev results and live new-family model runs |
| [`evidence/phase2-inference-reliability.md`](evidence/phase2-inference-reliability.md) | Streaming/timeout changes, live doctor metrics, and 4/4 repair-generation acceptance |
| [`evidence/phase3-llm-dev-core/summary.md`](evidence/phase3-llm-dev-core/summary.md) | Live-model benchmark on the v2 dev split, four core baselines, ceiling effect, and limits |
| [`evidence/phase3-llm-test-core/summary.md`](evidence/phase3-llm-test-core/summary.md) | Live-model benchmark on the held-out v2 test split, four core baselines, and cache caveat |
| [`architecture.md`](architecture.md) | Components, data flow, and the separation between retrieval and selection |
| [`data-model.md`](data-model.md) | Postgres tables, Elasticsearch mappings, ids, versions, and lifecycle fields |

## Design

| Document | What it covers |
|---|---|
| [`retrieval.md`](retrieval.md) | Query construction, BM25/vector/RRF, reranking, compatibility, applicability, utility policy |
| [`repair-validation.md`](repair-validation.md) | Failure windows, candidate generation, static and executing validators, promotion gate |
| [`drift-detection.md`](drift-detection.md) | MemoryShift compatibility, degradation, and revalidation queue |
| [`providers.md`](providers.md) | Chat and embedding ports, OpenAI-compatible configuration, capability reporting |
| [`threat-model.md`](threat-model.md) | Trust boundaries, redaction, sandbox assumptions, and deployment warnings |

## Operations and evaluation

| Document | What it covers |
|---|---|
| [`operations.md`](operations.md) | Local stack, health, migrations, index initialization, outbox, reconcile, backup/restore |
| [`evaluation-methodology.md`](evaluation-methodology.md) | Dataset, baselines, metrics, reproducibility rules, and what the benchmark does not prove |
| [`results.md`](results.md) | Generated benchmark table from the committed deterministic v2 dev JSON report |
| [`results-llm.md`](results-llm.md) | Generated benchmark table from the committed live-model v2 dev JSON report |
| [`results-llm-test.md`](results-llm-test.md) | Generated benchmark table from the committed live-model v2 test JSON report |
| [`results/2026-10-20-dev.md`](results/2026-10-20-dev.md) | Earlier measured dev-split matrix and caveats |
| [`evidence/phase0-audit.md`](evidence/phase0-audit.md) | Fresh-clone quickstart, live LM Studio probe, and capability-claim corrections |

## Decisions

Architecture decision records live in [`adr/`](adr/):

- [`0001-source-of-truth.md`](adr/0001-source-of-truth.md)
- [`0002-elasticsearch-hybrid-retrieval.md`](adr/0002-elasticsearch-hybrid-retrieval.md)
- [`0003-memory-promotion-policy.md`](adr/0003-memory-promotion-policy.md)
- [`0004-embedding-model.md`](adr/0004-embedding-model.md)
- [`0005-event-aware-chunking.md`](adr/0005-event-aware-chunking.md)

## Contributing and security

- [`../CONTRIBUTING.md`](../CONTRIBUTING.md) - setup, tests, migrations, benchmark checks, PR checklist
- [`../SECURITY.md`](../SECURITY.md) - private vulnerability reporting and deployment assumptions
- [`../CODE_OF_CONDUCT.md`](../CODE_OF_CONDUCT.md) - participation standards and reporting
- [`../CHANGELOG.md`](../CHANGELOG.md) - notable changes and known limitations
- [`github-repository.md`](github-repository.md) - About description, topics, and social-preview guidance
