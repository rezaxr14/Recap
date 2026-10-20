# Documentation index

The README is the front door; this page is the map. If a document and the code disagree,
the code and its tests are authoritative and the document should be fixed in the same change.

## Start here

| Document | What it answers |
|---|---|
| [`../README.md`](../README.md) | What RECAP is, what works, quickstart, measured results, known limits |
| [`../IMPLEMENTATION_STATUS.md`](../IMPLEMENTATION_STATUS.md) | Current implementation state, evidence, review follow-up, and remaining gaps |
| [`../NEXT_STEPS.md`](../NEXT_STEPS.md) | The staged plan for real models, benchmark v2, RAG depth, packaging, and the web app |
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
| [`results/2026-10-20-dev.md`](results/2026-10-20-dev.md) | Measured dev-split matrix and caveats |
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
