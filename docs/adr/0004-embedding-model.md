# ADR 0004: Embedding model, prefixes, fingerprint, and cache

- **Status:** Accepted
- **Date:** 2026-10-19
- **Deciders:** RECAP maintainers

## Context

The vector branch is only meaningful if it uses a real semantic embedding model. The current
default, `HashingEmbeddingProvider`, is a deterministic lexical projection. It keeps the
pipeline usable offline and in CI, but it must never be presented as semantic retrieval.

RECAP has an `openai_compatible` embeddings adapter and a strict dimension check. It does not
need a new provider class or a model runtime. What it needs is a selected, documented model,
correct document/query handling, a stable fingerprint, and a cache so benchmark reruns do not
re-embed the same text.

## Decision

### 1. Defaults and comparison models

| Role | Model | Dimension | Input limit | Notes |
|---|---|---|---|---|
| Hosted default | `text-embedding-3-small` | 1536 (request fewer with `send_dimensions`) | 8191 tokens | Native OpenAI-compatible `/embeddings`; $0.02 / 1M input tokens as announced by OpenAI on 2024-01-25. |
| Open-weight comparison | `BAAI/bge-m3` | 1024 | 8192 tokens | Verified from the official model card; served behind an OpenAI-compatible endpoint (vLLM/TEI/Ollama). RECAP downloads nothing. |
| Optional open-weight | `nomic-embed-text-v1.5` | 768 | 8192 tokens | Requires task prefixes. |
| Always-present baseline | `recap-hashing-embedding-v1` | 256 | none (local) | Explicit non-semantic baseline in every report. |

Verification note (2026-10-19): the BGE-M3 dimension and sequence length were checked on the
official Hugging Face model card. The OpenAI announcement confirms `text-embedding-3-small`
pricing and the `dimensions` shortening feature; the public API reference returned HTTP 403
from this environment, so the 1536/8191 values must be confirmed against the deployed
endpoint (the adapter already rejects any response whose dimension differs from
`EMBEDDING_DIMENSIONS`). Record verified values, provider, and date in benchmark reports.

The model catalog is not a stable API. Changing the model or dimension requires a new index
generation and a full re-embed; there is no in-place vector-field migration.

### 2. Asymmetric document/query text

Models that use task prefixes or instructions need different text for documents and queries.
Add `document_prefix` and `query_prefix` to `EmbeddingSettings` and apply them inside
`embed_documents()` and `embed_query()`. Prefixes are configuration, never persisted into
chunk text. They are part of the model fingerprint, so changing a prefix forces re-embedding.

BGE-M3 does not require prefixes; the defaults are empty. Nomic-style models should set
`search_document: ` and `search_query: `.

### 3. Fingerprint

The embedding fingerprint is a SHA-256 over canonical JSON of:

```
provider, model_id, model_version, dimension, normalize,
document_prefix, query_prefix
```

It is stored on every chunk document and every retrieval decision. Elasticsearch must never
contain vectors from two fingerprints in one field. A fingerprint change requires a new
`index_generation`, a new chunk index, and a full rebuild from Postgres.

### 4. Normalization and dimension safety

The ES mapping uses cosine similarity. The adapter records whether it assumes normalized
vectors (`normalize: true` by default for OpenAI-compatible APIs, which return unit-length
vectors) but does not silently normalize provider output. Any response dimension mismatch is
a hard error, as today.

### 5. Postgres embedding cache (implemented)

Migration `0008_embedding_cache` adds an `embedding_cache` table keyed by
`(embedding_fingerprint, text_hash)` and storing the JSON vector, dimension, provider, model,
document/query kind, creation time, last access time, and hit count. The cache is an
optimization only and is never the source of truth:

- lookups are scoped to the exact fingerprint, so no vector is shared across models,
  dimensions, prefixes, or normalization settings;
- the namespaced text hash includes the document/query role, preventing an asymmetric model's
  query vector from being reused as a document vector;
- a cache hit with the wrong dimension is a hard error, never padded or truncated;
- cache read/write failures fail open and are counted, because an unavailable optimization
  must not make indexing or retrieval unavailable;
- cache misses call the provider once for duplicate texts in a batch, then write vectors in a
  short transaction; the cache is not coupled to the chunk persistence transaction;
- `recap embedding-cache --clear` is the explicit invalidation path; a fingerprint change
  naturally produces different keys and a new index generation still requires a full rebuild;
- benchmark reports count cache hits/misses, provider batches/texts, provider latency, and
  prompt tokens when the endpoint returns usage. Missing usage is reported as `null`, not
  invented.

### 6. No downloads and no silent fallback

- `tiktoken` and Hugging Face snapshot downloads are forbidden on the chunker/embedding path.
- A configured real endpoint that fails returns an error; RECAP does not fall back to hashing
  inside a run. An operator can intentionally select the hashing baseline.
- Production still rejects hashing/fake unless `embeddings.allow_degraded=true`.
- `recap providers` and `recap doctor` report `semantic_embeddings` and the fingerprint.

### 7. Batching and retries

`embed_documents()` keeps the configured batch size and bounded retry policy in
`OpenAICompatibleClient`. The chunk indexer records embedding tokens when the provider returns
usage; if it does not, it records `null`, not an invented count.

## Consequences

### Positive

- The vector branch can be evaluated with a real model while CI remains offline and cheap.
- Prefix and fingerprint handling prevents the most common "mixed vector spaces" failure.
- The cache makes ablation reruns practical and reports hits, provider calls, latency, and
  provider-reported tokens without pretending a missing count is zero.

### Negative

- Hosted and open-weight comparisons need operator-provided endpoints; CI cannot produce a
  semantic claim without one.
- A new fingerprint requires a full reindex.
- The cache adds one table and a write path.

## Alternatives considered

**Use hashing as the only embedder.** Rejected: it cannot demonstrate semantic retrieval and
makes the vector branch redundant with BM25.

**Bundle a sentence-transformers model.** Rejected: violates the no-download rule and bloats
the install.

**Use Elasticsearch-native inference/ELSER.** Deferred: it moves embedding into ES and
conflicts with the app-side embedding and cache design. It remains a separate experiment.

**Normalize all vectors in RECAP.** Rejected for now: it hides provider behavior. Cosine is
scale-invariant, and the fingerprint records the assumption.
