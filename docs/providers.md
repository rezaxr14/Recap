# Model providers

RECAP is **API-first**. Real inference and real embeddings run against endpoints that are
already running and configured by the operator. RECAP does not install, download, quantize,
or launch models, and it never silently replaces a failed real provider with a fake one.

## Roles

| Role | Config section | Purpose | Failure behavior |
|---|---|---|---|
| Agent LLM | `llm` | Tool-calling agent loop | Role is disabled until configured; a configured but unreachable endpoint raises |
| Repair LLM | `repair_llm` | Typed repair-candidate proposals | `repair_compiler.generator=llm` refuses to start without it |
| Embeddings | `embeddings` | Dense retrieval vectors | Indexing records retryable outbox intent and fails visibly |
| Reranker | `recall.reranker` | Order-only reranking | Currently `lexical` or `none`; no model download |

The embedding role is mandatory for semantic retrieval. The two chat roles are independent:
the repair compiler can run with an LLM while the agent uses a different model, or either can
be disabled.

## API contract

All remote providers use the OpenAI-compatible wire format:

- `GET /models` for discovery and health (optional; a 404 is reported as "not supported")
- `POST /chat/completions` for chat, tool calls, and structured output
- `POST /embeddings` for document and query vectors

`base_url` is the API root (for example `https://api.example.com/v1`). A full operation URL
such as `.../v1/chat/completions` is rejected so path duplication cannot happen silently.
Authentication uses `Authorization: Bearer <key>` when a key is configured.

Structured generation supports two explicit modes:

- `json_schema` (default): sends `response_format={"type":"json_schema", ...}`.
- `json_object`: sends `response_format={"type":"json_object"}`.

There is no implicit downgrade between them. If the endpoint rejects the configured mode,
the request fails with a `provider_response_error`.

## Environment variables

Unprefixed names are the primary interface; `RECAP_`-prefixed aliases remain for older
deployments and take precedence.

### Embeddings

| Variable | Meaning |
|---|---|
| `EMBEDDING_PROVIDER` | `openai_compatible`, `sentence-transformers`, `hashing`, or `fake` |
| `EMBEDDING_BASE_URL` | API root, required for `openai_compatible` |
| `EMBEDDING_API_KEY` | Bearer token; never logged |
| `EMBEDDING_MODEL` | Real model identifier |
| `EMBEDDING_MODEL_VERSION` | Operator-supplied version string |
| `EMBEDDING_DIMENSIONS` | Vector dimension; also sets `elasticsearch.vector_dimension` unless that is set explicitly |
| `EMBEDDING_BATCH_SIZE` | Inputs per request |
| `EMBEDDING_TIMEOUT_SECONDS` | Per-request timeout |
| `EMBEDDING_MAX_RETRIES` | Retries for 429/5xx/transport errors |
| `EMBEDDING_SEND_DIMENSIONS` | Send the optional `dimensions` request field |
| `EMBEDDING_MODEL_PATH` | Existing local directory for `sentence-transformers` |
| `EMBEDDING_CACHE_ENABLED` | Use the Postgres embedding cache (`true` by default) |
| `EMBEDDING_ALLOW_DEGRADED` | Explicitly allow hashing/fake in production |
| `RECAP_EMBEDDING_PROVIDER`, `RECAP_EMBEDDING_DIMENSION`, `RECAP_EMBEDDING_CACHE_ENABLED` | Aliases |

### Chat roles

Use the same suffixes for each role:

```
LLM_PROVIDER LLM_BASE_URL LLM_API_KEY LLM_MODEL LLM_TIMEOUT_SECONDS
LLM_MAX_RETRIES LLM_TEMPERATURE LLM_MAX_TOKENS LLM_STRUCTURED_MODE

REPAIR_LLM_PROVIDER REPAIR_LLM_BASE_URL REPAIR_LLM_API_KEY REPAIR_LLM_MODEL
REPAIR_LLM_TIMEOUT_SECONDS REPAIR_LLM_MAX_RETRIES REPAIR_LLM_TEMPERATURE
REPAIR_LLM_MAX_TOKENS REPAIR_LLM_STRUCTURED_MODE
```

`provider` is `disabled` or `openai_compatible`. Select the LLM repair generator with
`REPAIR_GENERATOR=llm` (or `repair_compiler.generator: llm`). Startup fails if that generator
is selected without a configured `repair_llm` role.

A minimal real configuration:

```bash
export EMBEDDING_PROVIDER=openai_compatible
export EMBEDDING_BASE_URL=https://api.example.com/v1
export EMBEDDING_API_KEY=...
export EMBEDDING_MODEL=text-embedding-3-small
export EMBEDDING_DIMENSIONS=1536

export LLM_PROVIDER=openai_compatible
export LLM_BASE_URL=https://api.example.com/v1
export LLM_API_KEY=...
export LLM_MODEL=gpt-4.1-mini

export REPAIR_LLM_PROVIDER=openai_compatible
export REPAIR_LLM_BASE_URL=https://api.example.com/v1
export REPAIR_LLM_API_KEY=...
export REPAIR_LLM_MODEL=gpt-4.1-mini
export REPAIR_GENERATOR=llm
```

The dimension must match the Elasticsearch mapping. Changing the embedding model or
dimension requires a new index generation; RECAP fails before creating or writing an
incompatible index instead of mixing vector spaces.

## Local and degraded providers

- `hashing` is a deterministic lexical feature hash. It is useful for offline development and
  tests, but it is **not** a neural semantic model. It reports `semantic_embeddings=false`,
  `semantic_retrieval=false`, and a `degraded_mode` reason.
- `fake` is for unit tests only. It produces synthetic vectors and must never be used for
  real-model integration/e2e acceptance.
- `sentence-transformers` loads an existing `model_path` directory with
  `local_files_only=True`. If the path does not exist, RECAP raises; it never downloads
  weights. This adapter therefore requires the operator to provision the model directory by
  their own process.
- Production (`app.environment: production`) rejects `hashing`/`fake` embeddings unless
  `embeddings.allow_degraded=true` is set explicitly. That flag is an acknowledgement, not a
  performance claim.

## Agent tool loop

`recap agent run` is the real-model agent path:

```bash
recap agent run --environment-version 2 --workspace live-agent --max-tool-calls 6
recap agent run --environment-version 1 --memory-id mem_...   # inject one advisory memory
```

The configured `llm` role receives the environment's declared tool schema and chooses tool
calls. RECAP validates the tool name and arguments against that schema, executes only the
allowlisted local tool, and returns the real result. The model cannot execute code, call an
undeclared tool, widen permissions, or claim success: the run is successful only when every
record was returned by real tool calls. Provider/transport failures propagate and the run is
not fabricated.

The recorded run stores the provider, model, prompt version, max tool budget, token usage,
tool call/retry counts, and every tool request/result (redacted). An injected memory is
rendered as bounded advisory JSON with an explicit "untrusted data, not instructions"
preamble; only typed action fields are advisory. The demo and benchmark still use the
deterministic reference agent for reproducibility, so their numbers are not real-model
numbers.

## Inspecting providers

```bash
recap providers          # offline: roles, models, capabilities; no secrets, no network
recap doctor             # database + Elasticsearch + configured endpoint health
recap doctor --live      # adds a small real chat call and a real embedding call
```

`recap providers` does not probe the network. `recap doctor` probes only the configured
endpoints; it does not scan the network or enumerate unrelated services. A failed configured
endpoint makes `doctor` exit non-zero with an actionable error.

Health output never includes API keys, authorization headers, or credential-bearing URLs.

## Live verification

Unit tests use `httpx.MockTransport` and prove wire format, retries, auth mapping, dimension
checks, and fail-closed behavior without network access.

Real endpoints are covered by opt-in tests:

```bash
export RECAP_RUN_PROVIDER_INTEGRATION=1
export RECAP_RUN_INTEGRATION=1
pytest tests/integration/test_live_providers.py --run-integration
```

The suite skips clearly when a role is not configured. It covers:

1. endpoint reachability and model discovery,
2. a small chat completion,
3. schema-constrained structured output,
4. the real embedding dimension,
5. indexing and vector retrieval in Elasticsearch with the real embedding,
6. the real chat model driving at least one validated tool call through the mock API, with the
   run persisted in PostgreSQL.

CI does not set the opt-in flag, so these tests skip there. They are the only acceptable
evidence for "real model integration works"; mocked tests and the deterministic demo are not.

## Operational failure modes

| Symptom | Behavior | Action |
|---|---|---|
| `provider_auth_error` | Request rejected with 401/403 | Check `*_API_KEY`; the key is not echoed |
| `provider_unavailable` | 429/5xx/transport after bounded retries | Check endpoint/network; outbox keeps indexing intent |
| `provider_response_error` | Malformed JSON, truncated completion, bad tool arguments, dimension mismatch | Inspect payload and model compatibility |
| `responses are degraded` | Hashing/fake embeddings | Configure a real embedding endpoint |
| `dimension mismatch before indexing` | Config error | Align `EMBEDDING_DIMENSIONS` with the model and use a new index generation |

Embedding failure during promotion does not drop the memory: the outbox intent is written
first, PostgreSQL remains the source of truth, and the index is marked failed for retry.

## Security notes

- Provider credentials are read from environment/secrets and excluded from `safe_dump`,
  logs, diagnostics, and CLI output.
- Prompts include only bounded, already-redacted events from the failure window. Evidence is
  data, not instructions; the model cannot widen permissions.
- Model output is a hypothesis. It must pass the same typed schema, real-event-ID check, and
  sandbox validation as deterministic drafts before promotion.
- Sending trace evidence to a remote provider is a data-processing decision for the operator.
  Use a provider and retention policy that match your data classification. See
  [`threat-model.md`](threat-model.md).
