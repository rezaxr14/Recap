# Phase 0 audit evidence

- **Date:** 2026-10-20
- **Commit audited:** `7a529e4` (`feat: harden chunking and add embedding cache`)
- **Environment:** Windows 11, Python 3.12.3, Docker Desktop, PostgreSQL 17-alpine,
  Elasticsearch 8.19.3, LM Studio at `http://localhost:1234/v1`

This file records what was run, not projections. No model weights were downloaded by the
agent.

## 1. Fresh-clone quickstart

Cloned the repository to `%TEMP%\recap-phase0`, created `.venv`, installed
`pip install -e ".[dev]"`, and ran the README sequence:

```text
recap version                  -> recap 0.1.0
recap db migrate               -> 0007_chunk_strategy -> 0008_embedding_cache
recap search init              -> recap-memories-v1 and recap-chunks-v2, aliases present
recap health                   -> database ok; Elasticsearch 8.19.3 green
python scripts/run_smoke_test.py
                               -> all 9 checks passed
```

The smoke test covered the recorded failure, compiled/promoted v1 memory, fail-closed v2
rejection, fresh v2 memory retrieval, and the no-memory baseline.

### Deviation found and documented

`docker compose up -d postgres elasticsearch` from the second checkout failed because the
first checkout's stack owns the fixed container names `recap-postgres` and
`recap-elasticsearch` and the host ports. The audit reused the already-running stack rather
than deleting the operator's development data. The README now tells operators to stop the
other checkout first, and notes that fixed names/ports prevent two simultaneous checkouts.

### Documentation fix

The model-configuration steps used Unix `export` syntax only. README now gives PowerShell
and Bash/Zsh variants.

## 2. Live provider probe

Command (environment values only; no key is printed):

```powershell
recap doctor --live
```

Observed:

- database: ok
- Elasticsearch: 8.19.3, `green`
- agent LLM: `llama-3.2-3b-instruct`, chat probe ok
- repair LLM: `llama-3.2-3b-instruct`, chat probe ok
- embeddings: `text-embedding-nomic-embed-text-v1.5`, configured dimension 768, live
  embedding returned 768 dimensions
- `semantic_retrieval: true`
- prompts use nomic task prefixes:
  - document: `search_document: `
  - query: `search_query: `

Raw output: [`phase0-doctor-live.txt`](phase0-doctor-live.txt).

## 3. Opt-in live provider suite

```powershell
$env:RECAP_RUN_PROVIDER_INTEGRATION = "1"
pytest tests/integration/test_live_providers.py -q --run-integration
```

Result: `6 passed in 59.73s`.

The suite exercised the real chat endpoint, structured JSON, a 768-dimensional embedding,
real embeddings indexed and retrieved through Elasticsearch, and the tool-using agent against
the mock API with real PostgreSQL recording.

Raw output: [`phase0-live-provider-tests.txt`](phase0-live-provider-tests.txt).

## 4. Capability-claim corrections

- README verification count changed from the stale `238 passed` to `245 passed, 1 skipped`.
- The embedding cache is implemented; older "not implemented / no cache metric" claims were
  removed from README-facing status and the capability table.
- Chat/embedding/tool claims now distinguish **local live evidence** from **CI coverage**.
- LLM repair-candidate generation remains **partial**: it is wired and unit-tested, but live
  candidate-generation/validation acceptance has not run.

## 5. Human actions before Phase 1

1. Decide the agent model. The plan recommends `Qwen3-4B-Instruct-2507`; the current LM
   Studio endpoint serves `llama-3.2-3b-instruct`, which passes the live suite but is older
   and smaller. The human must either confirm the substitute or download the recommended
   model.
2. If the cross-encoder reranker is in scope, download it locally (the agent must not
   download weights), for example:
   `cross-encoder/ms-marco-MiniLM-L-6-v2` into `models/ms-marco-MiniLM-L-6-v2`.
3. No GitHub Pages or other hosting action is needed yet; Phase 6 will require enabling Pages.
