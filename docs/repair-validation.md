# Repair Validation

## 1. What a repair is

A repair is **not** a patch, a script, or generated code. It is a typed object:

```json
{
  "id": "rc_...",
  "source_run_id": "run_...",
  "workspace_id": "ws_...",
  "task_family": "api_pagination",
  "risk_level": "low",
  "environment_id": "mock-api-v1",
  "environment_fingerprint": "sha256:...",
  "diagnosis": {
    "failure_category": "expired_pagination_cursor",
    "hypothesis": "The pagination cursor expired; restart pagination from the beginning.",
    "evidence_event_ids": ["evt_..."],
    "confidence": 0.7
  },
  "failure_window": {
    "normalized_signature": "ValidationError|INVALID_CURSOR|list_records",
    "error_code": "INVALID_CURSOR",
    "tool_name": "list_records",
    "event_ids": ["evt_..."]
  },
  "preconditions": [ ... ],
  "strategy": {
    "summary": "Restart pagination with a fresh cursor",
    "actions": [
      {
        "action_type": "refresh_pagination_cursor",
        "parameters": {"parameter": "cursor"},
        "side_effect_class": "read_only"
      }
    ],
    "action_signature": "refresh_pagination_cursor",
    "side_effect_class": "read_only",
    "limitations": ["..."]
  },
  "postconditions": [ ... ]
}
```

`ActionType` is a closed enum:

```
retry_with_backoff
refresh_pagination_cursor
map_parameter
request_missing_field
verify_resource_exists
halt
```

RECAP never executes these. It hands them to the host agent as structured guidance, and the
agent interprets only allowlisted actions. There is no code generation and no eval.

## 2. Pipeline

```
run events
   |
   v
[extract_failure_window]  bounded to N events, normalized signature, error code
   |
   v
[CandidateGenerator]        deterministic templates or configured repair LLM
   |
   v
[validate_draft]          schema, bounds, evidence linkage, secret check
   |
   v
[StaticValidator]         no execution; structural and safety checks
   |
   v
[MockSandboxValidator]    real execution in a disposable environment
   |
   v
[PromotionService]        the only path to `active`
```

Each stage may reject. Rejection is a recorded outcome, not an error to hide.

## 3. Failure window extraction

`extract_failure_window`:

1. finds the last `tool_call_failed` event in the run
2. walks backward to include the causal lead-up (request, prior successful calls, state
   snapshots) up to `max_failure_window_events` (default 20)
3. normalizes the failure signature from error type, error code, and tool name
4. carries the error code forward so validators can reproduce the *same* failure

Normalization (`recorder/normalization.py`) strips volatile parts (ids, timestamps, cursor
values, quoted payloads) so that two instances of the same failure map to one signature. This
is what makes a memory reusable across runs rather than tied to one literal message.

## 4. Candidate generation

`DeterministicGenerator` maps failure categories to typed strategies:

| Failure category | Strategy |
|---|---|
| expired/invalid cursor | `refresh_pagination_cursor` |
| renamed parameter (`UNKNOWN_PARAMETER`) | `map_parameter` |
| rate limit / transient 5xx | `retry_with_backoff` |
| missing required field | `request_missing_field` |
| not found | `verify_resource_exists` |
| unrecognized | no candidate |

The default generator is deterministic by design. The optional LLM generator
(`REPAIR_GENERATOR=llm`) calls the configured `repair_llm` role and receives only bounded,
already-redacted events from the failure window. It is behind the same `CandidateGenerator`
protocol and the same validation contract: model output is untrusted until it passes schema,
real-event-ID, and execution checks. The model has no field that can claim validation passed
and no path to promotion.

`validate_draft` rejects drafts that:

- reference evidence events not in the run or outside the failure window
- use a non-allowlisted action type
- exceed `max_strategy_actions`
- omit preconditions or postconditions
- contain a detectable secret

## 5. StaticValidator

Checks, without executing anything:

| Check | Meaning |
|---|---|
| `has_source_run` | candidate is traceable to a run |
| `has_evidence` | diagnosis cites event ids |
| `has_preconditions` | applicability is decidable |
| `has_postconditions` | success is checkable |
| `actions_allowlisted` | every action type is in the enum |
| `no_destructive_actions` | no action is classified destructive |
| `no_secret_leakage` | payload scans clean |
| `bounded_strategy` | action count within limit |

Static validation is necessary and insufficient. It cannot tell you the strategy works.

## 6. MockSandboxValidator (real execution)

This is the part that makes validation more than a formality. For each candidate:

1. **Fresh environment.** A new `PaginatedApiEnvironment` is constructed per validation.
   No state leaks between candidates or attempts.
2. **Reproduce the failure.** The sandbox drives the same tool with the failing inputs and
   asserts that it produces the *expected error code derived from the failure window* (not a
   hardcoded one). If it cannot reproduce, the result is `inconclusive` — never `passed`.
3. **Check preconditions.** The candidate's preconditions are evaluated against the
   sandbox's actual state. If they do not hold, the result is `failed` or `blocked`.
4. **Negative guard.** The candidate is evaluated against a *mismatched* schema (the other
   API version). Its preconditions must reject it. A repair that claims to be applicable
   everywhere is rejected here. This is how stale repairs are caught before promotion.
5. **Apply the strategy.** Allowlisted actions are interpreted against the sandbox:
   - `refresh_pagination_cursor` -> drop the cursor and restart
   - `map_parameter` -> use the mapped parameter name
   - `retry_with_backoff` -> retry the call
   - `verify_resource_exists` -> probe existence first
6. **Check postconditions.** The operation must complete and the declared postconditions
   must hold in the sandbox state.
7. **Record provenance.** Environment id/fingerprint, validator version, candidate hash,
   attempt, timings, observed outcome, per-check results.

Failure handling:

| Situation | Status |
|---|---|
| reproduction fails | `inconclusive` |
| precondition violated | `failed` |
| negative guard not rejected | `failed` |
| strategy raises | `failed` (with observed error) |
| postcondition violated | `failed` |
| validator raises unexpectedly | `blocked` |
| everything passes | `passed` |

There is no path from an exception to `passed`.

## 7. Promotion gate

`PromotionService.evaluate` re-checks, at promotion time:

| Check | Failure mode it prevents |
|---|---|
| `validation_passed` | promoting an unvalidated or failed candidate |
| `evidence_present` | promoting a memory with no traceable cause |
| `preconditions_present` | promoting an unapplicable rule |
| `postconditions_present` | promoting an unverifiable rule |
| `environment_recorded` | promoting a memory with no origin |
| `no_secret_leakage` | promoting leaked credentials into the index |
| `no_destructive_actions` | promoting a destructive strategy |
| `no_duplicate_active_memory` | duplicate active content per workspace |
| `validation_required` | bypassing validation when config requires it |

Additionally, `promote` refuses a candidate already in `PROMOTED` state (found by test: it
previously allowed double promotion).

Promotion then:

1. builds `MemoryContent` from the candidate
2. creates the memory directly in `active` with provenance
3. updates the candidate to `promoted` with the memory id
4. writes an audit event
5. indexes the memory through the outbox-backed synchronizer

If indexing fails, the memory remains durable in Postgres with `index_status=failed` and an
outbox row pending. Promotion is not rolled back because the durable state is correct; the
index is rebuildable.

## 8. Why not execute in the real environment?

Because validation must be safe, repeatable, and side-effect free. The mock sandbox gives:

- deterministic reproduction
- no network, credentials, or irreversible operations
- the ability to run a *negative* guard (a real environment would be mutated)
- cheap repeated trials

The trade-off is explicit: the sandbox proves the strategy is coherent against a
schema-compatible environment, not that it is correct against every production quirk. That
is why memories carry `known_limitations`, environment fingerprints, and a reliability
posterior that degrades after real-world failures.

## 9. Tests that matter

- `tests/unit/test_repair_compiler.py`
  - draft with non-allowlisted action is rejected
  - draft referencing a foreign event id is rejected
  - static validator fails on missing preconditions
  - sandbox reproduces the failure and passes the real strategy
  - sandbox rejects a stale v1 repair against the v2 schema (negative guard)
  - promotion is refused without a passed validation
  - double promotion is refused
- `tests/integration/test_api.py`
  - compile -> validate -> promote through HTTP
  - promotion of an unvalidated candidate returns a structured rejection
- `tests/e2e/test_lifecycle.py`
  - full run -> compile -> validate -> promote -> retrieve -> apply -> v2 drift
