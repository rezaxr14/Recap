# ADR 0003: Memory promotion requires execution validation through a single gate

- **Status:** Accepted
- **Date:** 2026-10-19
- **Deciders:** RECAP maintainers

## Context

A memory is a durable instruction that a future agent may act on. If it is wrong, the failure
is deferred, silent, and potentially repeated. The system therefore needs an answer to: *what
must be true before a piece of generated recovery knowledge is allowed to influence future
runs?*

Three plausible answers:

1. **Trust the generator.** If an LLM produced it from a real trace, promote it.
2. **Static checks only.** Validate schema, allowlist, and evidence, then promote.
3. **Execute it.** Prove in a safe environment that the strategy resolves the failure.

Option 1 fails on the first hallucination. Option 2 catches malformed candidates but not
wrong ones — a perfectly well-formed strategy that does nothing will pass every structural
check.

## Decision

**Promotion requires a passing execution validation, and there is exactly one promotion
path.**

### 1. Typed, allowlisted repairs only

Repairs are objects with a closed `ActionType` enum and a declared `SideEffectClass`. There
is no code generation, no patch application, and no `eval`. Expressiveness is traded for
safety deliberately.

### 2. Two validators, increasing rigor

- `StaticValidator` — schema, evidence linkage, allowlist, bounded actions, no destructive
  actions, no secret leakage. No execution.
- `MockSandboxValidator` — constructs a fresh disposable environment, reproduces the
  failure with the **error code derived from the failure window**, checks preconditions,
  runs a **negative guard** against a mismatched schema, applies the strategy, and verifies
  postconditions.

The negative guard is the key control: a candidate whose preconditions do not reject an
incompatible environment fails validation. That is what catches stale repairs before they
are promoted.

### 3. A single gate

`PromotionService` is the only code path that creates an `active` memory from a candidate.
It re-evaluates every requirement at promotion time, including validation, evidence,
preconditions, postconditions, environment, secret leakage, destructive actions, and
duplicate content. A candidate that was valid when validated but edited afterward is caught
by re-evaluation (validation records also bind `candidate_hash`).

### 4. Failure is not success

Validator exceptions, timeouts, and irreproducible failures produce `blocked`,
`inconclusive`, or `failed` — never `passed`. Promotion refuses anything without a `passed`
record when `require_validation_for_promotion=true` (the default).

### 5. Validation does not auto-heal

A passing revalidation can restore `suspect -> active`. A failing revalidation does not
rewrite or retire the memory; degradation comes from verified real-world outcomes, and
replacement comes from compiling and promoting a new candidate.

## Consequences

### Positive

- **Wrong-but-well-formed repairs are caught.** Execution + negative guard is strictly
  stronger than static checks.
- **No code execution surface.** Allowlisted typed actions mean a malicious candidate cannot
  run arbitrary code.
- **Traceability.** Every promoted memory links to a source run, evidence events, and
  validation records with hashes.
- **No bypass.** Tests assert that unvalidated promotion and double promotion fail.
- **Safe iteration.** The sandbox is disposable and deterministic, so validation is cheap and
  repeatable.

### Negative

- **Sandbox fidelity is a ceiling.** A strategy that works in the mock environment may fail
  in production. Mitigated by reliability tracking and drift detection, not eliminated.
- **Only one environment family is covered.** The negative guard is meaningful because two
  schema versions exist; broader drift taxonomies need more environments.
- **Deterministic generation is narrow.** The generator covers a handful of failure
  categories. An LLM generator could produce richer candidates, but it would not change the
  validation contract.
- **Validation cost.** Every candidate needs at least two validation runs (positive and
  negative guard) before promotion.

### Neutral

- Manual memory creation exists for operators and imports. It records provenance and is
  subject to the same lifecycle table, but it does not require a repair candidate. This is a
  deliberate escape hatch, documented rather than hidden.

## Alternatives considered

**Promote on generation (option 1).** Rejected: unvalidated LLM output as durable operational
policy is the failure mode the project exists to prevent.

**Static validation only (option 2).** Rejected: it cannot distinguish a working strategy
from a plausible no-op, and it cannot detect staleness against a changed schema.

**Execute in the real environment.** Rejected: unsafe, non-repeatable, potentially
destructive, and impossible to run a negative guard without mutating production.

**Human review for every promotion.** Valuable for high-risk memories and supported via
`risk_level` and the reviewer field, but requiring it for every candidate makes the system
unusable at any volume. The gate is automated; human review is an escalation, not the
baseline.

**LLM-as-judge validation.** Rejected as the *primary* validator: a model grading a model's
output is not execution evidence. It could be added as an additional signal, never as a
replacement for the sandbox.

## References

- `src/recap/repair_compiler/validation.py`
- `src/recap/repair_compiler/promotion.py`
- `src/recap/repair_compiler/candidate_generation.py`
- `src/recap/domain/enums.py` (`ActionType`, `SideEffectClass`)
- `tests/unit/test_repair_compiler.py`
- `docs/repair-validation.md`
