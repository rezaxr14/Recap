# Drift Detection and Memory Degradation

## 1. The problem

A memory is validated against one environment state. Environments change. Tools rename
parameters, schemas tighten, endpoints disappear, and the same failure message starts
meaning something different. A memory that was correct yesterday can be actively harmful
today, and the harm is silent: the memory still retrieves well because its *text* is
unchanged.

RECAP handles this in two layers:

1. **Structural drift** — the environment fingerprint or tool schema hashes changed.
   Detected deterministically at retrieval time and when an environment is observed.
2. **Empirical drift** — the memory is still structurally plausible but stops working.
   Detected from verified outcomes and a Beta-Binomial reliability posterior.

Neither layer rewrites memory content. Both can only *reduce* trust and *queue work*.

## 2. Environment fingerprints

`EnvironmentFingerprint` is a canonical hash of:

- environment id, type, and version
- each tool's name, version, schema hash, and side-effect class
- the API schema
- the task-suite version

```
fingerprint = sha256(canonical_json({environment_id, environment_type,
                                     environment_version, tools, api_schema,
                                     task_suite_version}))
```

The mock API demonstrates this concretely:

| Version | Parameter | Schema hash |
|---|---|---|
| v1 | `cursor` | hash of V1 schema |
| v2 | `page_token` | hash of V2 schema |

Changing `cursor` -> `page_token` changes the tool schema hash, which changes the
fingerprint. That single fact is what blocks the stale memory in demo step 7.

## 3. Compatibility states

| Status | Meaning | Operational selection |
|---|---|---|
| `compatible` | fingerprint matches, schemas verified | allowed |
| `unknown` | no fingerprint or no observed environment | allowed only if the rest of the policy passes; carries a penalty |
| `suspect` | fingerprint changed but schemas appear compatible, or a prior suspect verdict | rejected by default |
| `incompatible` | schema conflict, or the memory is quarantined/retired | always rejected |

`evaluate_compatibility` rules, in order:

1. missing memory fingerprint or missing observed fingerprint -> `unknown`
2. fingerprints equal -> `compatible`
3. else compare declared tool schema hashes with observed:
   - any mismatch -> `incompatible` (+ `tool_schema_mismatch:<tool>`)
   - no mismatch but some declared tool unobserved -> `suspect`
     (+ `tool_schema_unverified`)
   - otherwise -> `suspect` (+ `environment_fingerprint_changed`)
4. memory `quarantined` -> `incompatible` (+ `memory_quarantined`)
5. memory `retired` -> `incompatible` (+ `memory_retired`)

Results are persisted per `(memory_id, memory_version, environment_fingerprint)`. This
matters: a memory can be incompatible with v2 and perfectly compatible with v1 at the same
time, and a single global status flag would lose that.

## 4. Retrieval-time combination

At retrieval, `combine_compatibility` merges the stored verdict with a live fingerprint
comparison:

- stored `incompatible` wins (never resurrected by a matching live fingerprint)
- live fingerprint mismatch -> `suspect` (or `suspect` with a note if all declared schema
  hashes match)
- stored `suspect` -> `suspect`
- live match -> `compatible`
- no memory fingerprint -> `unknown`

Then eligibility hard-rejects `incompatible`, and `suspect` unless
`allow_suspect_operational_memories=true`. The utility policy also applies a -0.25 weight to
the compatibility penalty, so a suspect memory cannot win on relevance alone even if
somebody flips the flag.

## 5. Empirical drift and degradation

Reliability is a Beta-Binomial posterior per memory:

```
alpha = prior_alpha + verified_successes
beta  = prior_beta  + verified_failures
mean  = alpha / (alpha + beta)
```

Only `verified_success` / `verified_failure` count. A `partial` or `unknown` outcome is
recorded (for audit and for the "we don't know" rate) but does not move the posterior. This
prevents optimistic self-reports from inflating reliability.

Degradation rules (`_maybe_degrade`, only applies to `active` memories):

| Condition | Target |
|---|---|
| high-severity verified failure AND mean <= quarantine threshold | `quarantined` |
| >= `minimum_verified_outcomes_for_alert` (5) AND mean < suspect threshold (0.55) | `suspect` |
| consecutive verified failures >= 3 | `suspect` |

On degradation:

1. transition with an explicit reason
2. reindex (so the index status reflects the new state)
3. write an audit event with before/after
4. enqueue a revalidation job (`reliability_drop`), deduplicated

The defaults are intentionally conservative: a single failure does not degrade a memory.
Three consecutive verified failures, or a statistically meaningful drop after at least five
observations, does.

## 6. Incompatible-environment handling

`on_incompatible_environment(memory, environment)` is the explicit entry point for "we
observed this memory against an environment it does not match":

1. evaluate and persist compatibility
2. if `incompatible` and the memory is `active`: transition `active -> suspect`
3. reindex and audit (`transition:active->suspect`)
4. enqueue a revalidation job (`environment_changed`, priority 1)

What it deliberately does **not** do:

- auto-retire the memory (loses the lesson)
- rewrite the memory content (destroys audit evidence)
- create a replacement automatically (that requires a new validated repair)

The old memory stays in the database with its full history, marked `suspect`, and a v2
memory is produced through the normal compile -> validate -> promote path. Demo step 8 is
exactly this.

## 7. Revalidation

A revalidation job is a durable record that a memory needs to be re-checked against a target
environment.

```
queued -> running -> passed | failed | inconclusive
                  -> blocked | cancelled
```

`dedupe_key = "{memory_id}:v{version}:{fingerprint}:{plan}"` with a unique constraint, so
repeated observations of the same incompatibility do not flood the queue.

`run_revalidation_job(job_id, validator)`:

1. mark `running`, increment attempts
2. load the memory
3. call the validator against the target environment
4. `passed` -> if the memory is `suspect`, transition back to `active` and reindex
5. `failed` -> job stays failed; the memory is not promoted by this path
6. unexpected exception -> `inconclusive` with the error recorded

Note the asymmetry: a passing revalidation can restore `suspect -> active`; a failing
revalidation does **not** escalate to `quarantined` on its own. Escalation comes from
verified outcomes, because a single inconclusive sandbox run is weaker evidence than real
failures.

## 8. Why not auto-heal?

Automatic rewriting is tempting and dangerous:

- it destroys the evidence of what actually worked
- it can silently change semantics of a memory already in use
- it creates a feedback loop where the system grades its own homework

RECAP's rule is: **trust can be reduced automatically; trust can only be restored by
validation.** That is why degradation is automatic and re-promotion requires a passing
validation record.

## 9. Observability

Everything drift-related is queryable:

- `GET /api/v1/memories/{id}/compatibility` — all persisted verdicts for a memory
- `recap memory compatibility <id> --environment-id ...` — evaluate on demand
- `retrieval_decisions.decision_json` — the compatibility status and reason per candidate
- `audit_events` — every transition with before/after and reason
- `revalidation_jobs` — queue depth, state, attempts, errors

A `suspect` memory showing up in a decision record always has a reason string attached
(`suspect_compatibility`, `environment_fingerprint_mismatch`,
`tool_schema_mismatch:list_records`, ...). Nothing is silently filtered.

## 10. Limitations

- Only one drift taxonomy is exercised end to end (renamed parameter / schema hash change).
  Auth expiry, endpoint removal, and semantic changes in error codes are modelled by the
  same fingerprint mechanism but not covered by dedicated tests.
- No automatic revalidation scheduler; jobs are processed on demand.
- No time-based expiry. `age_review` is an enum value but no policy schedules it; recency
  only affects utility scoring (90-day half-life).
- Compatibility is keyed by fingerprint, not by a full environment version graph, so
  transitive compatibility is not modelled.
