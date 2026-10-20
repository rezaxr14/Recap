# Security policy

RECAP records agent runs and stores retrieved memories, so it can process sensitive tool
inputs and outputs. Please report security problems privately and give the maintainers a
chance to fix them before public disclosure.

## Supported versions

| Version | Supported |
|---|---|
| `0.1.x` | Yes |
| Older versions | No |

The project is alpha software. Security fixes are applied to the current development line
and released as patch versions when practical.

## Reporting a vulnerability

Use GitHub's private vulnerability reporting:

**https://github.com/rezaxr14/Recap/security/advisories/new**

If that form is unavailable, contact the repository owner through their GitHub profile
rather than opening a public issue. Do not include exploit details, credentials, recorded
agent content, or customer data in a public issue.

Please include:

- the affected commit or version;
- the operating system and installation method;
- a minimal reproduction or proof of concept;
- the impact you believe is possible;
- any suggested mitigation;
- whether you intend to disclose publicly and on what timeline.

The maintainers will acknowledge a report as soon as practical, investigate it, and keep the
reporter informed of the remediation and disclosure plan. This is a volunteer project, so no
specific response SLA or bug bounty is offered.

## Scope and threat model

The design assumptions and trust boundaries are documented in
[`docs/threat-model.md`](docs/threat-model.md). In particular:

- The local Docker Compose stack disables Elasticsearch security and uses development
  credentials. It is not safe to expose directly to an untrusted network.
- There is no multi-tenant authentication or authorization layer in this repository. Put the
  API behind your own authenticated gateway before exposing it.
- Recorded events pass through fail-closed redaction, but redaction is defence in depth, not
  a guarantee that arbitrary user content is safe to store or display.
- Provider API keys are read from environment variables and must never be committed. Reports
  and diagnostics are designed not to print secrets; verify this before sharing logs.
- Repair candidates are executed only in the mock sandbox used by the validator. That sandbox
  is not a general-purpose untrusted-code isolation boundary.
- `REPAIR_GENERATOR=llm` and live model endpoints expand the trust boundary to the configured
  provider. Review the provider's data handling before sending real agent content.

## If a secret was exposed

1. Revoke and rotate the credential at the provider immediately.
2. Remove it from the working tree and from the current branch history if it was committed.
3. Assume that anything pushed to a public remote may already have been copied; rotation is
   the only reliable remediation.
4. Report the exposure through the private channel above if it affects other users.
