# Contributing to RECAP

Thanks for taking the time to improve RECAP. This repository values reproducible evidence
over confident prose: if a claim is not covered by a test, a benchmark report, or documented
reasoning, it should be labelled as a limitation.

## Before you start

- Read [`README.md`](README.md), especially [Known limits](README.md#known-limits).
- Read [`docs/architecture.md`](docs/architecture.md) and
  [`docs/data-model.md`](docs/data-model.md) before changing persistence or retrieval.
- For security issues, follow [`SECURITY.md`](SECURITY.md) instead of opening a public issue.
- Participation is governed by [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).
- Keep pull requests focused. A behavioural change, its tests, and its documentation belong
  in the same change.

## Development setup

Requirements: Python 3.12+, Docker, and a POSIX shell or PowerShell.

```bash
docker compose up -d postgres elasticsearch
python -m venv .venv
# Windows: .venv\Scripts\activate
# Unix:    source .venv/bin/activate
python -m pip install -e ".[dev]"
# Windows PowerShell: Copy-Item .env.example .env
# Unix:              cp .env.example .env
recap db migrate
recap search init
recap health
python scripts/run_smoke_test.py
```

The local stack uses PostgreSQL on `localhost:5433` and Elasticsearch on
`localhost:9200`. The Compose file disables Elasticsearch security for local development;
do not expose that stack to an untrusted network.

If `make` is unavailable (common on Windows), use the equivalent targets in
`python scripts/dev_targets.py --help` or run the commands directly.

## Tests and checks

```bash
ruff check src tests scripts
ruff format --check src tests scripts
mypy src

pytest tests/unit                  # no services required
pytest --run-integration           # real Postgres + Elasticsearch
```

Integration tests are opt-in. They use an isolated database (`recap_test`) and an isolated
index prefix (`recap-test-*`), so they must not touch development data. Recreate test indices
after changing Elasticsearch mappings; the test fixtures validate the mapping on startup.

Live model-endpoint tests are separately gated and skip when no endpoint is configured:

```bash
RECAP_RUN_PROVIDER_INTEGRATION=1 pytest tests/integration/test_live_providers.py --run-integration
```

Mocked transport tests are useful for wire-format coverage, but they are not evidence of a
working live provider. Say so explicitly in the pull request if that is all you ran.

## Benchmark changes

The benchmark is deterministic by construction: a unique workspace per run, a reset before
and after, content-addressed chunk ids, and stable tie-breaks. If you change retrieval,
chunking, the dataset, or scoring, run the reproducibility check:

```bash
python scripts/run_benchmark_repro.py
```

That runs the dev benchmark twice in temporary directories and compares the summaries. The
equivalent manual form is:

```bash
recap eval run --split dev --output benchmark/reports/repro-a
recap eval run --split dev --output benchmark/reports/repro-b
python scripts/check_benchmark_repro.py benchmark/reports/repro-a benchmark/reports/repro-b
```

Update [`docs/results/2026-10-20-dev.md`](docs/results/2026-10-20-dev.md) only with measured
output. Do not turn a degenerate result into a superiority claim. If a dataset change is
intended to make an ablation meaningful, include the new dataset and the new report in the
same pull request.

## Code conventions

- Keep the domain layer pure. Infrastructure adapters own I/O; domain models do not import
  SQLAlchemy, Elasticsearch, FastAPI, or HTTP clients.
- Prefer explicit typed models over dictionaries at boundaries. `mypy src` must stay clean.
- Add or update a test for every behavioural change. Reproduce bugs before fixing them when
  practical, and include the regression test in the same commit.
- No secrets in code, tests, fixtures, logs, or reports. Use `.env` locally and the documented
  environment variables in CI.
- Migrations are append-only. Add a new Alembic revision; never edit an applied revision.
  Test both upgrade paths when a migration touches existing data.
- Keep line endings LF (see `.gitattributes`). Do not commit caches, virtual environments,
  generated benchmark reports, or `demo_report.md`.
- Documentation is part of the change: update the README, `docs/`, and
  `IMPLEMENTATION_STATUS.md` when behaviour or evidence changes.

## Database migrations

```bash
recap db migrate
recap db current
```

New revisions belong in `migrations/versions/` and must have a single clear `down_revision`.
If a migration adds a column used by retrieval, update the corresponding Elasticsearch
mapping validation and the migration test in the same pull request.

## Pull request checklist

- [ ] The change is focused and explained in the pull request description.
- [ ] `ruff check src tests scripts` passes.
- [ ] `ruff format --check src tests scripts` passes.
- [ ] `mypy src` passes.
- [ ] `pytest tests/unit` passes.
- [ ] `pytest --run-integration` passes if services or persistence changed.
- [ ] New behaviour has a regression test.
- [ ] Benchmarks were re-run and the reproducibility check passes if retrieval, chunking, or
      scoring changed.
- [ ] Documentation and `IMPLEMENTATION_STATUS.md` reflect the new state.
- [ ] No secrets, generated reports, or local environment files are included.
- [ ] The pull request states which limitations remain.

## Reporting bugs

Use the GitHub issue templates. Include the commit, operating system, Python version, the
commands you ran, and the smallest reproduction you can produce. Redact API keys, tokens,
connection strings, and any recorded agent content that contains sensitive data.
