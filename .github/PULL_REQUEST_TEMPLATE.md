# Pull request

## What changed

<!-- A short, concrete description. Link the issue if one exists. -->

## Why

<!-- The problem, the evidence that it is real, and why this approach is appropriate. -->

## How it was tested

<!-- Commands and results. Include the commit or environment when useful. -->

- [ ] `ruff check src tests scripts`
- [ ] `ruff format --check src tests scripts`
- [ ] `mypy src`
- [ ] `pytest tests/unit`
- [ ] `pytest --run-integration` (if persistence, retrieval, chunking, or scoring changed)
- [ ] Benchmark reproducibility check (if retrieval, chunking, dataset, or scoring changed)

## Evidence

<!-- Paste measured output or link a committed report. Do not claim improvements that the data does not show. -->

## Limitations

<!-- What this change does not solve, and what claims should not be made from it. -->

## Checklist

- [ ] The change is focused and has a regression test.
- [ ] Migrations are append-only and have a clear `down_revision`.
- [ ] Documentation and `IMPLEMENTATION_STATUS.md` are updated.
- [ ] No secrets, generated reports, local environment files, or caches are included.
- [ ] The pull request states which known gaps remain.
