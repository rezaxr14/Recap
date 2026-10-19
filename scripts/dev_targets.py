"""Makefile-equivalent developer commands.

On Windows PowerShell, ``make`` is often unavailable; ``scripts/dev.py`` provides
the same targets. On Unix, ``make test`` etc. work directly.
"""

TARGETS = {
    "up": "docker compose up -d postgres elasticsearch",
    "down": "docker compose down",
    "init": "recap db migrate && recap search init",
    "lint": "ruff check src tests && ruff format --check src tests",
    "format": "ruff format src tests && ruff check --fix src tests",
    "typecheck": "mypy src",
    "test": "pytest -m 'not integration'",
    "test-integration": "pytest -m integration",
    "test-all": "pytest",
    "demo": "recap demo run",
    "eval": "recap eval run --split dev",
    "smoke": "python scripts/run_smoke_test.py",
}
