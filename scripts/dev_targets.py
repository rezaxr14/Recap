"""Makefile-equivalent developer commands.

On Windows PowerShell, ``make`` is often unavailable. This script runs the same
targets so local development and CI use one definition of "green":

    python scripts/dev_targets.py test
    python scripts/dev_targets.py lint typecheck
    python scripts/dev_targets.py --list

Targets run sequentially and the process exits non-zero on the first failure.
"""

from __future__ import annotations

import subprocess
import sys

TARGETS: dict[str, str] = {
    "up": "docker compose up -d postgres elasticsearch",
    "down": "docker compose down",
    "init": "recap db migrate && recap search init",
    "lint": "ruff check src tests scripts && ruff format --check src tests scripts",
    "format": "ruff format src tests scripts && ruff check --fix src tests scripts",
    "typecheck": "mypy src",
    "test": "pytest tests/unit",
    "test-integration": "pytest tests/integration tests/e2e --run-integration",
    "test-all": "pytest --run-integration",
    "demo": "recap demo run",
    "eval": "recap eval run --split dev",
    "repro": "python scripts/run_benchmark_repro.py",
    "smoke": "python scripts/run_smoke_test.py",
}


def _run(target: str) -> int:
    command = TARGETS[target]
    print(f"[{target}] {command}", flush=True)
    # Shell is intentional: targets are human-authored command lines, not user input.
    return subprocess.call(command, shell=True)


def main(argv: list[str]) -> int:
    if not argv or argv[0] in {"-h", "--help"}:
        print(__doc__)
        print("targets:")
        for name, command in TARGETS.items():
            print(f"  {name:<16} {command}")
        return 0 if argv else 2
    if argv[0] in {"-l", "--list"}:
        for name, command in TARGETS.items():
            print(f"{name}\t{command}")
        return 0

    unknown = [target for target in argv if target not in TARGETS]
    if unknown:
        print(f"unknown target(s): {', '.join(unknown)}", file=sys.stderr)
        print(f"available: {', '.join(TARGETS)}", file=sys.stderr)
        return 2

    for target in argv:
        code = _run(target)
        if code != 0:
            print(f"[{target}] FAILED with exit code {code}", file=sys.stderr)
            return code
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
