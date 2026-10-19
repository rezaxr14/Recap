#!/usr/bin/env python
"""End-to-end smoke test against a live stack.

Unlike the pytest suite, this script is intentionally operator-facing: it runs
the real CLI-level flow and prints each verified step, so a human can see the
system working. It fails loudly with a non-zero exit code on any deviation.

Usage::

    docker compose up -d postgres elasticsearch
    recap db migrate
    python scripts/run_smoke_test.py
"""

from __future__ import annotations

import sys
from typing import Any

from recap.config import get_settings
from recap.container import Container
from recap.demo import run_demo


def main() -> int:
    settings = get_settings()
    container = Container.build(settings)
    checks: list[tuple[str, bool, str]] = []

    def check(name: str, passed: bool, detail: str = "") -> None:
        checks.append((name, passed, detail))
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))

    try:
        health = container.health()
        check("postgres reachable", health.get("database") == "ok", str(health.get("database")))
        es = health.get("elasticsearch")
        es_ok = isinstance(es, dict) and es.get("status") in {"green", "yellow"}
        check("elasticsearch reachable", es_ok, str(es))

        index_info = container.init_search()
        check(
            "index and aliases initialized",
            bool(index_info.get("index")),
            f"{index_info.get('index')} -> {index_info.get('read_alias')}",
        )

        report = run_demo(container, write_report=False)
        summary: dict[str, Any] = report["summary"]
        check("demo recorded a failed run", bool(summary.get("failed_run_id")))
        check("demo compiled and promoted a memory", bool(summary.get("memory_v1")))
        check(
            "stale v1 memory blocked on v2",
            summary.get("stale_memory_blocked") is True,
        )
        check(
            "fresh v2 memory retrieved",
            summary.get("fresh_memory_retrieved") is True,
        )

        benchmark: dict[str, Any] = report.get("benchmark", {})
        if benchmark:
            check(
                "outcome-aware made no incompatible selections",
                benchmark["outcome_aware"]["incompatible_selection_rate"] == 0.0,
                str(benchmark["outcome_aware"]["incompatible_selection_rate"]),
            )
            check(
                "no-memory baseline cannot recover",
                benchmark["no_memory"]["task_success_rate"] == 0.0,
                str(benchmark["no_memory"]["task_success_rate"]),
            )
    except Exception as exc:  # operator-facing: report, do not traceback-dump
        print(f"[FAIL] smoke test raised {type(exc).__name__}: {exc}")
        return 1
    finally:
        container.es_client.close()

    failures = [name for name, passed, _ in checks if not passed]
    print()
    if failures:
        print(f"{len(failures)} check(s) failed: {', '.join(failures)}")
        return 1
    print(f"all {len(checks)} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
