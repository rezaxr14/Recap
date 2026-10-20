"""Fail when two benchmark report directories disagree on the summary.

The benchmark is deterministic by construction: unique workspace per run, reset
before and after, content-addressed chunk ids, and stable tie-breaks. CI runs it
twice and compares the summaries to catch stale-state regressions. Latency lives
in the separate ``performance`` section and is intentionally excluded: machine
speed is not a reproducibility property.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _load_report(directory: Path) -> dict:
    reports = sorted(directory.glob("*.json"))
    if len(reports) != 1:
        raise SystemExit(f"expected exactly one JSON report in {directory}, found {len(reports)}")
    with reports[0].open(encoding="utf-8") as handle:
        return json.load(handle)


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        raise SystemExit("usage: check_benchmark_repro.py <first-dir> <second-dir>")
    first = _load_report(Path(argv[1]))
    second = _load_report(Path(argv[2]))
    if first["summary"] != second["summary"]:
        keys = sorted(set(first["summary"]) | set(second["summary"]))
        for key in keys:
            if first["summary"].get(key) != second["summary"].get(key):
                print(f"baseline {key} differs:", file=sys.stderr)
                print(f"  first:  {first['summary'].get(key)}", file=sys.stderr)
                print(f"  second: {second['summary'].get(key)}", file=sys.stderr)
        return 1
    print(f"benchmark summaries identical across {len(first['summary'])} baselines")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
