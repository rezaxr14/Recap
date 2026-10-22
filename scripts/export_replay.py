"""Export recorded benchmark reports for the static replay web app.

The web app never talks to a database in replay mode. This script copies only
artifacts that already exist in a benchmark JSON report and adds the provenance
the UI needs to label every number. It does not invent, smooth, or aggregate
values beyond the summary that the benchmark runner already produced.

Usage::

    python scripts/export_replay.py \
        --input benchmark/reports/benchmark_test_20240101T000000Z.json \
        --out web/public/replay

Multiple inputs are copied as selectable datasets. The newest v3 report with a
real LLM agent is selected as the default when one is present.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _meta(payload: dict[str, Any], source: Path) -> dict[str, Any]:
    protocol = payload.get("protocol", {})
    embedding = payload.get("embedding", {})
    retrieval = payload.get("retrieval", {})
    return {
        "source": source.name,
        "suite": payload.get("suite", "unknown"),
        "split": payload.get("split", "unknown"),
        "generated_at": payload.get("generated_at"),
        "dataset_hash": payload.get("dataset_hash"),
        "agent": protocol.get("agent", payload.get("agent", {}).get("agent_id", "unknown")),
        "agent_mode": protocol.get("agent_mode"),
        "provider": protocol.get("provider"),
        "model": protocol.get("model"),
        "embedder": embedding.get("model_id")
        or embedding.get("model")
        or retrieval.get("embedding_model"),
        "embedder_provider": embedding.get("provider_type") or embedding.get("provider"),
        "fixture_count": protocol.get("fixture_count"),
        "inference": protocol.get("inference"),
        "reranker": payload.get("retrieval", {}).get("reranker"),
        "synthetic_store": payload.get("store", {}),
        "warnings": payload.get("warnings", []),
        "note": protocol.get("note"),
    }


def _compact(payload: dict[str, Any], source: Path) -> dict[str, Any]:
    return {
        "meta": _meta(payload, source),
        "summary": payload.get("summary", {}),
        "by_tier": payload.get("by_tier", {}),
        "by_task_family": payload.get("by_task_family", {}),
        "per_task": payload.get("per_task", []),
        "tasks": payload.get("tasks", []),
        "acceptance": payload.get("acceptance"),
        "performance": payload.get("performance", {}),
        "retrieval": payload.get("retrieval", {}),
        "embedding": payload.get("embedding", {}),
        "store": payload.get("store", {}),
        "protocol": payload.get("protocol", {}),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True, type=Path)
    parser.add_argument("--out", type=Path, default=Path("web/public/replay"))
    args = parser.parse_args()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    datasets: list[dict[str, Any]] = []
    for source in args.input:
        payload = json.loads(source.read_text(encoding="utf-8"))
        datasets.append(_compact(payload, source))

    def rank(item: dict[str, Any]) -> tuple[int, int, int, int, str]:
        meta = item["meta"]
        llm = 1 if meta.get("agent_mode") in {"verifying", "credulous"} else 0
        v3 = 1 if meta.get("suite") == "v3" else 0
        held_out = 1 if meta.get("split") == "test" else 0
        # A report with the full baseline ladder is more useful as the default
        # replay than a one-baseline smoke run. Live runs still win over
        # deterministic runs when both expose the same ladder.
        baselines = len(item.get("summary", {}))
        return (v3, baselines, llm, held_out, str(meta.get("generated_at", "")))

    datasets.sort(key=rank, reverse=True)
    primary = datasets[0]
    (out / "benchmark.json").write_text(
        json.dumps(primary, indent=2, default=str), encoding="utf-8"
    )
    (out / "datasets.json").write_text(
        json.dumps(
            {
                "generated_at": datetime.now(UTC).isoformat(),
                "datasets": [item["meta"] for item in datasets],
            },
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    for index, item in enumerate(datasets):
        name = f"dataset-{index}.json"
        (out / name).write_text(json.dumps(item, indent=2, default=str), encoding="utf-8")
    print(f"wrote {len(datasets)} dataset(s) to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
