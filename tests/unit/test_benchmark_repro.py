from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from scripts.check_benchmark_repro import _load_report


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_repro_loader_ignores_checkpoint_metadata(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    for directory in (first, second):
        _write(
            directory / "benchmark_dev_20260101T000000Z.json",
            {"summary": {"no_memory": {"task_success_rate": 0.0}}},
        )
        _write(
            directory / "checkpoint_v2_dev_reference_seed1.meta.json",
            {"workspace_id": "ignored", "signature": {}},
        )

    assert _load_report(first) == _load_report(second)
    assert _load_report(first)["summary"]["no_memory"]["task_success_rate"] == 0.0
