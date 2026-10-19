"""Deterministic task suite with train/dev/test splits and controlled failures.

Tasks are grouped by template so no scenario is duplicated across splits.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class BenchmarkTask:
    task_id: str
    task_family: str
    task_version: str
    instruction: str
    environment_version: str
    split: str
    failure_mode: str
    expected_memory_signature: str = ""
    relevant_memory_tags: tuple[str, ...] = ()
    seed: int = 7
    difficulty: str = "easy"
    side_effect_risk: str = "read_only"

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "task_family": self.task_family,
            "task_version": self.task_version,
            "instruction": self.instruction,
            "environment_version": self.environment_version,
            "split": self.split,
            "failure_mode": self.failure_mode,
            "expected_memory_signature": self.expected_memory_signature,
            "relevant_memory_tags": list(self.relevant_memory_tags),
            "seed": self.seed,
            "difficulty": self.difficulty,
            "side_effect_risk": self.side_effect_risk,
        }


def default_tasks() -> list[BenchmarkTask]:
    return [
        BenchmarkTask(
            task_id="paginate-train-01",
            task_family="api_pagination",
            task_version="1",
            instruction="Collect every record from the paginated API.",
            environment_version="1",
            split="train",
            failure_mode="invalid_cursor",
            expected_memory_signature="ValidationError|INVALID_CURSOR|list_records",
            relevant_memory_tags=("pagination", "invalid_cursor"),
        ),
        BenchmarkTask(
            task_id="paginate-train-02",
            task_family="api_pagination",
            task_version="1",
            instruction="Page through all tenant records and report the count.",
            environment_version="1",
            split="train",
            failure_mode="invalid_cursor",
            expected_memory_signature="ValidationError|INVALID_CURSOR|list_records",
            relevant_memory_tags=("pagination", "invalid_cursor"),
            seed=11,
        ),
        BenchmarkTask(
            task_id="paginate-dev-01",
            task_family="api_pagination",
            task_version="1",
            instruction="Fetch all records for the tenant despite cursor expiry.",
            environment_version="1",
            split="dev",
            failure_mode="invalid_cursor",
            expected_memory_signature="ValidationError|INVALID_CURSOR|list_records",
            relevant_memory_tags=("pagination", "invalid_cursor"),
            seed=13,
        ),
        BenchmarkTask(
            task_id="paginate-dev-02",
            task_family="api_pagination",
            task_version="2",
            instruction="Collect all records after the API renamed the pagination parameter.",
            environment_version="2",
            split="dev",
            failure_mode="renamed_parameter",
            expected_memory_signature="ValidationError|UNKNOWN_PARAMETER|list_records",
            relevant_memory_tags=("pagination", "page_token"),
            seed=23,
            difficulty="medium",
        ),
        BenchmarkTask(
            task_id="paginate-test-01",
            task_family="api_pagination",
            task_version="1",
            instruction="Collect all records from the paginated API.",
            environment_version="1",
            split="test",
            failure_mode="invalid_cursor",
            expected_memory_signature="ValidationError|INVALID_CURSOR|list_records",
            relevant_memory_tags=("pagination", "invalid_cursor"),
            seed=17,
        ),
        BenchmarkTask(
            task_id="paginate-test-v2-01",
            task_family="api_pagination",
            task_version="2",
            instruction="Collect all records from the upgraded paginated API.",
            environment_version="2",
            split="test",
            failure_mode="renamed_parameter",
            expected_memory_signature="ValidationError|INVALID_CURSOR|list_records",
            relevant_memory_tags=("pagination", "page_token"),
            seed=19,
            difficulty="medium",
        ),
    ]


def split_hash(tasks: list[BenchmarkTask]) -> str:
    payload = json.dumps(
        [task.to_dict() for task in sorted(tasks, key=lambda item: item.task_id)],
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"sha256:{hashlib.sha256(payload.encode()).hexdigest()}"


def check_leakage(tasks: list[BenchmarkTask]) -> list[str]:
    """Flag task IDs/instructions that appear in more than one split."""
    seen: dict[str, set[str]] = {}
    issues: list[str] = []
    for task in tasks:
        key = hashlib.sha256(task.instruction.strip().lower().encode()).hexdigest()
        seen.setdefault(key, set()).add(task.split)
    for key, splits in seen.items():
        if len(splits) > 1:
            issues.append(f"identical instruction across splits: {sorted(splits)} ({key[:12]})")
    return issues
