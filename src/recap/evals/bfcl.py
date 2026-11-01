"""BFCL v3 loading and paired-scenario construction (pure, no I/O services).

The retrieval and transfer scripts, and the RECAP-stack experiment, all need
the same view of the public BFCL files. Keeping it here means the drift
definition and the pair filter cannot silently diverge between reports.

Everything in this module is CPU-only and network-free. Callers pass the
directory holding the downloaded BFCL v3 files.
"""

from __future__ import annotations

import ast
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any

DATA_FILES = (
    "BFCL_v3_simple.json",
    "possible_answer/BFCL_v3_simple.json",
    "BFCL_v3_exec_simple.json",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    import json

    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def user_text(messages: Any) -> str:
    """Flatten the first user turn exactly as written, without the tool docs."""
    if isinstance(messages, str):
        return messages
    parts: list[str] = []
    for turn in messages or []:
        if isinstance(turn, list):
            for message in turn:
                if isinstance(message, dict) and message.get("role") == "user":
                    parts.append(str(message.get("content") or ""))
        elif isinstance(turn, dict) and turn.get("role") == "user":
            parts.append(str(turn.get("content") or ""))
    return " ".join(part for part in parts if part).strip()


def parse_call(text: str) -> dict[str, Any]:
    """Parse a BFCL ``f(a=1, b="x")`` call into a keyword-argument dict."""
    try:
        call = ast.parse(text, mode="eval").body
    except SyntaxError:
        return {}
    if not isinstance(call, ast.Call):
        return {}
    args: dict[str, Any] = {}
    for keyword in call.keywords:
        if keyword.arg is None:
            continue
        try:
            args[keyword.arg] = ast.literal_eval(keyword.value)
        except (ValueError, SyntaxError):
            return {}
    return args


def load_tasks(data: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Load labeled BFCL tasks: one row per (task, gold function, args)."""
    answers = {
        row["id"]: row for row in read_jsonl(data / "possible_answer" / "BFCL_v3_simple.json")
    }
    tasks: list[dict[str, Any]] = []
    for row in read_jsonl(data / "BFCL_v3_simple.json"):
        gold = answers.get(row["id"], {}).get("ground_truth") or []
        for entry in gold:
            if not isinstance(entry, dict):
                continue
            for name, params in entry.items():
                if isinstance(params, dict):
                    # BFCL stores each value as a list of accepted alternatives.
                    args = {
                        key: (value[0] if isinstance(value, list) and value else value)
                        for key, value in params.items()
                    }
                    tasks.append(
                        {
                            "id": row["id"],
                            "function": name,
                            "query": user_text(row.get("question")),
                            "args": args,
                            "schema": (row.get("function") or [{}])[0],
                        }
                    )
    for row in read_jsonl(data / "BFCL_v3_exec_simple.json"):
        for text in row.get("ground_truth") or []:
            if not isinstance(text, str):
                continue
            args = parse_call(text)
            if not args:
                continue
            tasks.append(
                {
                    "id": row["id"],
                    "function": text.split("(", 1)[0].strip(),
                    "query": user_text(row.get("question")),
                    "args": args,
                    "schema": (row.get("function") or [{}])[0],
                }
            )
    hashes = {name: sha256_file(data / name) for name in DATA_FILES}
    return tasks, hashes


def drift_schema(
    schema: dict[str, Any],
    *,
    parameter: str | None = None,
    new_name: str | None = None,
) -> tuple[dict[str, Any], str, str] | None:
    """Rename one required parameter to ``{old}_v2`` (the injected drift).

    With ``parameter`` omitted the alphabetically first required parameter is
    renamed, which is what the transfer report has always used. Passing a name
    lets the experiment chain a second drift on the same parameter, so the
    schema keeps changing inside one environment id (silent drift).
    ``new_name`` overrides the generated ``{old}_v2`` suffix, which lets the
    drift experiment rename ``old`` straight to ``old_v3`` on the second step
    instead of accumulating suffixes (``old_v2_v2``).
    """
    params = schema.get("parameters") or {}
    properties = dict(params.get("properties") or {})
    required = list(params.get("required") or properties)
    renameable = [name for name in required if name in properties]
    if not renameable:
        return None
    if parameter is not None:
        if parameter not in renameable:
            return None
        old = parameter
    else:
        old = sorted(renameable)[0]
    new = new_name or f"{old}_v2"
    if new == old or new in properties:
        return None
    properties[new] = properties.pop(old)
    required = [new if name == old else name for name in required]
    drifted = {
        "name": schema.get("name"),
        "description": schema.get("description"),
        "parameters": {**params, "properties": properties, "required": required},
    }
    return drifted, old, new


def call_status(schema: dict[str, Any], args: dict[str, Any]) -> tuple[set[str], set[str]]:
    params = schema.get("parameters") or {}
    properties = set(params.get("properties") or {})
    required = set(params.get("required") or properties)
    return set(args) - properties, required - set(args)


def is_valid_call(schema: dict[str, Any], args: dict[str, Any]) -> bool:
    unknown, missing = call_status(schema, args)
    return not unknown and not missing


def apply_rename(args: dict[str, Any], old: str, new: str) -> dict[str, Any]:
    updated = dict(args)
    if old in updated:
        updated[new] = updated.pop(old)
    return updated


def build_transfer_scenarios(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pair two real queries per function around the injected parameter rename.

    A pair is measurable only when the first call fails on the drifted schema,
    the second call would fail too, and applying the repair makes it valid.
    """
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        grouped[task["function"]].append(task)
    scenarios: list[dict[str, Any]] = []
    for function, items in sorted(grouped.items()):
        unique = {item["query"]: item for item in items}
        if len(unique) < 2:
            continue
        ordered = [unique[key] for key in sorted(unique)][:2]
        drifted = drift_schema(ordered[0]["schema"])
        if drifted is None:
            continue
        schema, old, new = drifted
        first, second = ordered
        if is_valid_call(schema, first["args"]) or is_valid_call(schema, second["args"]):
            continue
        if not is_valid_call(schema, apply_rename(second["args"], old, new)):
            continue
        properties = (schema.get("parameters") or {}).get("properties") or {}
        parameter_text = "; ".join(
            f"{name}: {spec.get('description') or ''}"
            for name, spec in properties.items()
            if isinstance(spec, dict)
        )
        scenarios.append(
            {
                "function": function,
                "schema": schema,
                "old": old,
                "new": new,
                "query1": first["query"],
                "args1": first["args"],
                "query2": second["query"],
                "args2": second["args"],
                "error": f"unknown parameter '{old}'; required '{new}'",
                "memory_text": (
                    f"Tool {function}: {schema.get('description') or ''} "
                    f"Parameters: {parameter_text}. "
                    f"Repair: parameter '{old}' was renamed to '{new}'. "
                    f"A call failed with unknown parameter '{old}'."
                ),
            }
        )
    return scenarios


def sibling_pairs(scenarios: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Return (memory-owner, query-owner) pairs from the same tool family.

    A family is the function name with its last dot-segment or trailing
    ``_suffix`` removed. Only pairs whose parameter sets differ are useful:
    the same parameters would make transfer valid and uninteresting.
    """
    families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for scenario in scenarios:
        name = scenario["function"]
        if "_by_" in name:
            family = name.split("_by_", 1)[0]
        elif "." in name:
            family = name.rsplit(".", 1)[0]
        elif "_" in name:
            family = name.rsplit("_", 1)[0]
        else:
            family = ""
        if family:
            families[family].append(scenario)
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for members in families.values():
        for owner in members:
            for query_owner in members:
                if owner["function"] == query_owner["function"]:
                    continue
                owner_params = set(
                    (owner["schema"].get("parameters") or {}).get("properties") or {}
                )
                query_params = set(
                    (query_owner["schema"].get("parameters") or {}).get("properties") or {}
                )
                if owner_params != query_params:
                    pairs.append((owner, query_owner))
    return pairs
