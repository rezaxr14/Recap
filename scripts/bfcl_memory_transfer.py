"""Failure-memory transfer on real BFCL schemas (scripted drift).

Every function schema, user query, and gold argument comes from BFCL v3. What
is injected is the *failure*: one required parameter is renamed in a second
schema version, so a previously valid call becomes invalid. That is the same
class of drift RECAP exists to survive, but it is not an observed production
failure and the report says so.

Protocol per function (scenario):
  1. Episode 1 calls the function with its real gold arguments on the drifted
     schema. The call fails with an unknown-parameter error.
  2. The failure and its repair (old name -> new name) are recorded as one
     memory, alongside memories from every other scenario.
  3. Episode 2 is a *different real query* for the same function. The agent
     must call the drifted schema again. A method succeeds only if it applies
     the correct repair and produces a schema-valid call.

Baselines:
  * ``no_memory`` / ``retry_only``: repeat the stale call.
  * ``bm25`` / ``vector`` / ``hybrid_rrf``: retrieve one memory by the episode-2
    query text from the whole memory store, then apply its repair.
  * ``function_keyed``: address the memory by function name (what a tool-calling
    agent already knows), no text retrieval.
  * ``oracle_memory``: always apply the correct repair (upper bound).

Run:
    python scripts/bfcl_memory_transfer.py --embedding qwen \
        --model-path C:/Users/rezax/Home/Code/models/Qwen3-Embedding-0.6B \
        --out docs/evidence
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from recap.config import ElasticsearchSettings, EmbeddingSettings
from recap.infrastructure.elasticsearch.client import build_client
from recap.infrastructure.embeddings import build_embedding_provider
from recap.recall.rrf import RankedList, reciprocal_rank_fusion

DEFAULT_DATA = Path(r"C:\Users\rezax\Home\Code\models\datasets\bfcl")
INDEX = "bfcl-v3-memories"
BOOTSTRAP = 2000
RNG_SEED = 20261030
TOP_K = 5


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _parse_call(text: str) -> dict[str, Any]:
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


def _load_tasks(data: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    answers = {
        row["id"]: row for row in _read_jsonl(data / "possible_answer" / "BFCL_v3_simple.json")
    }
    tasks: list[dict[str, Any]] = []
    for row in _read_jsonl(data / "BFCL_v3_simple.json"):
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
                            "query": _user_text(row.get("question")),
                            "args": args,
                            "schema": (row.get("function") or [{}])[0],
                        }
                    )
    for row in _read_jsonl(data / "BFCL_v3_exec_simple.json"):
        for text in row.get("ground_truth") or []:
            if not isinstance(text, str):
                continue
            args = _parse_call(text)
            if not args:
                continue
            tasks.append(
                {
                    "id": row["id"],
                    "function": text.split("(", 1)[0].strip(),
                    "query": _user_text(row.get("question")),
                    "args": args,
                    "schema": (row.get("function") or [{}])[0],
                }
            )
    hashes = {
        name: _sha256(data / name)
        for name in (
            "BFCL_v3_simple.json",
            "possible_answer/BFCL_v3_simple.json",
            "BFCL_v3_exec_simple.json",
        )
    }
    return tasks, hashes


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _user_text(messages: Any) -> str:
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


def _drift(schema: dict[str, Any]) -> tuple[dict[str, Any], str, str] | None:
    params = schema.get("parameters") or {}
    properties = dict(params.get("properties") or {})
    required = list(params.get("required") or properties)
    renameable = [name for name in required if name in properties]
    if not renameable:
        return None
    old = sorted(renameable)[0]
    new = f"{old}_v2"
    properties[new] = properties.pop(old)
    required = [new if name == old else name for name in required]
    drifted = {
        "name": schema.get("name"),
        "description": schema.get("description"),
        "parameters": {**params, "properties": properties, "required": required},
    }
    return drifted, old, new


def _call_status(schema: dict[str, Any], args: dict[str, Any]) -> tuple[set[str], set[str]]:
    params = schema.get("parameters") or {}
    properties = set(params.get("properties") or {})
    required = set(params.get("required") or properties)
    return set(args) - properties, required - set(args)


def _valid(schema: dict[str, Any], args: dict[str, Any]) -> bool:
    unknown, missing = _call_status(schema, args)
    return not unknown and not missing


def _scenarios(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        grouped[task["function"]].append(task)
    scenarios: list[dict[str, Any]] = []
    for function, items in sorted(grouped.items()):
        unique = {item["query"]: item for item in items}
        if len(unique) < 2:
            continue
        ordered = [unique[key] for key in sorted(unique)][:2]
        drifted = _drift(ordered[0]["schema"])
        if drifted is None:
            continue
        schema, old, new = drifted
        first, second = ordered
        # A pair is measurable only if the first call fails on the drifted schema,
        # the second call would fail too, and applying the repair makes it valid.
        if _valid(schema, first["args"]) or _valid(schema, second["args"]):
            continue
        if not _valid(schema, _apply_rename(second["args"], old, new)):
            continue
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
                    f"{function}: parameter '{old}' was renamed to '{new}'. "
                    f"A call failed with {f'unknown parameter {old!r}'}."
                ),
            }
        )
    return scenarios


def _apply_rename(args: dict[str, Any], old: str, new: str) -> dict[str, Any]:
    updated = dict(args)
    if old in updated:
        updated[new] = updated.pop(old)
    return updated


def _auc(positives: list[float], negatives: list[float]) -> float:
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def _bootstrap_ci(values: list[float], rng: np.random.Generator) -> list[float]:
    array = np.asarray(values, dtype=float)
    samples = rng.choice(array, size=(BOOTSTRAP, array.size), replace=True).mean(axis=1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def _rank(ids: list[str], scores: np.ndarray, k: int = TOP_K) -> list[str]:
    return [ids[index] for index in np.argsort(-scores)[:k]]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--embedding", choices=["hashing", "qwen"], default="qwen")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("docs/evidence"))
    parser.add_argument("--doc", type=Path, default=None)
    parser.add_argument("--es-url", default="http://localhost:9200")
    args = parser.parse_args(argv)

    tasks, hashes = _load_tasks(args.data_dir)
    scenarios = _scenarios(tasks)
    print(f"tasks={len(tasks)} scenarios={len(scenarios)}")
    if len(scenarios) < 10:
        raise SystemExit("too few paired scenarios to report")

    if args.embedding == "qwen":
        model_path = args.model_path or Path(
            r"C:\Users\rezax\Home\Code\models\Qwen3-Embedding-0.6B"
        )
        settings = EmbeddingSettings(
            provider="sentence-transformers", model_path=str(model_path), dimension=1024
        )
        model_id = model_path.name
    else:
        settings = EmbeddingSettings(provider="hashing", dimension=256)
        model_id = settings.model_id
    provider = build_embedding_provider(settings)

    ids = [scenario["function"] for scenario in scenarios]
    memory_texts = [scenario["memory_text"] for scenario in scenarios]
    queries = [scenario["query2"] for scenario in scenarios]

    client = build_client(ElasticsearchSettings(url=args.es_url))
    try:
        client.indices.delete(index=INDEX, ignore_unavailable=True)
        client.indices.create(
            index=INDEX,
            mappings={
                "properties": {
                    "function": {"type": "text"},
                    "text": {"type": "text"},
                }
            },
        )
        operations: list[dict[str, Any]] = []
        for scenario in scenarios:
            operations.append({"index": {"_index": INDEX, "_id": scenario["function"]}})
            operations.append({"function": scenario["function"], "text": scenario["memory_text"]})
        client.bulk(operations=operations, refresh=True)
        bm25_rankings: list[list[str]] = []
        for query in queries:
            response = client.search(
                index=INDEX,
                size=TOP_K,
                query={
                    "multi_match": {
                        "query": query,
                        "fields": ["function^2", "text"],
                        "type": "best_fields",
                        "tie_breaker": 0.3,
                    }
                },
            )
            bm25_rankings.append([hit["_id"] for hit in response["hits"]["hits"]])
    finally:
        client.close()

    matrix = np.asarray(provider.embed_documents(memory_texts), dtype=float)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12
    query_vectors = np.asarray(provider.embed_documents(queries), dtype=float)
    query_vectors /= np.linalg.norm(query_vectors, axis=1, keepdims=True) + 1e-12
    dense_rankings = [_rank(ids, matrix @ vector) for vector in query_vectors]
    hybrid_rankings = [
        [
            hit.doc_id
            for hit in reciprocal_rank_fusion(
                [RankedList("bm25", lexical), RankedList("vector", dense)], k=60, limit=TOP_K
            )
        ]
        for lexical, dense in zip(bm25_rankings, dense_rankings, strict=True)
    ]

    correct = {scenario["function"]: scenario for scenario in scenarios}
    retrieval = {
        "bm25": bm25_rankings,
        "vector": dense_rankings,
        "hybrid_rrf": hybrid_rankings,
    }
    outcomes: dict[str, list[float]] = {name: [] for name in retrieval}
    outcomes["no_memory"] = []
    outcomes["retry_only"] = []
    outcomes["function_keyed"] = []
    outcomes["oracle_memory"] = []
    retrieval_hits: dict[str, list[float]] = {name: [] for name in retrieval}

    for position, scenario in enumerate(scenarios):
        schema = scenario["schema"]
        stale = scenario["args2"]
        outcomes["no_memory"].append(1.0 if _valid(schema, stale) else 0.0)
        outcomes["retry_only"].append(1.0 if _valid(schema, stale) else 0.0)
        keyed = _apply_rename(stale, scenario["old"], scenario["new"])
        outcomes["function_keyed"].append(1.0 if _valid(schema, keyed) else 0.0)
        oracle = _apply_rename(stale, scenario["old"], scenario["new"])
        outcomes["oracle_memory"].append(1.0 if _valid(schema, oracle) else 0.0)
        for name, rankings in retrieval.items():
            top = rankings[position][0] if rankings[position] else ""
            retrieval_hits[name].append(1.0 if top == scenario["function"] else 0.0)
            if top == scenario["function"]:
                repaired = _apply_rename(stale, scenario["old"], scenario["new"])
            else:
                # A wrong memory renames a parameter this function does not have.
                source = correct.get(top)
                repaired = (
                    _apply_rename(stale, source["old"], source["new"]) if source else dict(stale)
                )
            outcomes[name].append(1.0 if _valid(schema, repaired) else 0.0)

    rng = np.random.default_rng(RNG_SEED)
    results: dict[str, Any] = {}
    for name, values in outcomes.items():
        results[name] = {
            "success_rate": float(np.mean(values)),
            "success_ci95": _bootstrap_ci(values, rng),
            "successes": int(sum(values)),
            "n_scenarios": len(values),
        }
        if name in retrieval_hits:
            hits = retrieval_hits[name]
            results[name]["retrieval_hit_at_1"] = float(np.mean(hits))
            results[name]["retrieval_hit_at_1_ci95"] = _bootstrap_ci(hits, rng)

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset": "gorilla-llm/Berkeley-Function-Calling-Leaderboard (BFCL v3)",
        "source_hashes": hashes,
        "scenarios": len(scenarios),
        "functions": sorted(correct),
        "embedding": {
            "provider": settings.provider,
            "model": model_id,
            "dimension": int(matrix.shape[1]),
        },
        "bootstrap_resamples": BOOTSTRAP,
        "rng_seed": RNG_SEED,
        "injected_failure": "one required parameter renamed per function (old -> old_v2)",
        "results": results,
        "limits": [
            "The failure is scripted: one required parameter is renamed on the real "
            "BFCL schema. No production failure log was used.",
            "Queries, schemas, and gold arguments are real BFCL data.",
            "Function selection is assumed correct; this isolates memory transfer, not "
            "the joint selection+memory problem.",
            "One scenario per function; functions are the resampling unit.",
            "Pairs where the second real call did not supply the drifted required "
            "parameter, or where the stale call was already valid, are excluded because "
            "no repair can be measured there.",
            "The memory is a single repair sentence. Real memories can be noisier.",
        ],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    json_path = args.out / f"real-data-bfcl-transfer-{args.embedding}.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {json_path}")
    for name, stats in results.items():
        print(f"  {name}: {stats['successes']}/{stats['n_scenarios']}")
    if args.doc:
        args.doc.parent.mkdir(parents=True, exist_ok=True)
        args.doc.write_text(render_markdown(payload), encoding="utf-8")
        print(f"wrote {args.doc}")
    return 0


def render_markdown(payload: dict[str, Any]) -> str:
    results = payload["results"]
    lines = [
        "# Failure-memory transfer on real BFCL schemas",
        "",
        "> Generated by `scripts/bfcl_memory_transfer.py` from the JSON report beside this "
        "file. Do not hand-edit the numbers.",
        "",
        "## What is real and what is injected",
        "",
        "Real: function schemas, user queries, and gold arguments come from BFCL v3.",
        "",
        "Injected: one required parameter is renamed per function, so the first call fails "
        "with an unknown-parameter error. This is a scripted drift, not an observed "
        "production failure log. Function selection is assumed correct, so this isolates "
        "memory transfer rather than the joint selection+memory problem.",
        "",
        "## Provenance",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| Dataset | `{payload['dataset']}` |",
        f"| Generated | `{payload['generated_at']}` |",
        f"| Scenarios (one per function) | {payload['scenarios']} |",
        f"| Embedder | `{payload['embedding']['provider']}` / "
        f"`{payload['embedding']['model']}` ({payload['embedding']['dimension']}d) |",
        f"| Injected failure | {payload['injected_failure']} |",
        "",
        "## Results (success = schema-valid call on episode 2)",
        "",
        "| Method | Successes | Success rate (95% CI) | Retrieval hit@1 |",
        "|---|---:|---:|---:|",
    ]
    order = [
        "no_memory",
        "retry_only",
        "bm25",
        "vector",
        "hybrid_rrf",
        "function_keyed",
        "oracle_memory",
    ]
    for name in order:
        stats = results[name]
        hit = f"{stats['retrieval_hit_at_1']:.3f}" if "retrieval_hit_at_1" in stats else "n/a"
        lines.append(
            f"| `{name}` | {stats['successes']}/{stats['n_scenarios']} | "
            f"{stats['success_rate']:.3f} "
            f"[{stats['success_ci95'][0]:.3f}, {stats['success_ci95'][1]:.3f}] | {hit} |"
        )
    lines.extend(["", "## Limits", ""])
    for limit in payload["limits"]:
        lines.append(f"- {limit}")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
