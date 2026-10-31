"""Retrieval measurement on a real public corpus (BFCL v3).

This is deliberately *not* the synthetic gold/decoy suite. Every query, function
schema, and function description comes from the Berkeley Function Calling
Leaderboard (BFCL v3). The only thing we add is the retrieval task itself:
given the user's request, rank a single global pool of 630 real functions.

Labels:
  * ``simple`` / ``exec_simple`` tasks: the ground-truth function name is the
    single relevant document.
  * ``irrelevance`` tasks: the single *provided* function does not apply. They
    are controls, not successes. We never fold them into recall. Because the
    global pool may still contain a function that fits, controls are scored as
    a matched single-candidate decision ("apply the offered tool?"), not as a
    630-way ranking.

Protocol:
  * Queries are split deterministically (sha1 of the task id) into dev/test.
  * RRF weights and abstention thresholds are chosen on dev only.
  * Headline numbers are test-only, with query-level bootstrap 95% CIs.
  * The lexical channel uses either the default analyzer or RECAP's
    identifier-aware subfields (``--analyzer``), and queries are embedded
    through ``embed_query`` so a model instruction prefix applies.

Data (public, one download):
    python -c "from huggingface_hub import hf_hub_download as d; \
      [d('gorilla-llm/Berkeley-Function-Calling-Leaderboard', f, repo_type='dataset', \
         local_dir='data/bfcl') for f in ('BFCL_v3_simple.json', \
         'possible_answer/BFCL_v3_simple.json', 'BFCL_v3_exec_simple.json', \
         'BFCL_v3_irrelevance.json')]"

Run:
    python scripts/bfcl_retrieval.py --embedding qwen \
        --model-path /path/to/Qwen3-Embedding-0.6B \
        --data-dir data/bfcl --out docs/evidence
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from recap.config import ElasticsearchSettings, EmbeddingSettings
from recap.infrastructure.elasticsearch.client import build_client
from recap.infrastructure.elasticsearch.mappings import (
    IDENTIFIER_ANALYZER,
    IDENTIFIER_FILTER,
)
from recap.infrastructure.embeddings import build_embedding_provider
from recap.recall.rrf import RankedList, reciprocal_rank_fusion

DEFAULT_DATA = Path("data/bfcl")
INDEX = "bfcl-v3-functions"
TOP_K = 10
STANDARD_FIELDS = ["name^3", "description", "params"]
IDENTIFIER_FIELDS = ["name.identifier^3", "description", "params.identifier"]
QWEN_QUERY_TASK = "Given a user request, retrieve the API function that fulfills it"
BOOTSTRAP = 2000
RNG_SEED = 20261030
IDENTIFIER_SETTINGS = {
    "analysis": {
        "filter": {
            IDENTIFIER_FILTER: {
                "type": "word_delimiter_graph",
                "generate_word_parts": True,
                "generate_number_parts": True,
                "split_on_case_change": True,
                "split_on_numerics": True,
                "preserve_original": True,
                "stem_english_possessive": False,
            }
        },
        "analyzer": {
            IDENTIFIER_ANALYZER: {
                "type": "custom",
                "tokenizer": "standard",
                "filter": [IDENTIFIER_FILTER, "lowercase"],
            }
        },
    }
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _split_of(task_id: str) -> str:
    digest = hashlib.sha1(task_id.encode("utf-8")).hexdigest()
    return "dev" if int(digest[:8], 16) % 2 == 0 else "test"


def _user_text(messages: Any) -> str:
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


def _doc_text(function: dict[str, Any]) -> str:
    params = function.get("parameters") or {}
    properties = params.get("properties") or {}
    pieces = [str(function.get("description") or "")]
    for name, spec in properties.items():
        if isinstance(spec, dict):
            pieces.append(f"{name}: {spec.get('description') or ''}")
        else:
            pieces.append(str(name))
    return " ".join(piece for piece in pieces if piece).strip()


def _gold_names(row: dict[str, Any]) -> set[str]:
    gold = row.get("ground_truth")
    if not gold:
        return set()
    names: set[str] = set()
    for entry in gold:
        if isinstance(entry, str):
            names.add(entry.split("(", 1)[0].strip())
        elif isinstance(entry, dict):
            names.update(str(key) for key in entry)
    return names


def _load(
    data: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, str]]:
    sources = {
        "BFCL_v3_simple.json": data / "BFCL_v3_simple.json",
        "possible_answer/BFCL_v3_simple.json": data / "possible_answer" / "BFCL_v3_simple.json",
        "BFCL_v3_exec_simple.json": data / "BFCL_v3_exec_simple.json",
        "BFCL_v3_irrelevance.json": data / "BFCL_v3_irrelevance.json",
    }
    hashes = {name: _sha256(path) for name, path in sources.items()}
    answers = {
        row["id"]: row for row in _read_jsonl(sources["possible_answer/BFCL_v3_simple.json"])
    }

    labeled: list[dict[str, Any]] = []
    for row in _read_jsonl(sources["BFCL_v3_simple.json"]):
        gold = _gold_names(answers.get(row["id"], {}))
        if gold:
            labeled.append(
                {
                    "id": row["id"],
                    "kind": "simple",
                    "split": _split_of(row["id"]),
                    "query": _user_text(row.get("question")),
                    "gold": sorted(gold),
                    "candidates": row.get("function") or [],
                }
            )
    for row in _read_jsonl(sources["BFCL_v3_exec_simple.json"]):
        gold = _gold_names(row)
        if gold:
            labeled.append(
                {
                    "id": row["id"],
                    "kind": "exec_simple",
                    "split": _split_of(row["id"]),
                    "query": _user_text(row.get("question")),
                    "gold": sorted(gold),
                    "candidates": row.get("function") or [],
                }
            )
    controls = [
        {
            "id": row["id"],
            "kind": "irrelevance",
            "split": _split_of(row["id"]),
            "query": _user_text(row.get("question")),
            "gold": [],
            "candidates": row.get("function") or [],
        }
        for row in _read_jsonl(sources["BFCL_v3_irrelevance.json"])
    ]

    corpus: dict[str, dict[str, Any]] = {}
    for row in [*labeled, *controls]:
        for function in row["candidates"]:
            name = str(function.get("name") or "")
            if name and name not in corpus:
                corpus[name] = {
                    "name": name,
                    "description": str(function.get("description") or ""),
                    "params": " ".join(
                        str(key) for key in (function.get("parameters") or {}).get("properties")
                    ),
                    "text": _doc_text(function),
                }
    return labeled, controls, corpus, hashes


def _rank(ids: list[str], scores: np.ndarray, k: int = TOP_K) -> list[str]:
    order = np.argsort(-scores)[:k]
    return [ids[index] for index in order]


def _auc(positives: list[float], negatives: list[float]) -> float:
    """Mann-Whitney AUC with tie handling, no distributional assumption."""
    wins = 0.0
    for positive in positives:
        for negative in negatives:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def _bm25(
    client: Any, corpus: dict[str, dict[str, Any]], queries: list[str], fields: list[str]
) -> list[list[str]]:
    client.indices.delete(index=INDEX, ignore_unavailable=True)
    client.indices.create(
        index=INDEX,
        settings=IDENTIFIER_SETTINGS,
        mappings={
            "properties": {
                "name": {
                    "type": "text",
                    "fields": {"identifier": {"type": "text", "analyzer": IDENTIFIER_ANALYZER}},
                },
                "description": {
                    "type": "text",
                    "fields": {"identifier": {"type": "text", "analyzer": IDENTIFIER_ANALYZER}},
                },
                "params": {
                    "type": "text",
                    "fields": {"identifier": {"type": "text", "analyzer": IDENTIFIER_ANALYZER}},
                },
            }
        },
    )
    operations: list[dict[str, Any]] = []
    for name, doc in corpus.items():
        operations.append({"index": {"_index": INDEX, "_id": name}})
        operations.append({key: doc[key] for key in ("name", "description", "params")})
    client.bulk(operations=operations, refresh=True)
    rankings: list[list[str]] = []
    for query in queries:
        response = client.search(
            index=INDEX,
            size=TOP_K,
            query={
                "multi_match": {
                    "query": query,
                    "fields": fields,
                    "type": "best_fields",
                    "tie_breaker": 0.3,
                }
            },
        )
        rankings.append([hit["_id"] for hit in response["hits"]["hits"]])
    return rankings


def _dense(
    provider: Any, corpus: dict[str, dict[str, Any]], queries: list[str]
) -> tuple[list[list[str]], list[str], np.ndarray, np.ndarray, int]:
    ids = list(corpus)
    matrix = np.asarray(
        provider.embed_documents([corpus[name]["text"] for name in ids]), dtype=float
    )
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12
    # Queries go through embed_query so a configured instruction prefix applies.
    query_vectors = np.asarray([provider.embed_query(query) for query in queries], dtype=float)
    query_vectors /= np.linalg.norm(query_vectors, axis=1, keepdims=True) + 1e-12
    rankings = [_rank(ids, matrix @ vector) for vector in query_vectors]
    return rankings, ids, matrix, query_vectors, matrix.shape[1]


def _fuse(
    bm25_rankings: list[list[str]], dense_rankings: list[list[str]], dense_weight: float
) -> tuple[list[list[str]], list[dict[str, float]]]:
    rankings: list[list[str]] = []
    scores: list[dict[str, float]] = []
    for lexical, dense in zip(bm25_rankings, dense_rankings, strict=True):
        branches = [
            RankedList("bm25", lexical, 1.0),
            RankedList("vector", dense, dense_weight),
        ]
        fused = reciprocal_rank_fusion(branches, k=60, limit=TOP_K)
        rankings.append([hit.doc_id for hit in fused])
        complete = reciprocal_rank_fusion(branches, k=60)
        scores.append({hit.doc_id: hit.score for hit in complete})
    return rankings, scores


def _provided_scores(
    client: Any,
    rows: list[dict[str, Any]],
    ids: list[str],
    matrix: np.ndarray,
    query_vectors: np.ndarray,
    fused_scores: list[dict[str, float]] | None,
    fields: list[str],
) -> list[float]:
    """Score the function each task actually offers, not the whole pool.

    BFCL ``irrelevance`` means the *provided* function does not apply. The pool
    may still contain some other function that does, so a global-pool abstention
    number would be wrong. This matched comparison asks the question BFCL asks:
    should the offered function be applied to this query?
    """
    index = {name: position for position, name in enumerate(ids)}
    scores: list[float] = []
    for position, row in enumerate(rows):
        name = str(row["candidates"][0]["name"])
        if fused_scores is not None:
            scores.append(float(fused_scores[position].get(name, 0.0)))
            continue
        if matrix.size == 0:
            response = client.search(
                index=INDEX,
                size=1,
                query={
                    "bool": {
                        "must": [
                            {
                                "multi_match": {
                                    "query": row["query"],
                                    "fields": fields,
                                    "type": "best_fields",
                                    "tie_breaker": 0.3,
                                }
                            }
                        ],
                        "filter": [{"ids": {"values": [name]}}],
                    }
                },
            )
            hits = response["hits"]["hits"]
            scores.append(float(hits[0]["_score"]) if hits else 0.0)
        else:
            scores.append(float(matrix[index[name]] @ query_vectors[position]))
    return scores


def _per_query(labeled: list[dict[str, Any]], rankings: list[list[str]]) -> dict[str, list[float]]:
    hits: dict[int, list[float]] = {1: [], 3: [], 5: []}
    reciprocal_ranks: list[float] = []
    for row, ranking in zip(labeled, rankings, strict=True):
        gold = set(row["gold"])
        for k in hits:
            hits[k].append(1.0 if gold & set(ranking[:k]) else 0.0)
        position = next((i for i, doc in enumerate(ranking, start=1) if doc in gold), None)
        reciprocal_ranks.append(1.0 / position if position else 0.0)
    return {
        "hit_at_1": hits[1],
        "hit_at_3": hits[3],
        "hit_at_5": hits[5],
        "reciprocal_rank": reciprocal_ranks,
    }


def _bootstrap_ci(values: list[float], rng: np.random.Generator) -> list[float]:
    array = np.asarray(values, dtype=float)
    samples = rng.choice(array, size=(BOOTSTRAP, array.size), replace=True).mean(axis=1)
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def _auc_ci(
    positives: list[float], negatives: list[float], rng: np.random.Generator
) -> list[float]:
    positives_array = np.asarray(positives, dtype=float)
    negatives_array = np.asarray(negatives, dtype=float)
    samples = []
    for _ in range(BOOTSTRAP):
        pos = rng.choice(positives_array, size=positives_array.size, replace=True)
        neg = rng.choice(negatives_array, size=negatives_array.size, replace=True)
        samples.append(_auc(list(pos), list(neg)))
    return [float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))]


def _evaluate(
    labeled: list[dict[str, Any]],
    controls: list[dict[str, Any]],
    rankings: list[list[str]],
    scores: list[float],
    split: str,
    rng: np.random.Generator,
    task_index: dict[str, int],
) -> dict[str, Any]:
    labeled_rows = [row for row in labeled if row["split"] == split]
    control_rows = [row for row in controls if row["split"] == split]
    labeled_rankings = [rankings[task_index[row["id"]]] for row in labeled_rows]
    per_query = _per_query(labeled_rows, labeled_rankings)
    positive_scores = [scores[task_index[row["id"]]] for row in labeled_rows]
    negative_scores = [scores[task_index[row["id"]]] for row in control_rows]
    result: dict[str, Any] = {"n_labeled": len(labeled_rows), "n_controls": len(control_rows)}
    for key, values in per_query.items():
        result[key] = float(np.mean(values))
        result[f"{key}_ci95"] = _bootstrap_ci(values, rng)
    result["auc_vs_irrelevance"] = _auc(positive_scores, negative_scores)
    result["auc_vs_irrelevance_ci95"] = _auc_ci(positive_scores, negative_scores, rng)
    result["provided_scores"] = positive_scores
    result["control_scores"] = negative_scores
    result["query_ids"] = [row["id"] for row in labeled_rows]
    result["ranks"] = [_gold_rank(row, rankings[task_index[row["id"]]]) for row in labeled_rows]
    return result


def _gold_rank(row: dict[str, Any], ranking: list[str]) -> int:
    gold = set(row["gold"])
    return next((position for position, doc in enumerate(ranking, start=1) if doc in gold), 0)


def _paired_hit1(
    treatment: dict[str, Any], reference: dict[str, Any], rng: np.random.Generator
) -> dict[str, Any]:
    """Paired bootstrap of hit@1 differences on the same queries."""
    treated = [1.0 if rank == 1 else 0.0 for rank in treatment["ranks"]]
    base = [1.0 if rank == 1 else 0.0 for rank in reference["ranks"]]
    differences = np.asarray(treated) - np.asarray(base)
    samples = rng.choice(differences, size=(BOOTSTRAP, differences.size), replace=True).mean(axis=1)
    return {
        "n_queries": int(differences.size),
        "observed": float(differences.mean()),
        "cluster_ci95": [
            float(np.quantile(samples, 0.025)),
            float(np.quantile(samples, 0.975)),
        ],
        "excludes_zero": bool(
            np.quantile(samples, 0.025) > 0.0 or np.quantile(samples, 0.975) < 0.0
        ),
    }


def _abstention(dev: dict[str, Any], test: dict[str, Any]) -> dict[str, Any]:
    # Calibrated on dev only, then applied unchanged to test.
    threshold = float(np.quantile(dev["provided_scores"], 0.05))
    coverage = float(np.mean(np.asarray(test["provided_scores"]) >= threshold))
    abstained = float(np.mean(np.asarray(test["control_scores"]) < threshold))
    return {
        "threshold": threshold,
        "identified": threshold > 0.0,
        "test_labeled_coverage": coverage,
        "test_control_abstention": abstained,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--embedding", choices=["hashing", "qwen"], default="qwen")
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("docs/evidence"))
    parser.add_argument("--doc", type=Path, default=None)
    parser.add_argument("--es-url", default="http://localhost:9200")
    parser.add_argument("--analyzer", choices=["standard", "identifier"], default="standard")
    args = parser.parse_args(argv)

    if not args.data_dir.is_dir():
        raise SystemExit(
            f"{args.data_dir} does not exist. Download BFCL v3 into it (see the module "
            "docstring) or pass --data-dir."
        )
    labeled, controls, corpus, hashes = _load(args.data_dir)
    tasks = [*labeled, *controls]
    print(f"labeled={len(labeled)} controls={len(controls)} functions={len(corpus)}")

    if args.embedding == "qwen":
        if args.model_path is None:
            raise SystemExit(
                "--model-path is required for --embedding qwen (a local "
                "sentence-transformers directory or Hugging Face model id)"
            )
        model_path = args.model_path
        settings = EmbeddingSettings(
            provider="sentence-transformers",
            model_path=str(model_path),
            dimension=1024,
            query_prefix=f"Instruct: {QWEN_QUERY_TASK}\nQuery: ",
        )
        model_id = model_path.name
    else:
        settings = EmbeddingSettings(provider="hashing", dimension=256)
        model_id = settings.model_id
    provider = build_embedding_provider(settings)

    fields = STANDARD_FIELDS if args.analyzer == "standard" else IDENTIFIER_FIELDS
    queries = [row["query"] for row in tasks]
    client = build_client(ElasticsearchSettings(url=args.es_url))
    try:
        bm25_rankings = _bm25(client, corpus, queries, fields)
        dense_rankings, ids, matrix, query_vectors, dimension = _dense(provider, corpus, queries)
        variants: dict[str, tuple[list[list[str]], list[float]]] = {
            "bm25": (
                bm25_rankings,
                _provided_scores(client, tasks, ids, np.empty(0), query_vectors, None, fields),
            ),
            "vector": (
                dense_rankings,
                _provided_scores(client, tasks, ids, matrix, query_vectors, None, fields),
            ),
        }
        for weight in (1.0, 2.0, 3.0):
            tag = f"hybrid_rrf_w{weight:g}"
            rankings, fused_scores = _fuse(bm25_rankings, dense_rankings, weight)
            variants[tag] = (
                rankings,
                _provided_scores(
                    client, tasks, ids, np.empty(0), query_vectors, fused_scores, fields
                ),
            )
    finally:
        client.close()

    rng = np.random.default_rng(RNG_SEED)
    task_index = {row["id"]: position for position, row in enumerate(tasks)}
    dev: dict[str, Any] = {}
    test: dict[str, Any] = {}
    for name, (rankings, scores) in variants.items():
        dev[name] = _evaluate(labeled, controls, rankings, scores, "dev", rng, task_index)
        test[name] = _evaluate(labeled, controls, rankings, scores, "test", rng, task_index)

    # Weight selection uses dev only; test is read once, after selection.
    best = max(dev, key=lambda name: (dev[name]["hit_at_1"], dev[name]["reciprocal_rank"]))
    results = {
        name: {
            "dev": {
                key: value
                for key, value in dev[name].items()
                if "scores" not in key and key not in {"ranks", "query_ids"}
            },
            "test": {
                key: value
                for key, value in test[name].items()
                if "scores" not in key and key not in {"ranks", "query_ids"}
            },
            "abstention": _abstention(dev[name], test[name]),
        }
        for name in variants
    }
    per_query = {
        name: {
            split: [
                {"id": query_id, "rank": rank}
                for query_id, rank in zip(
                    (dev if split == "dev" else test)[name]["query_ids"],
                    (dev if split == "dev" else test)[name]["ranks"],
                    strict=True,
                )
            ]
            for split in ("dev", "test")
        }
        for name in variants
    }
    paired = {
        f"{name}_minus_vector": _paired_hit1(test[name], test["vector"], rng)
        for name in variants
        if name != "vector"
    }

    payload = {
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset": "gorilla-llm/Berkeley-Function-Calling-Leaderboard (BFCL v3)",
        "source_hashes": hashes,
        "labeled_tasks": len(labeled),
        "control_tasks": len(controls),
        "dev_labeled": sum(1 for row in labeled if row["split"] == "dev"),
        "test_labeled": sum(1 for row in labeled if row["split"] == "test"),
        "function_pool": len(corpus),
        "top_k": TOP_K,
        "bootstrap_resamples": BOOTSTRAP,
        "rng_seed": RNG_SEED,
        "embedding": {"provider": settings.provider, "model": model_id, "dimension": dimension},
        "rrf_k": 60,
        "analyzer": args.analyzer,
        "selected_variant": best,
        "selection_rule": "highest dev hit@1, tie-broken by dev MRR; test never used",
        "results": results,
        "per_query": per_query,
        "paired_vs_vector": paired,
        "protocol_changes": [
            "Queries are embedded through embed_query with Qwen's instruction prefix; "
            "documents are embedded without it.",
            "The lexical channel can use either the default analyzer (`--analyzer "
            "standard`) or RECAP's identifier-aware `recap_identifier` subfields on "
            "name/params (`--analyzer identifier`). Identifier analysis on prose "
            "descriptions was tested on dev and rejected: it lowered dev BM25 hit@1.",
        ],
        "limits": [
            "Function-pool retrieval, not end-to-end function-call execution.",
            "The pool is the 630 unique functions present in these BFCL files; it is not "
            "the full BFCL function universe.",
            "Irrelevance controls score only the single function each task offers. That "
            "matches BFCL's definition (the provided function does not apply), but it is a "
            "matched single-candidate comparison, not a 630-way pool ranking.",
            "BM25 scores are query-length sensitive, so its control AUC is a weaker "
            "diagnostic than the cosine and RRF ones.",
        ],
    }
    args.out.mkdir(parents=True, exist_ok=True)
    json_path = args.out / f"real-data-bfcl-{args.embedding}.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"selected={best}")
    print(f"wrote {json_path}")

    if args.doc:
        args.doc.parent.mkdir(parents=True, exist_ok=True)
        args.doc.write_text(render_markdown(payload), encoding="utf-8")
        print(f"wrote {args.doc}")
    return 0


def render_markdown(payload: dict[str, Any]) -> str:
    results = payload["results"]
    selected = payload["selected_variant"]
    lines = [
        "# Retrieval on real data: BFCL v3",
        "",
        "> Generated by `scripts/bfcl_retrieval.py` from the JSON report beside this file. "
        "Do not hand-edit the numbers.",
        "",
        "## What is real and what is not",
        "",
        "Real: every query, function schema, function description, and distractor function "
        "comes from BFCL v3. There are no synthetic decoys and no synthetic gold fixtures.",
        "",
        "Not measured here: end-to-end tool execution and failure-memory transfer. This "
        "report covers the retrieval core only (BM25, dense, RRF).",
        "",
        "## Provenance",
        "",
        "| Item | Value |",
        "|---|---|",
        f"| Dataset | `{payload['dataset']}` |",
        f"| Generated | `{payload['generated_at']}` |",
        f"| Labeled tasks | {payload['labeled_tasks']} "
        f"(dev {payload['dev_labeled']} / test {payload['test_labeled']}) |",
        f"| Control tasks (irrelevance) | {payload['control_tasks']} |",
        f"| Function pool | {payload['function_pool']} |",
        f"| Embedder | `{payload['embedding']['provider']}` / "
        f"`{payload['embedding']['model']}` ({payload['embedding']['dimension']}d) |",
        f"| RRF k | {payload['rrf_k']} |",
        f"| Lexical analyzer | `{payload['analyzer']}` |",
        "",
        "## Test results (headline)",
        "",
        "| Method | hit@1 (95% CI) | hit@3 | hit@5 | MRR@10 | AUC vs irrelevance | "
        "Control abstention @95% coverage |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, stats in results.items():
        test = stats["test"]
        abstention = stats["abstention"]
        marker = " **(selected)**" if name == selected else ""
        abstain_text = (
            f"{abstention['test_control_abstention']:.3f}"
            if abstention.get("identified", True)
            else "n/a (threshold 0)"
        )
        lines.append(
            f"| `{name}`{marker} | {test['hit_at_1']:.3f} "
            f"[{test['hit_at_1_ci95'][0]:.3f}, {test['hit_at_1_ci95'][1]:.3f}] | "
            f"{test['hit_at_3']:.3f} | {test['hit_at_5']:.3f} | "
            f"{test['reciprocal_rank']:.3f} | {test['auc_vs_irrelevance']:.3f} | "
            f"{abstain_text} |"
        )
    lines.extend(
        [
            "",
            f"Selected by `{payload['selection_rule']}`: `{selected}`.",
            "",
            "## Protocol change (2026-10-31)",
            "",
            "Earlier numbers used the default analyzer and embedded queries as documents. "
            "This run adds two changes instead of one: Qwen's instruction prefix through "
            "`embed_query`, and an optional identifier-aware lexical channel. The earlier "
            "report is kept beside this one as `-v1`.",
            "",
            "### Analyzer result (dev-selected, test reported once)",
            "",
            "Identifier analysis on *prose* descriptions was tested first and lowered dev "
            "BM25 hit@1, so it was rejected on dev. The `standard` analyzer is therefore the "
            "canonical configuration; `--analyzer identifier` remains available for "
            "reproduction and applies identifier analysis only to name/params subfields.",
            "",
            "## Paired effects vs dense (test)",
            "",
            "| Comparison | Queries | Observed hit@1 difference | 95% CI | Excludes zero |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for name, stats in payload.get("paired_vs_vector", {}).items():
        lines.append(
            f"| `{name}` | {stats['n_queries']} | {stats['observed']:+.3f} | "
            f"[{stats['cluster_ci95'][0]:+.3f}, {stats['cluster_ci95'][1]:+.3f}] | "
            f"{stats['excludes_zero']} |"
        )
    lines.extend(
        [
            "",
            "## Dev results (tuning only)",
            "",
            "| Method | hit@1 | hit@3 | MRR@10 | AUC vs irrelevance |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, stats in results.items():
        dev = stats["dev"]
        lines.append(
            f"| `{name}` | {dev['hit_at_1']:.3f} | {dev['hit_at_3']:.3f} | "
            f"{dev['reciprocal_rank']:.3f} | {dev['auc_vs_irrelevance']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Abstention thresholds are taken from dev labeled scores at 95% coverage and "
            "applied unchanged to test; test control abstention is reported at that fixed "
            "operating point.",
            "",
            "## Limits",
            "",
        ]
    )
    for limit in payload["limits"]:
        lines.append(f"- {limit}")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
