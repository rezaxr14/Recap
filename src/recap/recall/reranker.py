"""Reranking stage.

Fusion (RRF) is rank-based and deliberately conservative: it cannot tell that a
document contains the exact error code while another only shares topic words.
A reranker is an optional second pass over the fused candidate set that can
reorder candidates using richer signals before the safety policy runs.

The default implementation is deterministic and lexical. It exists so
``hybrid_rerank`` is a real mode rather than an alias for fusion, and so the
reranker interface can be exercised without a model download or API key. A
cross-encoder can implement the same protocol later.

Reranking never overrides safety: it only changes candidate order. Eligibility,
applicability, compatibility, and utility are evaluated afterwards.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

_TOKEN_RE = re.compile(r"[a-z0-9_]+")

# Small stopword set; enough to keep generic task words from dominating.
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "to",
        "with",
    }
)


def tokenize(text: str) -> list[str]:
    return [token for token in _TOKEN_RE.findall(text.lower()) if token not in _STOPWORDS]


class LexicalReranker:
    """Deterministic query-document lexical overlap reranker.

    Score = 0.7 * query coverage + 0.3 * document precision.

    Query coverage rewards documents that contain the distinctive query terms
    (error code, tool name); document precision penalizes long documents that
    happen to contain one common term. The weights are fixed and documented
    rather than tuned on the test split.
    """

    name = "lexical-overlap-v1"

    def rerank(self, query: str, documents: Sequence[str], top_k: int) -> list[tuple[int, float]]:
        if top_k <= 0 or not documents:
            return []
        query_terms = set(tokenize(query))
        scored: list[tuple[int, float]] = []
        for index, document in enumerate(documents):
            document_terms = set(tokenize(document))
            if not query_terms or not document_terms:
                scored.append((index, 0.0))
                continue
            overlap = len(query_terms & document_terms)
            coverage = overlap / len(query_terms)
            precision = overlap / len(document_terms)
            scored.append((index, 0.7 * coverage + 0.3 * precision))
        # Stable sort: ties keep fusion order, so the result is reproducible.
        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored[:top_k]


def build_reranker(name: str) -> LexicalReranker | None:
    if name == "none":
        return None
    if name == "lexical":
        return LexicalReranker()
    raise ValueError(f"unknown reranker: {name}")
