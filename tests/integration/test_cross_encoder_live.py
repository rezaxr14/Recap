"""Local CrossEncoder reranker acceptance.

This test never downloads weights. It runs only when the operator has already
placed ``cross-encoder/ms-marco-MiniLM-L-6-v2`` in the repository's ignored
``models/`` directory.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from recap.recall.cross_encoder import CrossEncoderReranker

pytestmark = pytest.mark.integration

MODEL_PATH = Path("models/ms-marco-MiniLM-L-6-v2").resolve()


@pytest.mark.skipif(
    not MODEL_PATH.is_dir(),
    reason="local CrossEncoder weights are not present",
)
def test_local_cross_encoder_prefers_distinctive_failure_document() -> None:
    reranker = CrossEncoderReranker(str(MODEL_PATH), batch_size=4)
    query = "ValidationError INVALID_CURSOR list_records cursor expired"
    documents = [
        "Generic guidance about retrying API calls with exponential backoff.",
        "When list_records returns ValidationError INVALID_CURSOR, refresh the cursor.",
        "Pagination overview and tenant record counts.",
    ]

    started = time.perf_counter()
    ranked = reranker.rerank(query, documents, top_k=3)
    elapsed_ms = (time.perf_counter() - started) * 1000

    assert [index for index, _ in ranked][0] == 1
    assert ranked[0][1] >= ranked[-1][1]
    assert elapsed_ms < 10_000
