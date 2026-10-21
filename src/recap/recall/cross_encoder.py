"""Local CrossEncoder reranker.

This adapter is deliberately local-only. It requires an existing directory
containing a sentence-transformers CrossEncoder model and passes
``local_files_only=True`` so a typo in the path cannot trigger a download.
The optional ``sentence-transformers`` dependency is imported lazily so the
default lexical path and unit tests do not need it installed.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from recap.errors import ConfigurationError


class CrossEncoderReranker:
    """Pairwise query-document scorer backed by a local CrossEncoder."""

    def __init__(
        self,
        model_path: str,
        *,
        max_length: int = 512,
        batch_size: int = 16,
    ) -> None:
        path = Path(model_path).expanduser()
        if not path.is_dir():
            raise ConfigurationError(
                "cross_encoder reranker requires an existing local model directory; "
                "RECAP never downloads reranker weights",
                details={"rerank_model_path": str(path)},
            )
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ConfigurationError(
                "cross_encoder reranker requires the optional 'rerank' extra: "
                "pip install -e '.[rerank]'"
            ) from exc

        self.model_path = str(path.resolve())
        self.name = f"cross-encoder:{path.name}"
        self.max_length = max_length
        self.batch_size = batch_size
        self._model: Any = CrossEncoder(
            self.model_path,
            max_length=max_length,
            local_files_only=True,
        )

    def rerank(self, query: str, documents: Sequence[str], top_k: int) -> list[tuple[int, float]]:
        if top_k <= 0 or not documents:
            return []
        pairs = [(query, document) for document in documents]
        raw_scores = self._model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        scored: list[tuple[int, float]] = []
        for index, raw in enumerate(raw_scores):
            score = float(raw)
            # CrossEncoder heads are usually already sigmoid-activated. Keep
            # the contract bounded even for raw-logit heads.
            if math.isnan(score) or math.isinf(score):
                score = 0.0
            scored.append((index, max(0.0, min(1.0, score))))
        # Stable sort keeps fusion order for exact ties.
        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored[:top_k]
