"""Typed error hierarchy.

Errors crossing API/CLI boundaries carry a stable machine-readable ``code`` so
clients never need to parse prose or stack traces.
"""

from __future__ import annotations

from typing import Any


class RecapError(Exception):
    """Base class for all RECAP errors."""

    code = "recap_error"
    http_status = 500

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "details": self.details}


class ValidationError(RecapError):
    code = "validation_error"
    http_status = 422


class NotFoundError(RecapError):
    code = "not_found"
    http_status = 404


class ConflictError(RecapError):
    code = "conflict"
    http_status = 409


class SafetyError(RecapError):
    code = "safety_violation"
    http_status = 403


class TransitionError(RecapError):
    code = "invalid_transition"
    http_status = 409


class PromotionRejected(RecapError):
    code = "promotion_rejected"
    http_status = 422


class SearchUnavailable(RecapError):
    """Elasticsearch is unreachable or returned an unrecoverable error."""

    code = "search_unavailable"
    http_status = 503


class EmbeddingUnavailable(RecapError):
    code = "embedding_unavailable"
    http_status = 503


class ConfigurationError(RecapError):
    code = "configuration_error"
    http_status = 500


class IdempotencyConflict(RecapError):
    code = "idempotency_conflict"
    http_status = 409
