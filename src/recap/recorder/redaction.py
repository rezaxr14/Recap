"""Secret redaction at the persistence and indexing boundary.

Pattern-based redaction is defense-in-depth, not a guarantee. Callers should
avoid sending secrets in the first place. The module is deterministic and
covered by security tests.
"""

from __future__ import annotations

import re
from typing import Any

REDACTION_RULE_VERSION = "redaction-v1"
REDACTED = "[REDACTED]"

# Keys whose values are always redacted regardless of shape.
SENSITIVE_KEYS = {
    "password",
    "passwd",
    "secret",
    "api_key",
    "apikey",
    "access_token",
    "refresh_token",
    "token",
    "authorization",
    "auth",
    "cookie",
    "set-cookie",
    "private_key",
    "client_secret",
    "session_id",
    "credential",
    "credentials",
    "aws_access_key_id",
    "aws_secret_access_key",
}

# Key fragments that indicate a sensitive key (e.g. x-api-key).
SENSITIVE_KEY_FRAGMENTS = ("password", "secret", "token", "api_key", "apikey", "private_key")

PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S), "[REDACTED_PRIVATE_KEY]"),
    (re.compile(r"\bBearer\s+[A-Za-z0-9._\-~+/]{12,}", re.I), f"Bearer {REDACTED}"),
    (re.compile(r"\b(sk|pk|rk)[-_](live|test|proj)[-_][A-Za-z0-9]{10,}\b"), REDACTED),
    (re.compile(r"\bsk-[A-Za-z0-9]{16,}\b"), REDACTED),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), REDACTED),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"), REDACTED),
    (re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b"), REDACTED),
    (re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"), REDACTED),
    (
        re.compile(
            r"(?i)\b(api[_-]?key|access[_-]?token|refresh[_-]?token|token|password|secret)"
            r"\b\s*[:=]\s*['\"]?([^\s'\",;]{6,})"
        ),
        lambda m: f"{m.group(1)}={REDACTED}",
    ),
    (
        re.compile(r"(?i)([?&](?:api[_-]?key|access[_-]?token|token|sig|signature)=)([^&\s]+)"),
        lambda m: f"{m.group(1)}{REDACTED}",
    ),
]


def _is_sensitive_key(key: str) -> bool:
    lowered = key.lower().replace("-", "_")
    if lowered in SENSITIVE_KEYS:
        return True
    return any(fragment in lowered for fragment in SENSITIVE_KEY_FRAGMENTS)


def redact_text(text: str) -> tuple[str, bool]:
    applied = False
    for pattern, replacement in PATTERNS:
        new_text, count = pattern.subn(replacement, text)  # type: ignore[arg-type]
        if count:
            applied = True
            text = new_text
    return text, applied


def redact_value(value: Any) -> tuple[Any, bool]:
    """Recursively redact a JSON-compatible value."""
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, dict):
        applied = False
        result: dict[str, Any] = {}
        for key, item in value.items():
            key_str = str(key)
            if _is_sensitive_key(key_str):
                result[key_str] = REDACTED
                applied = True
                continue
            redacted, child_applied = redact_value(item)
            result[key_str] = redacted
            applied = applied or child_applied
        return result, applied
    if isinstance(value, (list, tuple)):
        applied = False
        items = []
        for item in value:
            redacted, child_applied = redact_value(item)
            items.append(redacted)
            applied = applied or child_applied
        return items, applied
    return value, False


def redact_payload(payload: dict[str, Any]) -> tuple[dict[str, Any], bool]:
    redacted, applied = redact_value(payload)
    if not isinstance(redacted, dict):  # pragma: no cover - defensive
        return {}, True
    return redacted, applied


def contains_secret(text: str) -> bool:
    _, applied = redact_text(text)
    return applied
