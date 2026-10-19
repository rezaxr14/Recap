"""Deterministic failure-signature normalization.

Similar instances normalize together while materially different errors stay
distinct. Volatile values (UUIDs, timestamps, request IDs, ports, hex IDs) are
masked; semantically important tokens (HTTP status, error codes, parameter names,
tool names) are preserved.
"""

from __future__ import annotations

import re

UUID_RE = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")
ISO_TS_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?\b")
HEX_ID_RE = re.compile(r"\b(?:0x)?[0-9a-f]{16,}\b", re.I)
NUMERIC_ID_RE = re.compile(r"\b(?:req|request|trace|span|id|run)[-_]?[A-Za-z0-9]{6,}\b", re.I)
IP_RE = re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(?::\d+)?\b")
PORT_RE = re.compile(r"(?<=:)\d{4,5}\b")
WHITESPACE_RE = re.compile(r"\s+")

# Preserve these important shapes before generic number masking.
HTTP_STATUS_RE = re.compile(r"\b(?:HTTP|status(?:[ _]code)?)\s*[:=]?\s*(\d{3})\b", re.I)
ERROR_CODE_RE = re.compile(r"\b([A-Z][A-Z0-9_]{3,})\b")
QUOTED_PARAM_RE = re.compile(r"['\"`]([a-zA-Z_][a-zA-Z0-9_.]*)['\"`]")
PATH_PARAM_RE = re.compile(r"\b([a-z][a-z0-9_]{2,})\s*(?:=|:)")


def normalize_signature(
    *,
    message: str,
    error_type: str = "",
    error_code: str = "",
    tool_name: str = "",
) -> str:
    """Build a normalized signature string.

    Format: ``<ErrorType>|<CODE>|<tool>|<normalized message>``. Empty components
    are omitted so signatures remain stable as optional metadata appears.
    """
    text = message or ""
    preserved: list[str] = []

    for match in HTTP_STATUS_RE.finditer(text):
        preserved.append(f"http={match.group(1)}")
    for match in ERROR_CODE_RE.finditer(text):
        preserved.append(f"code={match.group(1)}")
    for match in QUOTED_PARAM_RE.finditer(text):
        preserved.append(f"param={match.group(1)}")

    text = UUID_RE.sub("<uuid>", text)
    text = ISO_TS_RE.sub("<timestamp>", text)
    text = NUMERIC_ID_RE.sub("<id>", text)
    text = IP_RE.sub("<ip>", text)
    text = HEX_ID_RE.sub("<hex>", text)
    text = re.sub(r"\b\d{6,}\b", "<num>", text)
    text = WHITESPACE_RE.sub(" ", text).strip().lower()

    # Keep a bounded prefix; stack traces beyond this are noise for matching.
    if len(text) > 300:
        text = text[:300].rstrip()

    parts = [error_type or "unknown", error_code or "", tool_name or "", text]
    unique_preserved = list(dict.fromkeys(preserved))
    signature = "|".join(parts)
    if unique_preserved:
        signature += "|" + "|".join(unique_preserved)
    return signature


def signature_tokens(signature: str) -> list[str]:
    return [token for token in re.split(r"[|_\s]+", signature.lower()) if token]
