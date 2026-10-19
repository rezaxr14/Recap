"""Build typed retrieval requests and redacted query text."""

from __future__ import annotations

from recap.domain.retrieval import RetrievalContext
from recap.recorder.redaction import redact_payload

MAX_QUERY_CHARS = 2000


def build_query_text(context: RetrievalContext) -> str:
    """Combine natural-language and typed signals for lexical retrieval.

    Exact codes/tool names are appended so they are not lost to prose.
    """
    parts = [context.query.strip()]
    if context.error_type:
        parts.append(f"error_type:{context.error_type}")
    if context.error_code:
        parts.append(f"error_code:{context.error_code}")
    if context.failure_signature:
        parts.append(context.failure_signature)
    if context.task_family:
        parts.append(f"task_family:{context.task_family}")
    if context.tool_names:
        parts.append("tools:" + " ".join(context.tool_names))
    if context.error_message:
        parts.append(context.error_message.strip())
    text = " \n".join(part for part in parts if part)
    return text[:MAX_QUERY_CHARS]


def redact_context(context: RetrievalContext) -> RetrievalContext:
    payload = context.model_dump(mode="json")
    redacted, _ = redact_payload(payload)
    redacted["query"] = redacted.get("query", "")[:MAX_QUERY_CHARS]
    return RetrievalContext.model_validate(redacted)
