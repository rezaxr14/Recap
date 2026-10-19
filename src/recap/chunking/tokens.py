"""Approximate, download-free token estimation.

RECAP must not fetch tokenizer vocabularies (for example ``tiktoken``) on the
chunking path. This counter is deliberately simple and conservative; the
configured multiplier provides headroom below provider limits.
"""

from __future__ import annotations

import math
import re

# Words, numbers/identifiers, and single punctuation/whitespace-delimited marks.
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+|[^\sA-Za-z0-9_]")


def count_raw_tokens(text: str) -> int:
    return len(TOKEN_PATTERN.findall(text))


def estimate_tokens(text: str, multiplier: float = 1.3) -> int:
    """Return a conservative token estimate for ``text``.

    The estimate is monotonic and deterministic. It is not a substitute for a
    model tokenizer, so chunk budgets must leave margin for the real tokenizer.
    """

    if not text:
        return 0
    return max(1, math.ceil(count_raw_tokens(text) * multiplier))
