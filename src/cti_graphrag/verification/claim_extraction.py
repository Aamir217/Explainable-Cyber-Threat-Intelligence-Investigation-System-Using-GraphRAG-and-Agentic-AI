"""Split a generated answer into individually-checkable claims."""

from __future__ import annotations

import re

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def extract_claims(answer_text: str) -> list[str]:
    claims = [s.strip() for s in _SENTENCE_RE.split(answer_text) if s.strip()]
    return claims
