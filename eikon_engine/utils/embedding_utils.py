"""Lightweight embedding helpers used by planners and memory."""

from __future__ import annotations

import hashlib
from typing import Iterable, List


def embed_text(text: str) -> List[float]:
    """Return a deterministic pseudo-embedding for the provided text."""

    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return [int(b) / 255.0 for b in digest[:16]]


def average_embedding(chunks: Iterable[str]) -> List[float]:
    """Compute an average embedding over multiple chunks."""

    vectors = [embed_text(chunk) for chunk in chunks]
    if not vectors:
        return [0.0] * 16
    summed = [sum(values) for values in zip(*vectors)]
    return [value / len(vectors) for value in summed]
