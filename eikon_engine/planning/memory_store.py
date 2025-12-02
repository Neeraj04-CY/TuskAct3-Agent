"""Rich in-memory store combining short, long, and vector memories."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, List

from eikon_engine.utils.embedding_utils import average_embedding, embed_text


@dataclass
class MemoryRecord:
    """Stores a short snippet, its embedding, and optional metadata."""

    text: str
    embedding: List[float]
    metadata: Dict[str, Any] = field(default_factory=dict)


def _similarity(left: List[float], right: List[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


@dataclass
class MemoryStore:
    """Multi-tier memory with lightweight semantic retrieval."""

    short_term: Deque[Dict[str, Any]] = field(default_factory=lambda: deque(maxlen=5))
    long_term: List[str] = field(default_factory=list)
    vector_memory: List[MemoryRecord] = field(default_factory=list)

    def add(self, key: str, text: str) -> None:
        """Backward-compatible helper for seeding long-term memory."""

        self.vector_memory.append(MemoryRecord(text=text, embedding=embed_text(text), metadata={"key": key}))

    def remember(self, step: Dict[str, Any]) -> None:
        """Track a new step inside short-term and vector stores."""

        summary = step.get("summary") or step.get("description") or step.get("action") or str(step)
        entry = {"summary": summary, **step}
        self.short_term.append(entry)
        self.vector_memory.append(MemoryRecord(text=summary, embedding=embed_text(summary), metadata=step))
        if len(self.vector_memory) > 64:
            self.vector_memory.pop(0)

    def summarize(self) -> str:
        """Condense short-term memories into a long-term summary."""

        if not self.short_term:
            return ""
        recent = list(self.short_term)
        summary = "; ".join(entry.get("summary", "step") for entry in recent)
        self.long_term.append(summary)
        vector = average_embedding(entry.get("summary", "") for entry in recent)
        self.vector_memory.append(MemoryRecord(text=summary, embedding=vector, metadata={"type": "summary"}))
        return summary

    def retrieve(self, query: str, limit: int = 3) -> List[MemoryRecord]:
        """Return the most similar memories to the provided query."""

        if not self.vector_memory:
            return []
        query_vec = embed_text(query)
        ranked = sorted(
            self.vector_memory,
            key=lambda record: _similarity(record.embedding, query_vec),
            reverse=True,
        )
        return ranked[:limit]

    def similar(self, query: str, limit: int = 3) -> List[MemoryRecord]:
        """Alias for retrieve to maintain compatibility."""

        return self.retrieve(query, limit=limit)
