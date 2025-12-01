from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

try:  # pragma: no cover - optional dependency
    from openai import OpenAI  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None

try:  # pragma: no cover - optional dependency
    from sentence_transformers import SentenceTransformer  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    SentenceTransformer = None

from src.memory.vector_store import VectorStore

LOGGER = logging.getLogger(__name__)


@dataclass
class MemoryRecord:
    """High-level memory record used by legacy components."""

    kind: str
    payload: Dict[str, Any]
    embedding: List[float]


class MemoryManager:
    """Production-ready memory manager with persistence and fallbacks."""

    def __init__(
        self,
        store: VectorStore | None = None,
        embedding_model: str | None = None,
        openai_key: str | None = None,
    ) -> None:
        self._store = store or VectorStore()
        self._embedding_model = embedding_model or "all-MiniLM-L6-v2"
        self._openai_key = openai_key or os.getenv("OPENAI_API_KEY")
        self._openai_client = None
        if self._openai_key and OpenAI is not None:
            try:
                self._openai_client = OpenAI(api_key=self._openai_key)
            except Exception as exc:  # pragma: no cover - network issues
                LOGGER.warning("Failed to initialize OpenAI client: %s", exc)
                self._openai_client = None
        self._sentence_model: Optional[Any] = None
        self._max_retries = 3

        self._store.set_embedder(self._embed_sync)

    # Legacy API ---------------------------------------------------------
    def store(self, namespace: str, record: MemoryRecord) -> None:
        record_id = f"{namespace}:{record.kind}:{uuid.uuid4().hex}"
        vector = np.asarray(record.embedding, dtype="float32")
        metadata = {"namespace": namespace, "kind": record.kind, **record.payload}
        self._store.add_vector(record_id, vector, metadata)

    def search(self, namespace: str, query_embedding: List[float], top_k: int = 5) -> List[Dict[str, Any]]:
        vector = np.asarray(query_embedding, dtype="float32")
        results = self._store.query_by_vector(vector, top_k)
        return [result for result in results if result["metadata"].get("namespace") == namespace]

    # New async API ------------------------------------------------------
    async def add_memory(self, event_type: str, title: str, text: str, context: Dict[str, Any]) -> str:
        memory_id = str(uuid.uuid4())
        metadata = {
            "type": event_type,
            "title": title,
            "context": context,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        await asyncio.to_thread(self._store.add, memory_id, text, metadata)
        return memory_id

    async def query_similar(self, text: str, k: int = 5) -> List[Dict[str, Any]]:
        return await asyncio.to_thread(self._store.query, text, k)

    def save(self) -> None:
        self._store.save()

    def load(self) -> None:
        self._store.load()

    # Embedding helpers --------------------------------------------------
    def _embed_sync(self, text: str) -> np.ndarray:
        last_error: Exception | None = None
        delay = 0.5
        for attempt in range(1, self._max_retries + 1):
            try:
                if self._openai_client is not None:
                    response = self._openai_client.embeddings.create(
                        model="text-embedding-3-large",
                        input=text,
                    )
                    vector = np.asarray(response.data[0].embedding, dtype="float32")
                else:
                    vector = self._local_sentence_embedding(text)
                return vector
            except Exception as exc:
                last_error = exc
                if attempt == self._max_retries:
                    break
                time.sleep(delay)
                delay = min(delay * 2, 4.0)
        LOGGER.error("Failed to compute embedding: %s", last_error)
        raise RuntimeError("Failed to compute embedding") from last_error

    def _local_sentence_embedding(self, text: str) -> np.ndarray:
        if SentenceTransformer is None:
            return _hashed_embedding(text)
        if self._sentence_model is None:
            try:
                self._sentence_model = SentenceTransformer(self._embedding_model)
            except Exception as exc:  # pragma: no cover - download issues
                LOGGER.warning("Failed to load sentence-transformer: %s", exc)
                return _hashed_embedding(text)
        try:
            vector = self._sentence_model.encode(text)
            return np.asarray(vector, dtype="float32")
        except Exception as exc:  # pragma: no cover - runtime issues
            LOGGER.warning("sentence-transformer encode failed: %s", exc)
            return _hashed_embedding(text)


def _hashed_embedding(text: str, dim: int = 384) -> np.ndarray:
    seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=dim).astype("float32")
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm
