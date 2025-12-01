"""Vector store implementation with FAISS support and persistence."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np

try:  # pragma: no cover - optional dependency
    import faiss  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    faiss = None

try:  # pragma: no cover - optional dependency
    from sentence_transformers import SentenceTransformer  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    SentenceTransformer = None

LOGGER = logging.getLogger(__name__)


class VectorStore:
    """Lightweight vector store with FAISS and in-memory fallbacks."""

    def __init__(
        self,
        index_path: str | None = None,
        use_faiss: bool = True,
    ) -> None:
        self._base_dir = Path("data/memory")
        self._base_dir.mkdir(parents=True, exist_ok=True)
        default_ext = "faiss" if use_faiss and faiss is not None else "npy"
        self.index_path = Path(index_path) if index_path else self._base_dir / f"index.{default_ext}"
        self.meta_path = self.index_path.with_name("meta.json")
        self.cache_dir = Path("data/embeddings_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._use_faiss = use_faiss and faiss is not None
        if use_faiss and faiss is None:
            LOGGER.warning("FAISS not available, falling back to in-memory search.")

        self._ids: List[str] = []
        self._metadata: Dict[str, Dict[str, Any]] = {}
        self._vectors: List[np.ndarray] = []
        self._embedding_dim: Optional[int] = None
        self._index = None
        self._sentence_model: Optional[Any] = None
        self._embedding_fn: Optional[Callable[[str], np.ndarray]] = None

        if self.meta_path.exists():
            self.load()

    def set_embedder(self, embedder: Callable[[str], np.ndarray]) -> None:
        """Allow callers (e.g., MemoryManager) to supply a custom embedder."""

        self._embedding_fn = embedder

    def add(self, item_id: str, text: str, metadata: Dict[str, Any]) -> None:
        """Add a text entry to the store."""

        vector = self._get_or_create_embedding(item_id, text)
        self._add_vector(item_id, vector, metadata)

    def add_vector(self, item_id: str, vector: np.ndarray, metadata: Dict[str, Any]) -> None:
        """Insert a pre-computed embedding (compatibility helper)."""

        self._add_vector(item_id, np.asarray(vector, dtype="float32"), metadata)

    def query(self, text: str, k: int = 5) -> List[Dict[str, Any]]:
        """Return the top-k matches for the provided text."""

        if not self._ids:
            return []
        vector = self._compute_embedding(text)
        return self.query_by_vector(vector, k)

    def query_by_vector(self, vector: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """Return the top-k matches for a pre-computed embedding."""

        if not self._ids:
            return []

        k = min(k, len(self._ids))
        vector = np.asarray(vector, dtype="float32")
        if self._use_faiss:
            self._ensure_faiss_index()
            distances, indices = self._index.search(vector.reshape(1, -1), k)
            scored = [
                (self._ids[idx], float(distances[0][pos]))
                for pos, idx in enumerate(indices[0])
                if idx != -1
            ]
            scored.sort(key=lambda item: item[1])
            return [
                {
                    "id": item_id,
                    "score": _distance_to_score(distance),
                    "metadata": self._metadata[item_id],
                }
                for item_id, distance in scored
            ]

        matrix = np.vstack(self._vectors) if self._vectors else np.zeros((0, self._embedding_dim or 1))
        if matrix.size == 0:
            return []
        vector_norm = _safe_norm(vector)
        sims = matrix @ vector / (np.linalg.norm(matrix, axis=1) * vector_norm + 1e-10)
        order = np.argsort(-sims)[:k]
        results = []
        for idx in order:
            item_id = self._ids[idx]
            results.append(
                {
                    "id": item_id,
                    "score": float(sims[idx]),
                    "metadata": self._metadata[item_id],
                }
            )
        return results

    def save(self) -> None:
        """Persist the index and metadata to disk."""

        self.meta_path.parent.mkdir(parents=True, exist_ok=True)
        with self.meta_path.open("w", encoding="utf-8") as handler:
            json.dump(
                {
                    "ids": self._ids,
                    "metadata": self._metadata,
                    "embedding_dim": self._embedding_dim,
                    "use_faiss": self._use_faiss,
                },
                handler,
            )

        if self._use_faiss:
            if self._index is None:
                self._ensure_faiss_index()
            faiss.write_index(self._index, str(self.index_path))
            return

        if self._vectors:
            matrix = np.vstack(self._vectors)
            np.save(self.index_path, matrix)
        elif self.index_path.exists():
            self.index_path.unlink()

    def load(self) -> None:
        """Load metadata and vectors from disk if available."""

        if not self.meta_path.exists():
            return
        with self.meta_path.open("r", encoding="utf-8") as handler:
            payload = json.load(handler)
        self._ids = payload.get("ids", [])
        self._metadata = payload.get("metadata", {})
        self._embedding_dim = payload.get("embedding_dim")
        self._use_faiss = payload.get("use_faiss", self._use_faiss) and faiss is not None

        if self._use_faiss:
            if self.index_path.exists():
                self._index = faiss.read_index(str(self.index_path))
        else:
            if self.index_path.exists():
                matrix = np.load(self.index_path)
                self._vectors = [row.astype("float32") for row in matrix]

    def _add_vector(self, item_id: str, vector: np.ndarray, metadata: Dict[str, Any]) -> None:
        vector = vector.astype("float32").reshape(-1)
        if self._embedding_dim is None:
            self._embedding_dim = vector.shape[0]
        elif vector.shape[0] != self._embedding_dim:
            raise ValueError("Embedding dimension mismatch.")

        if item_id in self._metadata:
            LOGGER.debug("Overwriting existing embedding for id %s", item_id)
            idx = self._ids.index(item_id)
            self._ids.pop(idx)
            if not self._use_faiss and idx < len(self._vectors):
                self._vectors.pop(idx)

        self._ids.append(item_id)
        self._metadata[item_id] = metadata

        if self._use_faiss:
            self._ensure_faiss_index()
            self._index.add(vector.reshape(1, -1))
        else:
            self._vectors.append(vector)

    def _ensure_faiss_index(self) -> None:
        if not self._use_faiss:
            return
        if self._index is None:
            if self._embedding_dim is None:
                raise ValueError("Cannot create FAISS index without embeddings.")
            self._index = faiss.IndexFlatL2(self._embedding_dim)
            if self._vectors:
                matrix = np.vstack(self._vectors)
                self._index.add(matrix)

    def _get_or_create_embedding(self, item_id: str, text: str) -> np.ndarray:
        cache_file = self.cache_dir / f"{item_id}.npy"
        if cache_file.exists():
            return np.load(cache_file).astype("float32")
        vector = self._compute_embedding(text)
        np.save(cache_file, vector)
        return vector

    def _compute_embedding(self, text: str) -> np.ndarray:
        if self._embedding_fn is not None:
            vector = self._embedding_fn(text)
        else:
            vector = self._default_sentence_embedding(text)
        if vector is None:
            raise RuntimeError("Embedding function returned None.")
        vector = np.asarray(vector, dtype="float32").reshape(-1)
        return vector

    def _default_sentence_embedding(self, text: str) -> np.ndarray:
        if SentenceTransformer is None:
            LOGGER.warning("sentence-transformers not installed; using hashed embeddings.")
            return _hashed_embedding(text)
        if self._sentence_model is None:
            try:
                self._sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception as exc:  # pragma: no cover - download issues
                LOGGER.warning("Failed to load sentence-transformer: %s", exc)
                return _hashed_embedding(text)
        try:
            vector = self._sentence_model.encode(text)
            return np.asarray(vector, dtype="float32")
        except Exception as exc:  # pragma: no cover - runtime issues
            LOGGER.warning("sentence-transformer encode failed: %s", exc)
            return _hashed_embedding(text)


def _distance_to_score(distance: float) -> float:
    return float(1.0 / (1.0 + distance))


def _hashed_embedding(text: str, dim: int = 384) -> np.ndarray:
    seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)
    rng = np.random.default_rng(seed)
    vector = rng.normal(size=dim).astype("float32")
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def _safe_norm(vector: np.ndarray) -> float:
    value = np.linalg.norm(vector)
    return float(value if value > 0 else 1.0)
