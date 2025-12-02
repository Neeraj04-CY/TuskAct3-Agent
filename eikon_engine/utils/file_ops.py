"""Small helpers for interacting with the filesystem."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, content: str) -> None:
    """Write text to disk, creating parent directories when needed."""

    _ensure_parent(path)
    path.write_text(content, encoding="utf-8")


def write_bytes(path: Path, payload: bytes) -> None:
    """Write bytes to disk, creating parent directories when needed."""

    _ensure_parent(path)
    path.write_bytes(payload)


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    """Append a JSONL entry to the target file."""

    _ensure_parent(path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
