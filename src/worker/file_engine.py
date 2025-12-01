from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict


class FileEngine(ABC):
    """
    Manages file I/O, downloads/uploads, and artifact storage.
    """

    @abstractmethod
    def write_text(self, path: str, content: str) -> str:
        raise NotImplementedError

    @abstractmethod
    def read_text(self, path: str) -> str:
        raise NotImplementedError


class LocalFileEngine(FileEngine):
    """
    Basic local filesystem implementation.
    """

    def __init__(self, base_dir: str = "artifacts") -> None:
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)

    def write_text(self, path: str, content: str) -> str:
        full = self._base_dir / path
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(content, encoding="utf-8")
        return str(full)

    def read_text(self, path: str) -> str:
        full = self._base_dir / path
        return full.read_text(encoding="utf-8")