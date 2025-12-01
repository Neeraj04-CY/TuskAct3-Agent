from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class DocParser(ABC):
    """
    Parses documentation (OpenAPI, markdown, HTML) into structured tool descriptions.
    """

    @abstractmethod
    def parse(self, raw_content: str) -> Dict[str, Any]:
        raise NotImplementedError


class NoopDocParser(DocParser):
    """
    v1 stub parser.
    """

    def parse(self, raw_content: str) -> Dict[str, Any]:
        return {"raw": raw_content}