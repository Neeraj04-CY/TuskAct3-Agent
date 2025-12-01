from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict


class RequestParser(ABC):
    """
    Parses raw natural language input into a structured intermediate representation.

    Design:
    - Small, testable abstraction.
    - Does not know about tools or memory.
    """

    @abstractmethod
    def parse(self, user_input: str) -> Dict[str, str]:
        """
        Parse the raw user input into a normalized dict.

        Returns:
            Dict with at least:
            - "task": canonical task description
            - optional extra metadata in other keys
        """
        raise NotImplementedError


class SimpleRequestParser(RequestParser):
    """
    Basic v1 parser that treats the entire input as the task.
    Later versions can plug in LLMs or rule-based NLP.
    """

    def parse(self, user_input: str) -> Dict[str, str]:
        normalized = user_input.strip()
        return {
            "task": normalized,
            "raw_input": user_input
        }