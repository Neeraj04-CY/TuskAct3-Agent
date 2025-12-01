from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from src.common_types import RiskItem


class RiskAnalyzer(ABC):
    """
    Estimates potential risks for the given task description.
    """

    @abstractmethod
    def analyze(self, task_description: str) -> List[RiskItem]:
        raise NotImplementedError


class SimpleRiskAnalyzer(RiskAnalyzer):
    """
    v1 risk analyzer using simple keyword rules.
    """

    def analyze(self, task_description: str) -> List[RiskItem]:
        lowered = task_description.lower()
        risks: List[RiskItem] = []

        if "delete" in lowered or "remove" in lowered or "drop" in lowered:
            risks.append(
                RiskItem(
                    description="Potential destructive operation (delete/remove/drop).",
                    likelihood="medium",
                    impact="high",
                    mitigation="Confirm with user and perform dry-run when possible."
                )
            )

        if "shell" in lowered or "terminal" in lowered or "command" in lowered:
            risks.append(
                RiskItem(
                    description="Execution of shell commands may be unsafe.",
                    likelihood="medium",
                    impact="high",
                    mitigation="Restrict shell execution or sandbox commands."
                )
            )

        return risks