"""Goal management utilities for multi-step browser instructions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Protocol

from eikon_engine.core.completion import build_completion, CompletionPayload


@dataclass
class Goal:
    """Represents a granular unit of work derived from a natural instruction."""

    name: str
    description: str | None = None
    status: str = "pending"
    metadata: Dict[str, Any] = field(default_factory=dict)
    result: Dict[str, Any] | None = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "goal": self.name,
            "description": self.description,
            "status": self.status,
            "metadata": self.metadata,
            "result": self.result,
        }


class GoalSplitter(Protocol):
    """Interface for components that convert instructions to atomic goals."""

    def split(self, instruction: str) -> Iterable[Goal]:  # pragma: no cover - interface only
        ...


class RuleBasedSplitter:
    """Heuristic splitter that understands a few common browser flows."""

    def split(self, instruction: str) -> Iterable[Goal]:
        tokens = instruction.lower()
        goals: List[Goal] = []

        if "login" in tokens or "log in" in tokens:
            goals.append(Goal(name="open_login_page", description="Navigate to login page"))
            goals.append(Goal(name="perform_login", description="Submit known credentials"))
        elif "navigate" in tokens or "open" in tokens:
            goals.append(Goal(name="open_target", description="Open referenced destination"))

        if "secure" in tokens and "title" in tokens:
            goals.append(Goal(name="extract_secure_title", description="Capture secure area title"))
        elif "extract" in tokens or "capture" in tokens:
            goals.append(Goal(name="extract_content", description="Extract requested content"))

        if "logout" in tokens or "sign out" in tokens:
            goals.append(Goal(name="logout", description="Perform logout"))

        if not goals:
            goals.append(Goal(name="execute_instruction", description=instruction.strip() or "generic goal"))

        return goals


class GoalManager:
    """Tracks goal progress and exposes iteration helpers."""

    def __init__(
        self,
        instruction: str,
        splitter: GoalSplitter | None = None,
        goals: Iterable[Goal] | None = None,
    ) -> None:
        self.instruction = instruction
        self._splitter = splitter or RuleBasedSplitter()
        seed_goals = list(goals) if goals is not None else list(self._splitter.split(instruction))
        self._goals: List[Goal] = seed_goals
        self._current_idx: Optional[int] = None
        self._completion: CompletionPayload = build_completion(complete=False, reason="goals pending")

    @property
    def goals(self) -> List[Goal]:
        return self._goals

    def structured_goals(self) -> List[Dict[str, Any]]:
        return [goal.as_dict() for goal in self._goals]

    def next_goal(self) -> Optional[Goal]:
        for idx, goal in enumerate(self._goals):
            if goal.status == "pending":
                self._current_idx = idx
                goal.status = "in_progress"
                return goal
        self._completion = build_completion(complete=True, reason="all goals completed")
        return None

    def update(self, goal_result: Dict[str, Any]) -> None:
        goal_name = goal_result.get("goal")
        completion = goal_result.get("completion", {})
        for goal in self._goals:
            if goal.name == goal_name:
                goal.result = goal_result
                if completion.get("complete"):
                    goal.status = "complete"
                elif goal_result.get("error"):
                    goal.status = "error"
                else:
                    goal.status = "in_progress"
                break
        if all(goal.status == "complete" for goal in self._goals):
            self._completion = build_completion(complete=True, reason="all goals complete")

    def completion_state(self) -> CompletionPayload:
        return self._completion

    @classmethod
    def parse(cls, instruction: str) -> "GoalManager":
        return cls(instruction=instruction)

    @classmethod
    def from_goals(cls, *, instruction: str, goals: Iterable[Goal]) -> "GoalManager":
        return cls(instruction=instruction, goals=list(goals))

    def progress_report(self) -> Dict[str, Any]:
        return {
            "instruction": self.instruction,
            "goals": self.structured_goals(),
            "completion": self.completion_state(),
        }
