from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, List


class RuntimeState(Enum):
    PLANNING = auto()
    EXECUTING = auto()
    REFLECTING = auto()
    RETRYING = auto()
    HALT = auto()


class RuntimeTransitionError(RuntimeError):
    pass


@dataclass(slots=True)
class StateTransition:
    previous: RuntimeState
    next_state: RuntimeState
    context: Dict[str, Any]


class RuntimeStateMachine:
    """Finite-state machine guarding orchestrator loops."""

    _ALLOWED = {
        RuntimeState.HALT: {RuntimeState.PLANNING},
        RuntimeState.PLANNING: {RuntimeState.EXECUTING, RuntimeState.HALT},
        RuntimeState.EXECUTING: {RuntimeState.REFLECTING, RuntimeState.RETRYING, RuntimeState.HALT},
        RuntimeState.REFLECTING: {RuntimeState.RETRYING, RuntimeState.HALT, RuntimeState.PLANNING},
        RuntimeState.RETRYING: {RuntimeState.PLANNING, RuntimeState.HALT},
    }

    def __init__(self) -> None:
        self.current_state = RuntimeState.HALT
        self.transitions: List[StateTransition] = []

    def next(self, target: RuntimeState, context: Dict[str, Any] | None = None) -> RuntimeState:
        context = context or {}
        allowed_targets = self._ALLOWED.get(self.current_state, set())
        if target not in allowed_targets:
            raise RuntimeTransitionError(
                f"Illegal transition from {self.current_state.name} to {target.name}"
            )
        transition = StateTransition(self.current_state, target, context)
        self.transitions.append(transition)
        self.current_state = target
        return self.current_state
