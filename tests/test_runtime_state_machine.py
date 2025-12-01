from __future__ import annotations

import pytest

from src.runtime_state import RuntimeState, RuntimeStateMachine, RuntimeTransitionError


def test_runtime_state_machine_valid_transitions() -> None:
    machine = RuntimeStateMachine()
    machine.next(RuntimeState.PLANNING, {"iteration": 0})
    machine.next(RuntimeState.EXECUTING, {"iteration": 0})
    machine.next(RuntimeState.REFLECTING, {"iteration": 0})
    machine.next(RuntimeState.HALT, {"iterations": 1})

    assert machine.current_state is RuntimeState.HALT
    assert len(machine.transitions) == 4


def test_runtime_state_machine_blocks_invalid_transition() -> None:
    machine = RuntimeStateMachine()
    with pytest.raises(RuntimeTransitionError):
        machine.next(RuntimeState.EXECUTING, {})
