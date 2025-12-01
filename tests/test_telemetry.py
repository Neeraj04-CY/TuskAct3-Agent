from __future__ import annotations

import pytest

from src.telemetry import Telemetry
from tests.phase4_fakes import StubMemoryManager


@pytest.mark.asyncio
async def test_telemetry_records_events_and_saves_memory() -> None:
    memory = StubMemoryManager()
    telemetry = Telemetry(memory_manager=memory, sample_rate=1.0)

    await telemetry.trace_event("plan_created", {"goal": "test"})

    assert telemetry.events[0]["type"] == "plan_created"
    assert memory.saved
    assert memory.saved[0]["context"]["goal"] == "test"


@pytest.mark.asyncio
async def test_telemetry_sampling_can_skip_memory_writes() -> None:
    memory = StubMemoryManager()
    telemetry = Telemetry(memory_manager=memory, sample_rate=0.0)

    await telemetry.trace_event("plan_created", {"goal": "skip"})

    assert telemetry.events
    assert memory.saved == []
