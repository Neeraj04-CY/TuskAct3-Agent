from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.memory.memory_manager import MemoryManager


class Telemetry:
    """Captures trace events and optionally persists them via MemoryManager."""

    def __init__(
        self,
        memory_manager: Optional["MemoryManager"] = None,
        sample_rate: float = 1.0,
        rng: Optional[random.Random] = None,
    ) -> None:
        self.memory_manager = memory_manager
        self.sample_rate = max(0.0, min(sample_rate, 1.0))
        self._rng = rng or random.Random()
        self.events: List[Dict[str, Any]] = []

    async def trace_event(self, event_type: str, payload: Dict[str, Any] | None = None) -> None:
        payload = payload or {}
        event = {
            "type": event_type,
            "payload": payload,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.events.append(event)

        if not self.memory_manager:
            return
        if self._rng.random() > self.sample_rate:
            return

        context = {"event_type": event_type, **payload}
        summary = json.dumps(event, default=str)
        await self.memory_manager.add_memory(
            event_type="trace_event",
            title=f"Trace: {event_type}",
            text=summary,
            context=context,
        )
