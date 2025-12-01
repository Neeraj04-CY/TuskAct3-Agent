from __future__ import annotations

from typing import Any, Dict, Optional


def extract_completion_metadata(results: Dict[str, Any] | None) -> Optional[Dict[str, Any]]:
    """Return the first completion payload with complete=True from worker outputs."""

    if not results:
        return None
    for node_id, payload in results.items():
        if not isinstance(payload, dict):
            continue
        completion = payload.get("completion")
        if isinstance(completion, dict) and completion.get("complete"):
            enriched = {**completion}
            enriched.setdefault("node_id", node_id)
            return enriched
    return None
