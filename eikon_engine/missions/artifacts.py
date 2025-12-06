"""Mission-specific logging helpers for richer ActionResult persistence."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any, Dict, Mapping

from eikon_engine.utils.file_ops import append_jsonl, write_text
from eikon_engine.utils.logging_utils import ArtifactLogger
from eikon_engine.workers.browser_worker import ActionResult


def _serialize_action_result(action_result: ActionResult | Mapping[str, Any] | None) -> Dict[str, Any] | None:
    """Convert ActionResult objects into plain dicts for JSON emission."""

    if action_result is None:
        return None
    if isinstance(action_result, ActionResult):
        return {
            "status": action_result.status,
            "error": action_result.error,
            "details": action_result.details or {},
            "payload": action_result.payload,
        }
    return {
        "status": action_result.get("status"),
        "error": action_result.get("error"),
        "details": action_result.get("details"),
        "payload": action_result.get("payload"),
    }


class MissionArtifactLogger(ArtifactLogger):
    """Artifact logger that records full ActionResult metadata."""

    def log_step(
        self,
        metadata: Dict[str, Any],
        *,
        goal: str | None = None,
        action_result: ActionResult | Mapping[str, Any] | None = None,
    ) -> int:
        self._step_index += 1
        step_index = self._step_index
        entry: Dict[str, Any] = {"idx": step_index, "goal": goal or self.goal_name, "metadata": metadata}
        serialized = _serialize_action_result(action_result)
        if serialized:
            entry["result_status"] = serialized.get("status")
            entry["result_error"] = serialized.get("error")
            entry["result_details"] = serialized.get("details")
            entry["result_payload"] = serialized.get("payload")
        append_jsonl(self.steps_file, entry)
        step_dir = self._ensure_step_dir(step_index)
        write_text(step_dir / "step.json", json.dumps(entry, ensure_ascii=False, indent=2))
        return step_index

    def log_trace(
        self,
        *,
        goal: str | None,
        step_index: int,
        action: str,
        url: str | None,
        completion: Dict[str, Any] | None,
        action_result: ActionResult | Mapping[str, Any] | None = None,
    ) -> None:
        payload: Dict[str, Any] = {
            "timestamp": self._now_iso(),
            "goal": goal or self.goal_name,
            "step_index": step_index,
            "action": action,
            "url": url,
            "completion": completion,
        }
        serialized = _serialize_action_result(action_result)
        if serialized:
            payload["result"] = serialized
        append_jsonl(self.trace_file, payload)

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(UTC).isoformat()


__all__ = ["MissionArtifactLogger", "_serialize_action_result"]
