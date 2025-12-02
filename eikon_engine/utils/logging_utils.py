"""Artifact logging utilities for browser runs."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Optional

from .file_ops import append_jsonl, write_bytes, write_text


class ArtifactLogger:
    """Creates timestamped run folders and persists artifacts automatically."""

    def __init__(
        self,
        *,
        root: Path | None = None,
        prefix: str | None = None,
        base_dir: Path | str | None = None,
        goal_name: str | None = None,
    ) -> None:
        if base_dir is not None:
            self.base_dir = Path(base_dir)
            self.base_dir.mkdir(parents=True, exist_ok=True)
        else:
            run_root = root or Path("runs")
            timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
            label = prefix or "run"
            self.base_dir = run_root / f"{label}_{timestamp}"
            self.base_dir.mkdir(parents=True, exist_ok=True)
        self.goal_name = goal_name
        self.steps_file = self.base_dir / "steps.jsonl"
        self.trace_file = self.base_dir / "trace.jsonl"
        self.summary_file = self.base_dir / "run_summary.json"
        self._step_index = 0

    def create_child(self, name: str, *, goal_name: str | None = None) -> "ArtifactLogger":
        """Return a logger rooted at a subdirectory for goal-specific runs."""

        child_dir = self.base_dir / name
        return ArtifactLogger(base_dir=child_dir, goal_name=goal_name or self.goal_name)

    def _ensure_step_dir(self, step_index: int) -> Path:
        step_dir = self.base_dir / f"step_{step_index:03d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        return step_dir

    def log_step(self, metadata: Dict[str, Any], *, goal: str | None = None) -> int:
        """Persist a strategy step to steps.jsonl and write a step.json file."""

        self._step_index += 1
        step_index = self._step_index
        entry = {"idx": step_index, "goal": goal or self.goal_name, "metadata": metadata}
        append_jsonl(self.steps_file, entry)
        step_dir = self._ensure_step_dir(step_index)
        write_text(step_dir / "step.json", json.dumps(entry, indent=2))
        return step_index

    def save_dom(self, html: str, *, step_index: int | None = None, name: str | None = None) -> Path:
        """Save a DOM snapshot to disk and return the path."""

        idx = step_index or max(self._step_index, 1)
        step_dir = self._ensure_step_dir(idx)
        file_name = name or "dom.html"
        path = step_dir / file_name
        write_text(path, html)
        return path

    def save_layout_graph(self, layout: str, *, step_index: int | None = None) -> Path:
        """Persist layout graph details alongside each step."""

        idx = step_index or max(self._step_index, 1)
        step_dir = self._ensure_step_dir(idx)
        path = step_dir / "layout_graph.json"
        write_text(path, json.dumps({"layout": layout}, ensure_ascii=False, indent=2))
        return path

    def save_screenshot(self, payload: bytes, *, step_index: int | None = None, name: str | None = None) -> Path:
        """Persist screenshot bytes to disk within the step directory."""

        idx = step_index or max(self._step_index, 1)
        step_dir = self._ensure_step_dir(idx)
        file_name = name or "screenshot.png"
        path = step_dir / file_name
        write_bytes(path, payload)
        return path

    def log_trace(
        self,
        *,
        goal: str | None,
        step_index: int,
        action: str,
        url: Optional[str],
        completion: Optional[Dict[str, Any]],
    ) -> None:
        """Append an event to trace.jsonl capturing goal progress."""

        payload = {
            "timestamp": datetime.now(UTC).isoformat(),
            "goal": goal or self.goal_name,
            "step_index": step_index,
            "action": action,
            "url": url,
            "completion": completion,
        }
        append_jsonl(self.trace_file, payload)

    def to_dict(self) -> Dict[str, str]:
        """Expose directory paths for downstream reporting."""

        return {
            "base_dir": str(self.base_dir),
            "steps_file": str(self.steps_file),
            "trace_file": str(self.trace_file),
        }

    def write_summary(self, payload: Dict[str, Any]) -> Path:
        """Persist a run-level summary file in the current run directory."""

        write_text(self.summary_file, json.dumps(payload, ensure_ascii=False, indent=2))
        return self.summary_file
