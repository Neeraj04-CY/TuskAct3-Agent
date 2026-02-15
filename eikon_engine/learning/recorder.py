from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from .models import LearningRecord
from .scorer import score_learning

logger = logging.getLogger(__name__)


class LearningRecorder:
    def __init__(self, *, output_dir: Path | str = Path("learning_logs")) -> None:
        self.output_dir = Path(output_dir)

    def record(
        self,
        *,
        mission_result_path: Path | str,
        trace_path: Path | str | None,
        mission_instruction: Optional[str] = None,
        runtime_error: Any | None = None,
        force_outcome_failure: bool = False,
        skill_summary: Optional[Iterable[Dict[str, Any]]] = None,
    ) -> None:
        try:
            mission_path = Path(mission_result_path)
            trace_file = Path(trace_path) if trace_path else None
            self.output_dir.mkdir(parents=True, exist_ok=True)
            payload = self._build_record(
                mission_result_path=mission_path,
                trace_path=trace_file,
                mission_instruction=mission_instruction,
                runtime_error=runtime_error,
                force_outcome_failure=force_outcome_failure,
                skill_summary=skill_summary,
            )
            path = self.output_dir / f"{payload.mission_id}.json"
            path.write_text(json.dumps(payload.to_dict(), indent=2), encoding="utf-8")
        except Exception:  # noqa: BLE001
            logger.warning("learning record write failed", exc_info=True)

    def _build_record(
        self,
        *,
        mission_result_path: Path,
        trace_path: Path | None,
        mission_instruction: Optional[str],
        runtime_error: Any | None,
        force_outcome_failure: bool,
        skill_summary: Optional[Iterable[Dict[str, Any]]],
    ) -> LearningRecord:
        mission_result = self._load_json(mission_result_path)
        trace = self._load_json(trace_path) if trace_path and trace_path.exists() else None
        mission_type = self._infer_mission_type(mission_instruction or trace or mission_result)
        artifacts_exist = self._artifacts_present(mission_result)

        summary = mission_result.get("summary") if isinstance(mission_result, dict) else None
        resume_source = None
        escalation_used = False
        escalation_outcome = None
        if isinstance(summary, dict):
            resume_source = summary.get("resumed_from_checkpoint") or summary.get("resume_checkpoint")
            esc_state = summary.get("escalation_state") or {}
            escalation_used = bool(esc_state.get("used"))
            if escalation_used:
                escalation_outcome = "closed" if esc_state.get("ended_at") else "open"
        resumed = bool(resume_source)

        scoring = score_learning(
            mission_result=mission_result,
            trace=trace,
            skill_summary=skill_summary,
            runtime_error=runtime_error,
            artifacts_exist=artifacts_exist,
            force_outcome_failure=force_outcome_failure,
        )

        outcome_label = scoring.get("outcome", "unknown")
        if resumed:
            if outcome_label == "success":
                outcome_label = "resumed_success"
            elif outcome_label == "halted":
                outcome_label = "resumed_halted"
            else:
                outcome_label = "resumed_failure"
        if escalation_used:
            if outcome_label in {"success", "resumed_success"}:
                outcome_label = "escalation_success"
            elif outcome_label == "halted":
                outcome_label = "escalation_halted"
            else:
                outcome_label = "escalation_failure"

        trace_id = trace.get("id") if isinstance(trace, dict) else None

        record = LearningRecord(
            mission_id=str(mission_result.get("mission_id") or mission_result.get("mission", "unknown")),
            mission_type=mission_type,
            skills_used=scoring["skills"],
            failures=scoring["failures"],
            confidence_score=scoring["confidence"],
            outcome=outcome_label,
            resumed=resumed,
            resume_source=str(resume_source) if resume_source else None,
            escalation_used=escalation_used,
            escalation_outcome=escalation_outcome,
            trace_id=trace_id,
        )
        return record

    def _infer_mission_type(self, source: Optional[Dict[str, Any]] | str) -> str:
        if isinstance(source, str):
            instruction = source
        elif isinstance(source, dict):
            instruction = str(source.get("mission_text") or source.get("instruction") or "")
        else:
            instruction = ""
        instruction = instruction or ""
        lowered = instruction.lower()
        if "login" in lowered:
            return "login"
        if "listing" in lowered:
            return "listing"
        if "extract" in lowered:
            return "extraction"
        if "dashboard" in lowered:
            return "dashboard"
        return "unknown"

    def _load_json(self, path: Path) -> Dict[str, Any]:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("failed to read learning input file", exc_info=True)
            return {}

    def _artifacts_present(self, mission_result: Dict[str, Any]) -> bool:
        artifacts_path = mission_result.get("artifacts_path")
        if not artifacts_path:
            return False
        try:
            artifacts_dir = Path(artifacts_path)
            if artifacts_dir.exists() and artifacts_dir.is_dir():
                # At least mission_result.json should exist; consider directory non-empty as success signal.
                return any(artifacts_dir.iterdir())
        except Exception:  # pragma: no cover - defensive
            logger.debug("artifact presence check failed", exc_info=True)
        return False


__all__ = ["LearningRecorder"]
