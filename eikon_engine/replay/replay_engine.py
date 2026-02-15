from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

from eikon_engine.missions.artifacts import MissionArtifactLogger
from eikon_engine.trace.models import ActionTrace, ExecutionTrace, SkillUsage, SubgoalTrace
from eikon_engine.trace.reader import read_trace
from eikon_engine.workers.browser_worker import BrowserWorker
from eikon_engine.core.types import BrowserAction


@dataclass
class ReplaySummary:
    """Structured replay result reported to callers and summary files."""

    mission_id: str
    trace_id: str
    status: str = "pending"
    subgoals_replayed: int = 0
    actions_replayed: int = 0
    skills_replayed: int = 0
    output_dir: Optional[Path] = None
    divergence: Optional[Dict[str, str]] = None
    summary_path: Optional[Path] = None
    skill_details: List[Dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "mission_id": self.mission_id,
            "trace_id": self.trace_id,
            "status": self.status,
            "subgoals_replayed": self.subgoals_replayed,
            "actions_replayed": self.actions_replayed,
            "skills_replayed": self.skills_replayed,
            "output_dir": str(self.output_dir) if self.output_dir else None,
        }
        if self.divergence:
            payload["divergence"] = dict(self.divergence)
        if self.summary_path:
            payload["summary_path"] = str(self.summary_path)
        if self.skill_details:
            payload["skill_details"] = list(self.skill_details)
        return payload


class ReplayError(Exception):
    """Base class for deterministic replay failures."""

    def __init__(self, message: str, *, subgoal_id: str | None = None, action_sequence: int | None = None) -> None:
        super().__init__(message)
        self.subgoal_id = subgoal_id
        self.action_sequence = action_sequence

    def as_dict(self) -> Dict[str, str]:
        payload = {"message": str(self)}
        if self.subgoal_id:
            payload["subgoal_id"] = self.subgoal_id
        if self.action_sequence is not None:
            payload["action_sequence"] = str(self.action_sequence)
        return payload


class ReplayDivergenceError(ReplayError):
    """Raised when observed behaviour deviates from the recorded trace."""

    def __init__(
        self,
        message: str,
        *,
        subgoal_id: str | None = None,
        action_sequence: int | None = None,
        expected: str | None = None,
        observed: str | None = None,
    ) -> None:
        super().__init__(message, subgoal_id=subgoal_id, action_sequence=action_sequence)
        self.expected = expected
        self.observed = observed

    def as_dict(self) -> Dict[str, str]:  # type: ignore[override]
        payload = super().as_dict()
        if self.expected is not None:
            payload["expected"] = self.expected
        if self.observed is not None:
            payload["observed"] = self.observed
        payload["type"] = "replay_divergence"
        return payload


class ReplaySkillError(ReplayError):
    """Raised when a recorded skill outcome cannot be reproduced."""

    def __init__(
        self,
        message: str,
        *,
        subgoal_id: str | None = None,
        expected: Optional[str] = None,
        observed: Optional[str] = None,
    ) -> None:
        super().__init__(message, subgoal_id=subgoal_id)
        self.expected = expected
        self.observed = observed

    def as_dict(self) -> Dict[str, str]:  # type: ignore[override]
        payload = super().as_dict()
        if self.expected is not None:
            payload["expected"] = self.expected
        if self.observed is not None:
            payload["observed"] = self.observed
        payload["type"] = "skill_mismatch"
        return payload


@dataclass
class ReplayConfig:
    """Runtime configuration for the replay engine."""

    headless: bool = True
    output_root: Path = Path("replay_artifacts")
    settings: Dict[str, object] = field(default_factory=dict)
    worker_factory: Optional[Callable[[MissionArtifactLogger, bool], BrowserWorker]] = None


@dataclass
class ReplayTraceContext:
    """Resolved helper paths used during deterministic replay."""

    mission_dir: Optional[Path]
    subgoal_artifacts: Dict[str, Path]
    skill_usage: Dict[str, SkillUsage]


class ReplayEngine:
    """Deterministic re-execution of recorded mission traces."""

    def __init__(self, config: ReplayConfig | None = None) -> None:
        self.config = config or ReplayConfig()

    async def replay_trace(
        self,
        trace: ExecutionTrace,
        *,
        headless: bool | None = None,
        output_dir: Path | str | None = None,
    ) -> ReplaySummary:
        headless_mode = self.config.headless if headless is None else headless
        replay_root = Path(output_dir) if output_dir else self.config.output_root / trace.mission_id
        replay_root.mkdir(parents=True, exist_ok=True)
        summary = ReplaySummary(mission_id=trace.mission_id, trace_id=trace.id, output_dir=replay_root)
        divergence_error: ReplayError | None = None
        context = self._build_trace_context(trace)

        for index, subgoal in enumerate(trace.subgoal_traces, start=1):
            actions = self._build_actions(subgoal)
            needs_skill = bool(subgoal.skill_used)
            if not actions and not needs_skill:
                continue
            subgoal_dir = replay_root / f"subgoal_{index:02d}"
            logger = MissionArtifactLogger(base_dir=subgoal_dir, goal_name=subgoal.description)
            worker = self._build_worker(logger=logger, headless=headless_mode)
            print(f"[REPLAY] Subgoal {index:02d}: {subgoal.description}")
            worker_result: Dict[str, Any] = {"steps": []}
            try:
                if actions:
                    worker_result = await worker.execute({
                        "action": actions,
                        "goal": subgoal.description,
                        "description": subgoal.description,
                    })
                    summary.actions_replayed += len(actions)
                    self._validate_subgoal(subgoal, worker_result)
                summary.subgoals_replayed += 1
                if needs_skill and subgoal.skill_used:
                    recorded_skill = context.skill_usage.get(subgoal.subgoal_id)
                    artifact_hint = context.subgoal_artifacts.get(subgoal.subgoal_id)
                    skill_detail = await self._replay_skill(
                        skill_name=subgoal.skill_used,
                        worker=worker,
                        subgoal=subgoal,
                        subgoal_dir=subgoal_dir,
                        artifact_hint=artifact_hint,
                        recorded_usage=recorded_skill,
                        worker_result=worker_result,
                    )
                    summary.skills_replayed += 1
                    summary.skill_details.append(skill_detail)
            except ReplayError as err:  # deterministically raised divergences
                divergence_error = err
                break
            except Exception as exc:  # unexpected failures also treated as divergence
                divergence_error = ReplayError(str(exc), subgoal_id=subgoal.subgoal_id)
                break
            finally:
                await worker.shutdown()

        if divergence_error:
            summary.status = "failed"
            summary.divergence = divergence_error.as_dict()
        else:
            summary.status = "success"
        summary.summary_path = self._write_summary(replay_root, summary)
        return summary

    async def replay(
        self,
        trace_path: Path | str,
        *,
        headless: bool | None = None,
        output_dir: Path | str | None = None,
    ) -> ReplaySummary:
        trace = read_trace(trace_path)
        return await self.replay_trace(trace, headless=headless, output_dir=output_dir)

    def _build_worker(self, *, logger: MissionArtifactLogger, headless: bool) -> BrowserWorker:
        factory = self.config.worker_factory
        if factory:
            worker = factory(logger, headless)
        else:
            settings = dict(self.config.settings)
            worker = BrowserWorker(settings=settings, logger=logger, show_browser=not headless)
        if hasattr(worker, "demo_mode"):
            worker.demo_mode = False
        if hasattr(worker, "skip_retries"):
            worker.skip_retries = True
        return worker

    def _write_summary(self, replay_root: Path, summary: ReplaySummary) -> Path:
        lines = [
            f"Mission ID: {summary.mission_id}",
            f"Trace ID: {summary.trace_id}",
            f"Status: {summary.status}",
            f"Subgoals replayed: {summary.subgoals_replayed}",
            f"Actions replayed: {summary.actions_replayed}",
            f"Skills replayed: {summary.skills_replayed}",
        ]
        if summary.divergence:
            lines.append("Divergence detected: yes")
            for key, value in summary.divergence.items():
                lines.append(f"  {key}: {value}")
        else:
            lines.append("Divergence detected: no")
        if summary.skill_details:
            lines.append("Skill results:")
            for detail in summary.skill_details:
                skill = detail.get("skill")
                status = detail.get("status")
                location = detail.get("artifact")
                lines.append(f"  - {skill}: {status} ({location})")
        path = replay_root / "replay_summary.txt"
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def _build_trace_context(self, trace: ExecutionTrace) -> ReplayTraceContext:
        mission_dir: Optional[Path] = None
        artifact_map: Dict[str, Path] = {}
        for artifact in trace.artifacts:
            path_value = Path(artifact.path)
            if artifact.name == "mission_dir":
                mission_dir = path_value
            if artifact.name.startswith("subgoal_"):
                parts = artifact.name.split("_")
                if len(parts) < 2:
                    continue
                try:
                    index = int(parts[1]) - 1
                except ValueError:
                    continue
                if 0 <= index < len(trace.subgoal_traces):
                    artifact_map[trace.subgoal_traces[index].subgoal_id] = path_value
        skill_usage: Dict[str, SkillUsage] = {}
        for usage in trace.skills_used:
            if usage.subgoal_id:
                skill_usage[usage.subgoal_id] = usage
        return ReplayTraceContext(mission_dir=mission_dir, subgoal_artifacts=artifact_map, skill_usage=skill_usage)

    def _load_dom_snapshot(self, artifact_dir: Optional[Path]) -> Optional[str]:
        if not artifact_dir:
            return None
        try:
            candidates = sorted(
                [path for path in artifact_dir.glob("step_*") if path.is_dir()],
                reverse=True,
            )
        except FileNotFoundError:
            return None
        for step_dir in candidates:
            dom_path = step_dir / "dom.html"
            if dom_path.exists():
                return dom_path.read_text(encoding="utf-8")
        return None

    def _resolve_recorded_url(self, subgoal: SubgoalTrace) -> Optional[str]:
        for record in sorted(subgoal.actions_taken, key=lambda entry: entry.sequence, reverse=True):
            target = record.target or record.selector
            if isinstance(target, str) and target.lower().startswith("http"):
                return target
        return None

    async def _replay_skill(
        self,
        *,
        skill_name: str,
        worker: BrowserWorker,
        subgoal: SubgoalTrace,
        subgoal_dir: Path,
        artifact_hint: Optional[Path],
        recorded_usage: SkillUsage | None,
        worker_result: Dict[str, Any],
    ) -> Dict[str, str]:
        runner = getattr(worker, "run_skill", None)
        if not callable(runner):
            raise ReplaySkillError(
                "Browser worker does not expose run_skill",
                subgoal_id=subgoal.subgoal_id,
            )
        artifact_path = subgoal_dir / f"{skill_name}.json"
        dom_snapshot = worker_result.get("dom_snapshot") if isinstance(worker_result, dict) else None
        if not dom_snapshot:
            dom_snapshot = self._load_dom_snapshot(artifact_hint)
        context: Dict[str, Any] = {"artifact_path": str(artifact_path)}
        if dom_snapshot:
            context["html"] = dom_snapshot
        page_url = self._resolve_recorded_url(subgoal)
        if page_url:
            context["page_url"] = page_url
        replayed = await runner(skill_name, context)
        if not isinstance(replayed, dict):
            raise ReplaySkillError(
                "Skill replay returned non-dict payload",
                subgoal_id=subgoal.subgoal_id,
            )
        self._validate_skill_result(
            skill_name=skill_name,
            subgoal=subgoal,
            recorded_usage=recorded_usage,
            replayed=replayed,
        )
        status = str(replayed.get("status") or replayed.get("result", {}).get("status") or "unknown")
        return {
            "skill": skill_name,
            "status": status,
            "artifact": str(artifact_path),
        }

    def _validate_skill_result(
        self,
        *,
        skill_name: str,
        subgoal: SubgoalTrace,
        recorded_usage: SkillUsage | None,
        replayed: Dict[str, Any],
    ) -> None:
        if not recorded_usage:
            return
        recorded_status = (recorded_usage.status or "").lower()
        observed_status = str(replayed.get("status") or replayed.get("result", {}).get("status") or "").lower()
        if recorded_status and observed_status and recorded_status != observed_status:
            raise ReplaySkillError(
                f"Skill {skill_name} status mismatch",
                subgoal_id=subgoal.subgoal_id,
                expected=recorded_status,
                observed=observed_status,
            )
        recorded_payload = recorded_usage.metadata.get("result") if isinstance(recorded_usage.metadata, dict) else None
        if isinstance(recorded_payload, dict):
            if not self._deep_compare(recorded_payload, replayed):
                raise ReplaySkillError(
                    f"Skill {skill_name} output mismatch",
                    subgoal_id=subgoal.subgoal_id,
                )

    @staticmethod
    def _deep_compare(left: Any, right: Any) -> bool:
        return ReplayEngine._normalize_structure(left) == ReplayEngine._normalize_structure(right)

    @staticmethod
    def _normalize_structure(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: ReplayEngine._normalize_structure(val) for key, val in sorted(value.items(), key=lambda item: item[0])}
        if isinstance(value, list):
            return [ReplayEngine._normalize_structure(item) for item in value]
        return value

    def _build_actions(self, subgoal: SubgoalTrace) -> List[BrowserAction]:
        ordered_records = sorted(subgoal.actions_taken, key=lambda entry: entry.sequence)
        actions: List[BrowserAction] = []
        for record in ordered_records:
            action = self._translate_action(record)
            if action:
                actions.append(action)
        return actions

    def _translate_action(self, record: ActionTrace) -> Optional[BrowserAction]:
        action_type = (record.action_type or "").lower()
        if not action_type:
            return None
        payload: BrowserAction = {"action": action_type}
        metadata = record.metadata or {}
        if not isinstance(metadata, dict):
            metadata = {}
        details = metadata.get("details") if isinstance(metadata, dict) else None
        details = details if isinstance(details, dict) else {}
        if action_type == "navigate":
            payload["url"] = record.target or details.get("url") or details.get("target")
        else:
            selector = record.selector or details.get("selector") or self._maybe_selector(record.target)
            if selector:
                payload["selector"] = selector
        input_payload = record.input_data
        if (not input_payload or input_payload == "***") and details.get("value"):
            input_payload = details.get("value")
        if input_payload is not None:
            payload["text"] = str(input_payload)
        timeout = details.get("timeout_ms")
        if timeout is not None:
            try:
                payload["timeout"] = int(timeout)
            except (TypeError, ValueError):
                pass
        state = details.get("state")
        if state:
            payload["state"] = str(state)
        if action_type == "screenshot":
            name = details.get("name")
            payload_meta = metadata.get("payload") if isinstance(metadata, dict) else None
            if not name and isinstance(payload_meta, dict):
                name = payload_meta.get("name")
            if name:
                payload["name"] = str(name)
        return payload

    @staticmethod
    def _maybe_selector(value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        if value.startswith("//") or value.startswith("css="):
            return value
        if value.startswith("#") or value.startswith("."):
            return value
        if value.startswith("[") and value.endswith("]"):
            return value
        if value.lower().startswith("xpath="):
            return value
        return None

    def _validate_subgoal(self, subgoal: SubgoalTrace, worker_result: Dict[str, object]) -> None:
        expected_status = (subgoal.status or "unknown").lower()
        error = worker_result.get("error")
        if expected_status == "complete" and error:
            raise ReplayDivergenceError(
                "Recorded execution succeeded but replay failed",
                subgoal_id=subgoal.subgoal_id,
                expected=expected_status,
                observed="error",
            )
        if expected_status != "complete" and not error:
            raise ReplayDivergenceError(
                "Recorded execution failed/aborted but replay succeeded",
                subgoal_id=subgoal.subgoal_id,
                expected=expected_status,
                observed="success",
            )
        recorded_steps = len(subgoal.actions_taken)
        replay_steps = len(worker_result.get("steps") or [])
        if replay_steps != recorded_steps:
            raise ReplayDivergenceError(
                "Replay produced different action count",
                subgoal_id=subgoal.subgoal_id,
                expected=str(recorded_steps),
                observed=str(replay_steps),
            )
        ordered_records = sorted(subgoal.actions_taken, key=lambda entry: entry.sequence)
        replay_entries = worker_result.get("steps") or []
        for record, replay in zip(ordered_records, replay_entries):
            recorded_action = (record.action_type or "").lower()
            replay_action = str((replay or {}).get("action") or "").lower()
            if recorded_action != replay_action:
                raise ReplayDivergenceError(
                    "Replay action mismatch",
                    subgoal_id=subgoal.subgoal_id,
                    action_sequence=record.sequence,
                    expected=recorded_action,
                    observed=replay_action,
                )
            recorded_status = (record.status or "").lower()
            replay_status = str((replay or {}).get("status") or "").lower()
            if recorded_status and replay_status and recorded_status != replay_status:
                raise ReplayDivergenceError(
                    "Replay action status diverged",
                    subgoal_id=subgoal.subgoal_id,
                    action_sequence=record.sequence,
                    expected=recorded_status,
                    observed=replay_status,
                )


__all__ = [
    "ReplayEngine",
    "ReplaySummary",
    "ReplayConfig",
    "ReplayError",
    "ReplayDivergenceError",
    "ReplaySkillError",
]
