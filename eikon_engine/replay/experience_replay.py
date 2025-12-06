from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from eikon_engine.replay.curriculum_builder import CurriculumBatch, CurriculumBuilder
from eikon_engine.strategist.strategist_v2 import StrategistV2
from eikon_engine.strategist_v2.navigator_reward_model import compute_reward


@dataclass
class _DryPlanner:
    async def create_plan(self, goal: str, *, last_result: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:  # pragma: no cover - trivial
        return {"goal": goal, "tasks": []}


class ExperienceReplayEngine:
    """Offline experience replay to strengthen StrategistV2 without a browser."""

    def __init__(
        self,
        artifact_root: str | Path = "artifacts",
        *,
        strategist_factory: Optional[Callable[[], StrategistV2]] = None,
    ) -> None:
        self.artifact_root = Path(artifact_root)
        self._strategist_factory = strategist_factory or (lambda: StrategistV2(planner=_DryPlanner()))
        self._strategist = self._strategist_factory()

    @property
    def strategist(self) -> StrategistV2:
        return self._strategist

    def load_runs(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        runs: List[Dict[str, Any]] = []
        for run_dir in self._discover_run_dirs():
            payload = self._load_run_payload(run_dir)
            if payload:
                runs.append(payload)
            if limit and len(runs) >= limit:
                break
        return runs

    def build_curriculum(self, limit: Optional[int] = None) -> List[CurriculumBatch]:
        runs = self.load_runs(limit=limit)
        builder = CurriculumBuilder(runs)
        return builder.get_curriculum()

    def replay_curriculum(
        self,
        curriculum: Sequence[CurriculumBatch],
        *,
        output_dir: Optional[Path] = None,
    ) -> Dict[str, Any]:
        output_path = output_dir or (self.artifact_root / "replay")
        output_path.mkdir(parents=True, exist_ok=True)
        batches_summary: List[Dict[str, Any]] = []
        total_states = 0
        for batch in curriculum:
            batch_result = self._run_batch(batch)
            if not batch_result["states"]:
                continue
            total_states += batch_result["states"]
            batches_summary.append(batch_result)
            batch_file = output_path / f"batch_{batch_result['tag']}.json"
            batch_file.write_text(json.dumps(batch_result, indent=2), encoding="utf-8")
        summary = {
            "batches_processed": len(batches_summary),
            "states_processed": total_states,
            "batches": batches_summary,
        }
        (output_path / "replay_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return {"summary": summary, "output_dir": output_path}

    def save_memory_hints(self, output_dir: Optional[Path] = None) -> Path:
        output_path = output_dir or (self.artifact_root / "replay")
        output_path.mkdir(parents=True, exist_ok=True)
        payload = {
            "summary": self.strategist.agent_memory.summarize_experience(),
            "entries": self.strategist.agent_memory.export(),
        }
        target = output_path / "memory_hints.json"
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return target

    def _run_batch(self, batch: CurriculumBatch) -> Dict[str, Any]:
        states: List[Dict[str, Any]] = []
        for run in batch.get("runs", []):
            states.extend(self._extract_states(run))
        effects = self.strategist.learn_from_past({"tag": batch.get("tag"), "states": states, "reason": batch.get("reason")})
        return {
            "tag": batch.get("tag"),
            "reason": batch.get("reason"),
            "states": len(states),
            "effects": effects,
        }

    def _discover_run_dirs(self) -> Iterable[Path]:
        autonomy_root = self.artifact_root / "autonomy"
        if not autonomy_root.exists():
            return []
        directories = sorted(path for path in autonomy_root.iterdir() if path.is_dir())
        return [path for path in directories if (path / "result.json").exists()]

    def _load_run_payload(self, run_dir: Path) -> Optional[Dict[str, Any]]:
        summary = self._read_json(run_dir / "summary.json")
        result = self._read_json(run_dir / "result.json")
        stability = self._read_json(run_dir / "stability_report.json")
        if not summary and not result:
            return None
        return {"path": str(run_dir), "summary": summary, "result": result, "stability": stability, "goal": summary.get("goal") or result.get("goal")}

    def _extract_states(self, run_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        result = run_payload.get("result", {}) or {}
        steps = result.get("steps", []) if isinstance(result, dict) else []
        run_ctx = result.get("run_context", {}) if isinstance(result, dict) else {}
        states: List[Dict[str, Any]] = []
        prev_dom = ""
        for entry in steps:
            step_meta = entry.get("step", {})
            step_result = entry.get("result", {})
            step_id = step_meta.get("step_id") or step_meta.get("id")
            before_dom = prev_dom
            dom_after = step_result.get("dom_snapshot") or before_dom
            reward_payload = compute_reward(
                before_dom,
                dom_after or "",
                run_ctx.get("current_url"),
                step_meta.get("action"),
                run_ctx.get("active_subgoal"),
            )
            prev_dom = dom_after or before_dom
            reward_trace = self._trace_for_step(run_ctx, step_id, reward_payload)
            failure_reason = step_result.get("error") or self._low_confidence_reason(reward_trace)
            if not failure_reason and reward_payload.get("reward", 0.0) >= 0.25:
                continue
            state_record = {
                "goal": run_payload.get("goal"),
                "step_id": step_id,
                "fingerprint": self._fingerprint_for_step(run_ctx, step_id),
                "reward_trace": reward_trace,
                "reward_info": reward_payload,
                "repair_events": run_ctx.get("repair_events") or [],
                "planner_events": run_ctx.get("planner_events") or [],
                "plan_evolution": run_ctx.get("plan_evolution") or [],
                "behavior_summary": run_ctx.get("behavior_summary"),
                "stability": run_ctx.get("stability_summary") or run_payload.get("stability"),
                "skill_events": run_ctx.get("skills") or [],
                "skill_repairs": run_ctx.get("skill_repair_suggestions") or [],
                "suggested_subgoals": run_ctx.get("suggested_subgoals") or [],
                "alternate_subgoals": self._alternate_subgoals(run_ctx),
                "failure_reason": failure_reason or reward_payload.get("primary_reason"),
            }
            states.append(state_record)
        return states

    def _trace_for_step(self, run_ctx: Dict[str, Any], step_id: Optional[str], reward_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        trace = run_ctx.get("reward_trace") or []
        step_trace = [entry for entry in trace if entry.get("step_id") == step_id]
        if not step_trace:
            step_trace = [{"step_id": step_id, **reward_payload}]
        return step_trace

    def _fingerprint_for_step(self, run_ctx: Dict[str, Any], step_id: Optional[str]) -> Optional[str]:
        preds = run_ctx.get("behavior_predictions") or []
        for entry in preds:
            if entry.get("step_id") == step_id and entry.get("fingerprint"):
                return entry["fingerprint"]
        return run_ctx.get("current_fingerprint")

    def _alternate_subgoals(self, run_ctx: Dict[str, Any]) -> List[str]:
        base = run_ctx.get("suggested_subgoals") or []
        skills = run_ctx.get("skills") or []
        extras: List[str] = []
        for skill in skills:
            meta = skill.get("metadata") or {}
            recommended = meta.get("recommended_subgoals") or []
            extras.extend(recommended)
        combined = list(dict.fromkeys([*(entry for entry in base if isinstance(entry, str)), *extras]))
        return combined

    def _low_confidence_reason(self, reward_trace: Sequence[Dict[str, Any]]) -> Optional[str]:
        for entry in reward_trace:
            confidence = entry.get("confidence") or {}
            band = confidence.get("band") if isinstance(confidence, dict) else None
            if band == "low":
                return "low_confidence"
        return None

    def _read_json(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}


__all__ = ["ExperienceReplayEngine"]
