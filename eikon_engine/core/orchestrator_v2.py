"""State-aware orchestrator that wires Strategist V2 with the browser worker."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from eikon_engine.core.completion import build_completion
from eikon_engine.core.types import CompletionPayload
from eikon_engine.strategist.strategist_v2 import StrategistV2
from eikon_engine.utils.logging_utils import ArtifactLogger

UTC = timezone.utc
from eikon_engine.workers.browser_worker import BrowserWorker


@dataclass
class OrchestratorV2:
    strategist: StrategistV2
    worker: BrowserWorker
    logger: ArtifactLogger | None = None
    max_steps: int = 40
    transcript: List[Dict[str, Any]] = field(default_factory=list)

    async def run_goal(self, goal: str) -> Dict[str, Any]:
        self.transcript = []
        run_ctx: Dict[str, Any] = {"current_url": None, "history": []}
        await self.strategist.initialize(goal)
        bias_payload = self.strategist.learning_hints({"goal": goal})
        if bias_payload:
            run_ctx["learning_bias"] = bias_payload
            requested = run_ctx.setdefault("requested_skills", [])
            signal_index = {
                entry.get("skill"): entry
                for entry in bias_payload.get("signals", [])
                if isinstance(entry, dict) and entry.get("skill")
            }
            for skill in bias_payload.get("preferred_skills", []):
                if any(existing.get("name") == skill for existing in requested):
                    continue
                entry = {"name": skill, "reason": "learning_bias"}
                signal = signal_index.get(skill)
                if signal:
                    entry["signal"] = signal
                requested.append(entry)
        steps_run = 0
        abort_reason: Optional[str] = None
        start_time = datetime.now(UTC)
        while steps_run < self.max_steps:
            await self.strategist.ensure_plan()
            if not self.strategist.has_next():
                break
            planned_step = self.strategist.peek_step()
            if self.strategist.should_skip_step(run_ctx, planned_step):
                self.strategist.skip_current_step(reason="auto")
                continue
            strategy_step = self.strategist.next_step()
            action_payload = strategy_step.metadata.get("action_payload") or {}
            worker_result = await self.worker.execute({"action": action_payload, "goal": goal})
            self.transcript.append({"step": strategy_step.metadata, "result": worker_result})
            self.strategist.on_step_result(run_ctx, strategy_step.metadata, worker_result)
            self.strategist.record_result(worker_result)
            steps_run += 1
            run_ctx["recent_transition"] = (action_payload.get("action") == "navigate")
            if self.strategist.should_abort():
                abort_reason = "strategist abort"
                break
            dom_failure = worker_result.get("error") == "dom_presence_failed"
            if dom_failure:
                abort_reason = "dom_presence_failed"
                break
            if self.strategist.should_replan(run_ctx, strategy_step.metadata, worker_result):
                abort_reason = "replan requested"
                break
        completion = self.strategist.completion_state()
        if abort_reason:
            completion = build_completion(complete=False, reason=abort_reason, payload={"steps": steps_run})
        end_time = datetime.now(UTC)
        payload = {
            "goal": goal,
            "steps": self.transcript,
            "run_context": run_ctx,
            "completion": completion,
            "strategist_trace": self.strategist.run_trace,
            "duration_seconds": (end_time - start_time).total_seconds(),
            "step_count": steps_run,
        }
        if self.logger:
            payload["artifacts"] = self.logger.to_dict()
        stability_summary = self.strategist.finalize_run(
            run_ctx,
            completion,
            payload["duration_seconds"],
            payload.get("artifacts"),
        )
        if stability_summary:
            payload["stability"] = stability_summary
        return payload


__all__ = ["OrchestratorV2"]
