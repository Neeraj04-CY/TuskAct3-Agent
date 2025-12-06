from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from dashboard.data_loader import load_dashboard_payload
from eikon_engine.config_loader import load_settings
from eikon_engine.pipelines.browser_pipeline import PlannerV3Adapter
from eikon_engine.strategist.behavior_learner import BehaviorLearner
from run_autonomy_demo import build_autonomy_summary, run_single_demo


app = FastAPI(title="TustAct3 API", version="1.0.0")
_settings_cache = load_settings()
_planner_adapter = PlannerV3Adapter(context=_settings_cache.get("planner", {}))
_behavior_learner = BehaviorLearner()
_artifact_root = Path(os.environ.get("EIKON_ARTIFACT_ROOT") or "artifacts").resolve()


class RunRequest(BaseModel):
    goal: str = "Demonstrate the autonomy pipeline"
    execute: bool = False
    allow_sensitive: bool = False


class PlanRequest(BaseModel):
    goal: str


class PredictRequest(BaseModel):
    fingerprint: str
    recent_rewards: List[float] = []
    repair_events: List[Dict[str, Any]] = []


@app.post("/run")
def run_endpoint(payload: RunRequest) -> Dict[str, Any]:
    result = run_single_demo(
        payload.goal,
        execute=payload.execute,
        allow_sensitive=payload.allow_sensitive,
    )
    summary = result["summary"]
    return {
        "summary": summary,
        "run_dir": result["run_dir"],
        "stability": result.get("stability"),
    }


@app.post("/plan")
async def plan_endpoint(payload: PlanRequest) -> Dict[str, Any]:
    plan = await _planner_adapter.create_plan(payload.goal)
    return {"goal": payload.goal, "plan": plan}


@app.post("/predict")
def predict_endpoint(payload: PredictRequest) -> Dict[str, Any]:
    prediction = _behavior_learner.predict(
        payload.fingerprint,
        payload.recent_rewards,
        payload.repair_events,
    )
    return prediction


@app.get("/last_run")
def last_run() -> Dict[str, Any]:
    last_path = _artifact_root / "autonomy" / "latest_run.json"
    if not last_path.exists():
        raise HTTPException(status_code=404, detail="No autonomy runs recorded yet")
    return json.loads(last_path.read_text(encoding="utf-8"))


@app.get("/artifacts/{path:path}")
def artifact_proxy(path: str) -> FileResponse:
    target = (_artifact_root / path).resolve()
    try:
        target.relative_to(_artifact_root)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid artifact path")
    if not target.exists():
        raise HTTPException(status_code=404, detail="Artifact not found")
    return FileResponse(target)


@app.get("/dashboard")
def dashboard_snapshot() -> Dict[str, Any]:
    payload = load_dashboard_payload()
    if not payload.get("available"):
        raise HTTPException(status_code=404, detail=payload.get("message", "Dashboard not available"))
    return payload


__all__ = ["app"]
