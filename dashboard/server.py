from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from .data_loader import load_dashboard_payload

app = FastAPI(title="Showcase Dashboard", version="1.0.0")


@app.get("/api/dashboard")
def dashboard_payload() -> dict:
    payload = load_dashboard_payload()
    if not payload.get("available"):
        raise HTTPException(status_code=404, detail=payload.get("message", "No run available"))
    return payload


_static_dir = Path(__file__).parent / "static"
app.mount("/", StaticFiles(directory=_static_dir, html=True), name="dashboard-static")

_artifact_root = Path(os.environ.get("EIKON_ARTIFACT_ROOT") or "artifacts")
if _artifact_root.exists():
    app.mount("/artifacts", StaticFiles(directory=_artifact_root), name="dashboard-artifacts")


__all__ = ["app"]
