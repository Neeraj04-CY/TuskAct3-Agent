"""Schemas for Browser Worker v1 run traces."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, NotRequired, TypedDict


class StepAction(TypedDict, total=False):
    """Declarative step payload consumed by the browser worker."""

    id: str
    action: str
    task_id: str | None
    bucket: str | None
    url: str | None
    selector: str | None
    value: str | None
    fields: List[Dict[str, Any]]
    form: str | None
    name: str | None
    timeout: float | int | None
    attempts: int
    durability: str | None
    _precheck: bool
    _recovery: bool
    recovery_hint: str | None
    metadata: Dict[str, Any]


class RunTrace(TypedDict, total=False):
    """Detailed record for a single executed browser step."""

    step_id: str
    step_index: int
    action: str
    status: Literal["ok", "error"]
    start_time: str
    end_time: str
    screenshot_path: str | None
    dom_path: str | None
    error: str | None
    delta_state: Dict[str, Any]
    recovery_applied: bool


class RunSummary(TypedDict):
    """Aggregated execution report for a plan."""

    plan_id: str
    goal: str
    total_steps: int
    failures: int
    recovery_steps: int
    final_url: str
    run_duration: float
    traces: List[RunTrace]
    first_failure_type: str | None
    run_output: str


class FailureReport(TypedDict, total=False):
    """Minimal payload describing a failed browser step."""

    step_id: str
    error: str
    dom_excerpt: str
    worker_trace: Dict[str, Any]
