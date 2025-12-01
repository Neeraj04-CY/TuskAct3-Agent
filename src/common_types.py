from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RiskItem:
    """Represents a single risk associated with executing a workflow or step."""
    description: str
    likelihood: str  # e.g., "low", "medium", "high"
    impact: str      # e.g., "low", "medium", "high"
    mitigation: Optional[str] = None


@dataclass
class WorkflowStep:
    """Single executable step inside an execution plan."""
    step_id: str
    description: str
    tool: Optional[str] = None
    skill: Optional[str] = None
    inputs: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)


@dataclass
class WorkflowObject:
    """
    Strategist output: fully structured workflow description passed to the Worker.
    Mirrors the spec in the prompt but uses Pythonic types.
    """
    workflow_id: str
    task: str
    subtasks: List[str]
    tools_needed: List[str]
    skills_to_load: List[str]
    memory_references: List[str]
    constraints: List[str]
    risk_analysis: List[RiskItem]
    success_criteria: List[str]
    estimated_time: str
    execution_plan: List[WorkflowStep]


@dataclass
class WorkerStepResult:
    """
    Worker output for a single step, as per the spec.
    """
    step: str
    status: str  # "pending" | "running" | "success" | "failed" | "skipped"
    result: Any = None
    error: Optional[str] = None
    retry_count: int = 0
    artifacts: List[str] = field(default_factory=list)


@dataclass
class WorkflowSchedule:
    """
    Workflow Engine's higher-level schedule object.
    """
    init: Dict[str, Any]
    pre_checks: List[WorkflowStep]
    execution: List[WorkflowStep]
    post_actions: List[WorkflowStep]
    memory_update: List[WorkflowStep]