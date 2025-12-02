"""Planner v3: rule-based browser action synthesis with DAG validation.

Phase 2 adds semantic grouping, multi-goal splitting, reflection hints, and plan scoring
while keeping earlier behavior backward compatible. Phase 3 layers execution-aware
planning with durability analysis, automated prechecks, recovery steps, and richer
metadata to help downstream orchestration make safer choices. Phase 4 introduces
adaptive replanning hooks: failure classification, partial-plan deltas, repair
generators, and scoring hooks that let the orchestrator resume, restart, or abort
individual steps based on live execution feedback.

Example usage:
>>> from eikon_engine.planning import planner_v3_plan_by_default
>>> plan = planner_v3_plan_by_default("Log in to demo site", context={})

Adaptive replanning flow:
1. Execute plan tasks until a step fails.
2. Call classify_failure(step_result) to label the issue.
3. Provide the failure payload and original plan to replan_after_step to obtain
    PartialPlan deltas (insert/replace/repair steps) plus metadata flags for the
    orchestrator.
4. Apply returned deltas without regenerating the entire plan, then continue or
    restart according to the adaptive flags.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, TypedDict
from uuid import uuid4
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


class TaskInput(TypedDict):
    actions: List[Dict[str, Any]]


class Task(TypedDict):
    id: str
    tool: str
    inputs: TaskInput
    depends_on: List[str]
    bucket: str


class Plan(TypedDict):
    plan_id: str
    goal: str
    tasks: List[Task]
    meta: Dict[str, Any]


class PlanDelta(TypedDict):
    type: str
    target_task: str
    new_steps: List[Dict[str, Any]]


class PartialPlan(TypedDict):
    deltas: List[PlanDelta]
    meta: Dict[str, Any]


_URL_RE = re.compile(r"(?:https?://|file:///?)[^\s\'\"<>]+", re.IGNORECASE)
_ALLOWED_TOOLS = {"BrowserWorker"}
_ALLOWED_ACTIONS = {
    "navigate",
    "fill",
    "click",
    "wait_for",
    "extract",
    "screenshot",
    "wait_for_selector",
    "dom_presence_check",
    "retry",
    "reload_if_failed",
}
_SUBGOAL_SPLIT_RE = re.compile(r"\b(?:and then|then|after that|next)\b", re.IGNORECASE)
_RISK_WEIGHTS = {"low": 2, "medium": 1, "high": 0}
_REPLAN_PENALTIES = {
    "selector_missing": 0.05,
    "navigation_loop": 0.04,
    "auth_wall": 0.08,
    "dom_changed": 0.03,
    "rate_limited": 0.02,
    "unknown": 0.02,
}


def classify_goal(goal_text: str) -> str:
    text = (goal_text or "").lower()
    categories = []
    if any(keyword in text for keyword in ("login", "log in", "sign in")):
        categories.append("login")
    if any(keyword in text for keyword in ("navigate", "open", "go to")):
        categories.append("navigation")
    if any(keyword in text for keyword in ("extract", "scrape", "capture")):
        categories.append("extract")
    if any(keyword in text for keyword in ("form", "submit", "fill", "input")):
        categories.append("form")
    if len(categories) > 1:
        return "multi"
    return categories[0] if categories else "unknown"


def extract_known_urls(goal_text: str) -> List[str]:
    if not goal_text:
        return []
    matches = _URL_RE.findall(goal_text)
    logger.debug("Extracted URLs", extra={"urls": matches})
    return matches


def generate_initial_steps(goal_text: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
    steps: List[Dict[str, Any]] = []
    classification = context.get("classification") or classify_goal(goal_text)
    urls: Sequence[str] = context.get("known_urls") or []
    default_url: Optional[str] = context.get("default_url") or (urls[0] if urls else None)
    credentials: Dict[str, str] = context.get("credentials") or {}
    username = credentials.get("username", "")
    password = credentials.get("password", "")
    step_counter = 0

    def _new_step(action: str, args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        nonlocal step_counter
        step_counter += 1
        if action not in _ALLOWED_ACTIONS:
            raise ValueError(f"Unsupported action: {action}")
        return {"id": f"s{step_counter}", "action": action, "args": args or {}}

    if default_url:
        steps.append(_new_step("navigate", {"url": default_url}))

    if classification in {"login", "multi"}:
        steps.extend(
            [
                _new_step("fill", {"selector": "#username", "value": username, "form": "login"}),
                _new_step("fill", {"selector": "#password", "value": password, "form": "login"}),
                _new_step("click", {"selector": "button[type=submit]"}),
                _new_step("wait_for", {"selector": "#flash", "timeout": 2000}),
            ]
        )
    elif classification in {"navigation", "unknown"} and not steps:
        steps.append(_new_step("navigate", {"url": default_url or "https://example.com"}))
    elif classification in {"extract"}:
        steps.append(_new_step("extract", {"selector": context.get("extract_selector", "body")}))
    elif classification == "form":
        steps.append(_new_step("fill", {"selector": context.get("form_selector", "input"), "value": context.get("form_value", "")}))

    logger.debug("Generated raw steps", extra={"steps": steps})
    return steps


def optimize_steps(raw_steps: List[Dict[str, Any]], context: Dict[str, Any]) -> List[Dict[str, Any]]:
    _ = context
    optimized: List[Dict[str, Any]] = []
    pending_fills: List[Dict[str, Any]] = []

    last_step: Optional[Dict[str, Any]] = None

    def _flush_fills() -> None:
        nonlocal pending_fills, last_step
        if not pending_fills:
            return
        if len(pending_fills) == 1:
            optimized.append(pending_fills[0])
            last_step = pending_fills[0]
        else:
            form = pending_fills[0]["args"].get("form", "default")
            fields = [
                {"selector": step["args"].get("selector"), "value": step["args"].get("value")}
                for step in pending_fills
            ]
            merged_step = {
                "id": pending_fills[0]["id"],
                "action": "fill",
                "args": {"form": form, "fields": fields},
            }
            optimized.append(merged_step)
            last_step = merged_step
        pending_fills = []

    for step in raw_steps:
        action = step.get("action")
        args = step.get("args") or {}
        step["args"] = args
        if action == "fill":
            form_key = args.get("form", "default")
            if pending_fills and (pending_fills[-1].get("args") or {}).get("form", "default") != form_key:
                _flush_fills()
            pending_fills.append(step)
            continue
        _flush_fills()
        if action == "navigate" and last_step and last_step.get("action") == "navigate":
            if (last_step.get("args") or {}).get("url") == args.get("url"):
                logger.debug("Skipping redundant navigation", extra={"url": step.get("args", {}).get("url")})
                continue
        if action == "wait_for" and not args.get("selector"):
            logger.debug("Removing no-op wait", extra={"step": step})
            continue
        optimized.append(step)
        last_step = step

    _flush_fills()
    logger.debug("Optimized steps", extra={"steps": optimized})
    return optimized


def validate_dag(tasks: Sequence[Task]) -> None:
    ids = [task["id"] for task in tasks]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate task IDs detected")
    id_set = set(ids)
    graph = {task["id"]: task.get("depends_on", []) for task in tasks}

    for task in tasks:
        tool = task.get("tool")
        if tool not in _ALLOWED_TOOLS:
            raise ValueError(f"Unsupported tool: {tool}")
        for dep in task.get("depends_on", []):
            if dep not in id_set:
                raise ValueError(f"Missing dependency: {dep}")

    visited = set()
    stack = set()

    def _visit(node: str) -> None:
        if node in stack:
            raise ValueError("Cycle detected in task graph")
        if node in visited:
            return
        stack.add(node)
        for neighbor in graph.get(node, []):
            _visit(neighbor)
        stack.remove(node)
        visited.add(node)

    for node in graph:
        _visit(node)


def plan_from_goal(goal_text: str, context: Optional[Dict[str, Any]] = None) -> Plan:
    if not goal_text or not goal_text.strip():
        raise ValueError("goal_text cannot be empty")
    subgoals = split_goal_into_subgoals(goal_text)
    if len(subgoals) > 1:
        subplans = [_plan_single_goal(text.strip(), context or {}) for text in subgoals]
        return merge_plans_sequential(subplans)
    return _plan_single_goal(goal_text.strip(), context or {})


def _plan_single_goal(goal_text: str, context: Dict[str, Any]) -> Plan:
    ctx = dict(context)
    classification = classify_goal(goal_text)
    urls = extract_known_urls(goal_text)
    ctx.setdefault("classification", classification)
    ctx.setdefault("known_urls", urls)
    ctx.setdefault("goal_text", goal_text)
    raw_steps = generate_initial_steps(goal_text, ctx)
    if not raw_steps:
        raise ValueError("Planner v3 could not derive any steps from goal")
    optimized_steps = optimize_steps(raw_steps, ctx)
    if not optimized_steps:
        raise ValueError("Planner v3 optimization removed all steps")
    tasks = group_steps_into_tasks_grouped(optimized_steps)
    pipeline_order = ["raw", "grouping"]
    _apply_durability_annotations(tasks, ctx)
    pipeline_order.append("durability")
    tasks = inject_prechecks(tasks)
    pipeline_order.append("prechecks")
    tasks = inject_recovery_steps(tasks)
    pipeline_order.append("recovery")
    _link_dependencies(tasks)
    validate_dag(tasks)
    reflection = generate_reflection(goal_text, optimized_steps, ctx)
    pipeline_order.append("scoring")
    base_score = score_plan(tasks)
    if not reflection.get("warnings"):
        base_score += 0.05
    durability_summary, execution_risk_score = _summarize_durability(tasks)
    precheck_count = _count_flagged_actions(tasks, "_precheck")
    recovery_task_count = sum(1 for task in tasks if any(action.get("_recovery") for action in task["inputs"]["actions"]))
    total_retries = sum(action.get("attempts", 0) for task in tasks for action in task["inputs"]["actions"] if action.get("action") == "retry")
    stability_bonus = 0.1 if durability_summary.get("low", 0) == 0 else 0.0
    replanning_penalty = 0.0
    adjusted_score = round(max(base_score + stability_bonus - replanning_penalty, 0.0), 3)
    plan: Plan = {
        "plan_id": str(uuid4()),
        "goal": goal_text,
        "tasks": tasks,
        "meta": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source": "planner_v3",
            "classification": classification,
            "reflection": reflection,
            "score": adjusted_score,
            "durability_summary": durability_summary,
            "execution_risk_score": execution_risk_score,
            "precheck_count": precheck_count,
            "recovery_count": recovery_task_count,
            "total_retries": total_retries,
            "pipeline_order": pipeline_order,
            "stability_bonus": stability_bonus,
            "replanning_penalty": replanning_penalty,
        },
    }
    logger.info(
        "Generated single-goal plan",
        extra={"plan_id": plan["plan_id"], "task_count": len(tasks), "score": plan["meta"]["score"]},
    )
    return plan


def classify_step_into_bucket(step: Dict[str, Any]) -> str:
    action = (step.get("action") or "").lower()
    if action == "navigate":
        return "navigation"
    if action == "fill":
        return "form"
    if action == "extract":
        return "extraction"
    if action == "wait_for":
        return "wait"
    if action == "screenshot":
        return "screenshot"
    return "misc"


def group_steps_into_tasks_grouped(steps: Sequence[Dict[str, Any]]) -> List[Task]:
    if not steps:
        raise ValueError("Planner v3 requires at least one step to build tasks")

    blocks: List[List[Dict[str, Any]]] = []
    current_block: List[Dict[str, Any]] = []

    def _flush_block() -> None:
        nonlocal current_block
        if current_block:
            blocks.append(current_block)
            current_block = []

    for step in steps:
        if step.get("action") == "navigate":
            _flush_block()
            current_block.append(step)
            _flush_block()
            continue
        current_block.append(step)
    _flush_block()

    if not blocks:
        blocks = [list(steps)]

    tasks: List[Task] = []
    for idx, block in enumerate(blocks, start=1):
        bucket = classify_step_into_bucket(block[0])
        tasks.append(
            Task(
                id=f"task_{idx}",
                tool="BrowserWorker",
                inputs=TaskInput(actions=[_step_to_action(step) for step in block]),
                depends_on=[],
                bucket=bucket,
            )
        )

    _link_dependencies(tasks)
    return tasks


def estimate_durability(step: Dict[str, Any], context: Dict[str, Any]) -> str:
    action = (step.get("action") or "").lower()
    text = (context.get("goal_text") or "").lower()
    selector = _infer_selector(step)
    url = (step.get("url") or "").lower()

    if action == "navigate":
        if not url or any(token in url for token in ("login", "auth", "redirect")) or "dynamic" in text:
            return "medium"
        return "high"
    if action == "fill":
        return "low" if _is_vague_form_step(step) else "medium"
    if action == "click":
        return "low" if not selector else "medium"
    if action in {"wait_for", "wait_for_selector", "dom_presence_check"}:
        return "medium" if selector else "low"
    if action == "extract":
        return "medium" if "dynamic" in text else "high"
    if action in {"retry", "reload_if_failed"}:
        return "medium"
    if action == "screenshot":
        return "high"
    return "medium"


def inject_prechecks(tasks: Sequence[Task]) -> List[Task]:
    updated: List[Task] = []
    for task in tasks:
        new_actions: List[Dict[str, Any]] = []
        for action in task["inputs"]["actions"]:
            selector = _infer_selector(action) or action.get("url") or "body"
            durability = action.get("durability", "medium")
            if durability == "low":
                new_actions.append(
                    {
                        "action": "wait_for_selector",
                        "selector": selector,
                        "timeout": 2000,
                        "durability": "high",
                        "_precheck": True,
                    }
                )
            if action.get("action") in {"click", "fill"}:
                new_actions.append(
                    {
                        "action": "dom_presence_check",
                        "selector": selector,
                        "durability": "medium",
                        "_precheck": True,
                    }
                )
            new_actions.append(action)
        task["inputs"]["actions"] = new_actions
        updated.append(task)
    return updated


def inject_recovery_steps(tasks: Sequence[Task]) -> List[Task]:
    new_tasks: List[Task] = list(tasks)
    baseline_tasks = list(tasks)
    next_index = len(new_tasks) + 1
    for task in baseline_tasks:
        bucket = task.get("bucket", "misc")
        action_list = task["inputs"]["actions"]
        durability_levels = {action.get("durability", "medium") for action in action_list}
        has_navigation = any(action.get("action") == "navigate" for action in action_list)
        if any(level in {"low", "medium"} for level in durability_levels):
            recovery_task = Task(
                id=f"task_{next_index}",
                tool="BrowserWorker",
                inputs=TaskInput(
                    actions=[
                        {
                            "action": "retry",
                            "attempts": 2,
                            "target_task": task["id"],
                            "durability": "medium",
                            "_recovery": True,
                        }
                    ]
                ),
                depends_on=[],
                bucket=bucket,
            )
            new_tasks.append(recovery_task)
            next_index += 1
        if has_navigation:
            reload_task = Task(
                id=f"task_{next_index}",
                tool="BrowserWorker",
                inputs=TaskInput(
                    actions=[
                        {
                            "action": "reload_if_failed",
                            "target_task": task["id"],
                            "durability": "medium",
                            "_recovery": True,
                        }
                    ]
                ),
                depends_on=[],
                bucket=bucket,
            )
            new_tasks.append(reload_task)
            next_index += 1
    return new_tasks


def _apply_durability_annotations(tasks: Sequence[Task], context: Dict[str, Any]) -> None:
    for task in tasks:
        for action in task["inputs"]["actions"]:
            action["durability"] = estimate_durability(action, context)


def _summarize_durability(tasks: Sequence[Task]) -> tuple[Dict[str, int], int]:
    summary = {"low": 0, "medium": 0, "high": 0}
    risk_score = 0
    for task in tasks:
        for action in task["inputs"]["actions"]:
            durability = action.get("durability", "medium")
            summary.setdefault(durability, 0)
            summary[durability] += 1
            risk_score += _RISK_WEIGHTS.get(durability, 1)
    # Ensure all keys exist even if not encountered
    for key in ("low", "medium", "high"):
        summary.setdefault(key, 0)
    return summary, risk_score


def _infer_selector(step: Dict[str, Any]) -> str:
    if step.get("selector"):
        return str(step.get("selector"))
    fields = step.get("fields") or []
    if fields:
        first = fields[0] or {}
        return str(first.get("selector", ""))
    return str(step.get("target") or "")


def _is_vague_form_step(step: Dict[str, Any]) -> bool:
    fields = step.get("fields") or []
    if fields:
        return any(not field.get("selector") for field in fields)
    return not step.get("selector") or not (step.get("value") or fields)


def _count_flagged_actions(tasks: Sequence[Task], flag: str) -> int:
    return sum(1 for task in tasks for action in task["inputs"]["actions"] if action.get(flag))


def _link_dependencies(tasks: List[Task]) -> None:
    for idx in range(1, len(tasks)):
        tasks[idx]["depends_on"] = [tasks[idx - 1]["id"]]
    if tasks:
        tasks[0]["depends_on"] = []


def _step_to_action(step: Dict[str, Any]) -> Dict[str, Any]:
    payload = {"action": step.get("action")}
    payload.update(step.get("args", {}))
    return payload


def generate_reflection(goal_text: str, steps: Sequence[Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, List[str]]:
    text = goal_text.lower()
    missing_fields: List[str] = []
    warnings: List[str] = []
    hints: List[str] = []

    if "login" in text:
        creds = context.get("credentials") or {}
        if not creds.get("username"):
            missing_fields.append("username")
        if not creds.get("password"):
            missing_fields.append("password")
        for step in steps:
            if step.get("action") == "fill":
                if not (step.get("args") or {}).get("value"):
                    missing_fields.append(step.get("args", {}).get("selector", "field"))

    if ("stay logged in" in text and "log out" in text) or ("login" in text and "logout" in text):
        warnings.append("Goal contains conflicting login/logout instructions")

    domains = {
        urlparse(url).netloc.lower() or urlparse(url).path.lower()
        for url in extract_known_urls(goal_text)
    }
    domains.discard("")
    if len(domains) > 1:
        warnings.append("Multiple distinct domains referenced: " + ", ".join(sorted(domains)))

    if "screenshot" in text and not any(step.get("action") == "screenshot" for step in steps):
        hints.append("Consider adding a screenshot action")

    return {
        "missing_fields": sorted(set(missing_fields)),
        "warnings": warnings,
        "hints": hints,
    }


def split_goal_into_subgoals(goal_text: str) -> List[str]:
    segments = [segment.strip() for segment in _SUBGOAL_SPLIT_RE.split(goal_text) if segment.strip()]
    return segments if len(segments) >= 2 else [goal_text]


def merge_plans_sequential(plans: Sequence[Plan]) -> Plan:
    if not plans:
        raise ValueError("No plans provided for merging")
    merged_tasks: List[Task] = []
    combined_reflection = {
        "missing_fields": [],
        "warnings": [],
        "hints": [],
    }
    durability_summary = {"low": 0, "medium": 0, "high": 0}
    total_retries = 0
    precheck_count = 0
    recovery_count = 0
    execution_risk_score = 0
    stability_bonus = 0.0
    replanning_penalty = 0.0
    for plan in plans:
        reflection = plan["meta"].get("reflection") or {"missing_fields": [], "warnings": [], "hints": []}
        for key in combined_reflection:
            combined_reflection[key].extend(reflection.get(key, []))
        summary = plan["meta"].get("durability_summary") or {}
        for level in durability_summary:
            durability_summary[level] += summary.get(level, 0)
        total_retries += plan["meta"].get("total_retries", 0)
        precheck_count += plan["meta"].get("precheck_count", 0)
        recovery_count += plan["meta"].get("recovery_count", 0)
        execution_risk_score += plan["meta"].get("execution_risk_score", 0)
        stability_bonus += plan["meta"].get("stability_bonus", 0.0)
        replanning_penalty += plan["meta"].get("replanning_penalty", 0.0)
        for task in plan["tasks"]:
            existing_actions = task.get("inputs", {}).get("actions", [])
            actions_copy = [dict(action) for action in existing_actions]
            fallback_step = actions_copy[0] if actions_copy else {"action": "misc"}
            new_task = Task(
                id=f"task_{len(merged_tasks) + 1}",
                tool=task.get("tool", "BrowserWorker"),
                inputs=TaskInput(actions=actions_copy),
                depends_on=[],
                bucket=task.get("bucket", classify_step_into_bucket(fallback_step)),
            )
            merged_tasks.append(new_task)
    _link_dependencies(merged_tasks)
    validate_dag(merged_tasks)
    reflection = {
        key: sorted(set(values))
        for key, values in combined_reflection.items()
    }
    score = score_plan(merged_tasks)
    if not reflection.get("warnings"):
        score += 0.05
    final_score = round(max(score + stability_bonus - replanning_penalty, 0.0), 3)
    merged_plan: Plan = {
        "plan_id": str(uuid4()),
        "goal": " -> ".join(plan["goal"] for plan in plans),
        "tasks": merged_tasks,
        "meta": {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source": "planner_v3",
            "classification": "multi",
            "reflection": reflection,
            "score": final_score,
            "subplans": [plan["plan_id"] for plan in plans],
            "durability_summary": durability_summary,
            "total_retries": total_retries,
            "precheck_count": precheck_count,
            "recovery_count": recovery_count,
            "execution_risk_score": execution_risk_score,
            "pipeline_order": ["multi_merge"],
            "stability_bonus": stability_bonus,
            "replanning_penalty": replanning_penalty,
        },
    }
    logger.info(
        "Merged multi-goal plan",
        extra={"plan_id": merged_plan["plan_id"], "task_count": len(merged_tasks)},
    )
    return merged_plan


def score_plan(tasks: Sequence[Task]) -> float:
    if not tasks:
        return 0.0
    score = 1.0
    score -= 0.05 * max(len(tasks) - 1, 0)
    first_actions = tasks[0]["inputs"]["actions"]
    if first_actions and first_actions[0].get("action") == "navigate":
        score += 0.05
    if len(tasks) <= 2:
        score += 0.05
    return round(score, 3)


def classify_failure(step_result: Dict[str, Any]) -> Dict[str, str]:
    payload = step_result or {}
    meta = payload.get("meta") or {}
    error = (payload.get("error") or "").lower()
    detail = payload.get("detail") or error or meta.get("message") or "Adaptive replanning triggered"

    failure_type = "unknown"
    url_history = meta.get("url_history") or payload.get("url_history") or []
    repeated_url = isinstance(url_history, list) and len(url_history) >= 3 and len(set(url_history[-3:])) == 1
    status_code = meta.get("status_code") or payload.get("status_code")

    if payload.get("missing_selector") or "selector" in error:
        failure_type = "selector_missing"
    elif meta.get("navigation_loop") or meta.get("loop_url") or "too many redirects" in error or repeated_url:
        failure_type = "navigation_loop"
    elif meta.get("rate_limited") or status_code in {429, 503} or "rate limit" in error:
        failure_type = "rate_limited"
    elif meta.get("dom_changed") or payload.get("dom_changed") or "stale element" in error:
        failure_type = "dom_changed"
    elif status_code in {401, 403} or "auth" in error:
        failure_type = "auth_wall"
    return {"type": failure_type, "detail": detail}


def replan_after_step(step_result: Dict[str, Any], previous_plan: Plan) -> PartialPlan:
    if not previous_plan:
        raise ValueError("previous_plan is required for replanning")
    failure = classify_failure(step_result)
    failure_type = failure["type"]
    tasks = previous_plan.get("tasks", [])
    target_task_id = step_result.get("task_id") or (tasks[-1]["id"] if tasks else "")
    target_bucket = _get_task_bucket(previous_plan, target_task_id)
    context = {"goal_text": previous_plan.get("goal", "")}
    context.update(step_result.get("context") or {})

    repair_builders = {
        "selector_missing": generate_selector_repair,
        "navigation_loop": generate_navigation_repair,
        "auth_wall": generate_auth_repair,
        "dom_changed": generate_dom_repair,
        "rate_limited": generate_rate_limit_repair,
    }
    repair_fn = repair_builders.get(failure_type, generate_dom_repair)
    new_steps = repair_fn(step_result, previous_plan, target_bucket, context)

    delta_type_map = {
        "selector_missing": "replace",
        "navigation_loop": "insert",
        "auth_wall": "replace",
        "dom_changed": "insert",
        "rate_limited": "insert",
        "unknown": "insert",
    }
    delta_type = delta_type_map.get(failure_type, "insert")
    deltas: List[PlanDelta] = []
    if new_steps:
        deltas.append(
            {"type": delta_type, "target_task": target_task_id, "new_steps": new_steps}
        )

    flags = _derive_resume_flags(failure_type)
    penalty = _REPLAN_PENALTIES.get(failure_type, _REPLAN_PENALTIES["unknown"])
    partial: PartialPlan = {
        "deltas": deltas,
        "meta": {
            "reason": failure["detail"],
            "failure_type": failure_type,
            "should_resume": flags["should_resume"],
            "should_restart_task": flags["should_restart_task"],
            "should_abort_run": flags["should_abort_run"],
            "replanning_penalty": penalty,
        },
    }
    return partial


def generate_selector_repair(
    step_result: Dict[str, Any], previous_plan: Plan, bucket: str, context: Dict[str, Any]
) -> List[Dict[str, Any]]:
    step_payload = dict(step_result.get("step") or {})
    selector = step_payload.get("selector") or step_result.get("selector") or "#target"
    wait_step = {"action": "wait_for_selector", "selector": selector, "timeout": 2500, "_durability_hint": "high"}
    repaired_step = dict(step_payload) or {"action": "click"}
    repaired_step.setdefault("selector", selector)
    repaired_step.setdefault("action", "click")
    repaired_step["_durability_hint"] = "low"
    return _guard_steps_with_phase3([wait_step, repaired_step], bucket or "form", context)


def generate_navigation_repair(
    step_result: Dict[str, Any], previous_plan: Plan, bucket: str, context: Dict[str, Any]
) -> List[Dict[str, Any]]:
    meta = step_result.get("meta") or {}
    recovery_url = meta.get("recovery_url") or meta.get("loop_url") or meta.get("current_url")
    if not recovery_url:
        urls = extract_known_urls(previous_plan.get("goal", ""))
        recovery_url = urls[0] if urls else "https://example.com"
    wait_selector = meta.get("wait_selector") or "body"
    base_steps = [
        {"action": "navigate", "url": recovery_url, "_durability_hint": "medium"},
        {"action": "wait_for", "selector": wait_selector, "timeout": 2000, "_durability_hint": "medium"},
    ]
    return _guard_steps_with_phase3(base_steps, bucket or "navigation", context)


def generate_auth_repair(
    step_result: Dict[str, Any], previous_plan: Plan, bucket: str, context: Dict[str, Any]
) -> List[Dict[str, Any]]:
    meta = step_result.get("meta") or {}
    ctx = dict(context)
    ctx.update(step_result.get("context") or {})
    creds = ctx.get("credentials") or {}
    username = creds.get("username", "USERNAME_PLACEHOLDER")
    password = creds.get("password", "PASSWORD_PLACEHOLDER")
    login_url = meta.get("login_url")
    if not login_url:
        urls = extract_known_urls(previous_plan.get("goal", ""))
        login_url = urls[0] if urls else meta.get("current_url") or "https://example.com/login"
    base_steps = [
        {"action": "navigate", "url": login_url, "_durability_hint": "medium"},
        {"action": "fill", "selector": "#username", "value": username, "_durability_hint": "low"},
        {"action": "fill", "selector": "#password", "value": password, "_durability_hint": "low"},
        {"action": "click", "selector": "button[type=submit]", "_durability_hint": "low"},
    ]
    return _guard_steps_with_phase3(base_steps, bucket or "form", context)


def generate_dom_repair(
    step_result: Dict[str, Any], previous_plan: Plan, bucket: str, context: Dict[str, Any]
) -> List[Dict[str, Any]]:
    meta = step_result.get("meta") or {}
    selector = meta.get("changed_selector") or step_result.get("selector") or "body"
    base_steps = [
        {"action": "wait_for", "selector": selector, "timeout": 2500, "_durability_hint": "medium"},
        {"action": "screenshot", "name": "dom_repair", "_durability_hint": "high"},
    ]
    return _guard_steps_with_phase3(base_steps, bucket or "misc", context)


def generate_rate_limit_repair(
    step_result: Dict[str, Any], previous_plan: Plan, bucket: str, context: Dict[str, Any]
) -> List[Dict[str, Any]]:
    meta = step_result.get("meta") or {}
    wait_ms = meta.get("retry_after_ms") or 5000
    base_steps = [
        {"action": "wait_for", "selector": "body", "timeout": wait_ms, "_durability_hint": "high"},
        {"action": "retry", "attempts": 1, "target_task": step_result.get("task_id"), "_durability_hint": "medium"},
    ]
    return _guard_steps_with_phase3(base_steps, bucket or "misc", context)


def _guard_steps_with_phase3(base_steps: Sequence[Dict[str, Any]], bucket: str, context: Dict[str, Any]) -> List[Dict[str, Any]]:
    bucket = bucket or "misc"
    context = context or {}
    actions = [dict(step) for step in base_steps]
    temp_task = Task(
        id="delta_task",
        tool="BrowserWorker",
        inputs=TaskInput(actions=actions),
        depends_on=[],
        bucket=bucket,
    )
    tasks = [temp_task]
    _apply_durability_annotations(tasks, context)
    for action in temp_task["inputs"]["actions"]:
        hint = action.pop("_durability_hint", None)
        if hint:
            action["durability"] = hint
    tasks = inject_prechecks(tasks)
    tasks = inject_recovery_steps(tasks)
    guarded_steps: List[Dict[str, Any]] = []
    for idx, task in enumerate(tasks):
        for action in task["inputs"]["actions"]:
            action_copy = dict(action)
            if idx > 0:
                action_copy.setdefault("task_boundary", True)
            guarded_steps.append(action_copy)
    return guarded_steps


def _derive_resume_flags(failure_type: str) -> Dict[str, bool]:
    mapping = {
        "selector_missing": {"should_resume": True, "should_restart_task": False, "should_abort_run": False},
        "navigation_loop": {"should_resume": True, "should_restart_task": False, "should_abort_run": False},
        "auth_wall": {"should_resume": False, "should_restart_task": True, "should_abort_run": False},
        "dom_changed": {"should_resume": True, "should_restart_task": False, "should_abort_run": False},
        "rate_limited": {"should_resume": True, "should_restart_task": False, "should_abort_run": False},
        "unknown": {"should_resume": True, "should_restart_task": False, "should_abort_run": False},
    }
    return mapping.get(failure_type, mapping["unknown"])


def _get_task_bucket(plan: Plan, task_id: str) -> str:
    for task in plan.get("tasks", []):
        if task.get("id") == task_id:
            return task.get("bucket", "misc")
    return plan.get("tasks", [{}])[0].get("bucket", "misc") if plan.get("tasks") else "misc"


# Example snippet illustrating usage:
# import json
# example_goal = "Log in to https://the-internet.herokuapp.com/login with username tomsmith and password SuperSecretPassword!"
# example_plan = plan_from_goal(
#     example_goal,
#     context={"credentials": {"username": "tomsmith", "password": "SuperSecretPassword!"}},
# )
# print(json.dumps(example_plan, indent=2))
