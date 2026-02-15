"""Mission planning helpers that leverage Planner v3."""

from __future__ import annotations

import re
from urllib.parse import quote_plus, urlparse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from eikon_engine.planning.planner_v3 import plan_from_goal

from .mission_schema import MissionSpec, MissionSubgoal

_SENSITIVE_KEYS = ("password", "secret", "token", "key")
_DURABILITY_SCORES = {"low": 0, "medium": 1, "high": 2}
_MISSION_URL_PATTERN = re.compile(r"https?://[^\s]+")
_LOGIN_KEYWORDS = ("login", "log in", "sign in", "signin")
_LOGIN_USERNAME = "student"
_LOGIN_PASSWORD = "Password123"
_DEMO_LOGIN_HTML = Path(__file__).resolve().parents[2] / "examples" / "demo_local_testsite" / "login.html"
_DEFAULT_SEARCH_ENGINE = "https://duckduckgo.com/?q="


class MissionPlanningError(RuntimeError):
    """Raised when the mission planner cannot produce subgoals."""


def plan_mission(mission_spec: MissionSpec, *, settings: Optional[Dict[str, Any]] = None) -> List[MissionSubgoal]:
    """Return ordered mission subgoals using Planner v3."""

    safe_context = _scrub_constraints(mission_spec.constraints or {}, mission_spec.allow_sensitive)
    try:
        plan = plan_from_goal(mission_spec.instruction, context=safe_context or None)
    except Exception as exc:  # noqa: BLE001
        raise MissionPlanningError(str(exc)) from exc
    tasks = plan.get("tasks", [])
    if not tasks:
        raise MissionPlanningError("planner_empty")
    mission_url = _extract_mission_url(mission_spec.instruction)
    mission_url = _override_demo_navigation_url(mission_url, mission_spec, settings)
    if not mission_url:
        search_url = _build_default_search_url(mission_spec.instruction)
        safe_context = dict(safe_context or {})
        safe_context.setdefault("default_url", search_url)
        known_urls = list(safe_context.get("known_urls") or [])
        if search_url not in known_urls:
            safe_context["known_urls"] = [search_url] + known_urls
        plan = plan_from_goal(mission_spec.instruction, context=safe_context)
        tasks = _rewrite_navigation_targets(plan.get("tasks", []), search_url)
        mission_url = search_url
    if mission_url:
        print("[DEBUG] Mission URL:", mission_url)
        tasks = _rewrite_navigation_targets(tasks, mission_url)
        tasks = _remove_redundant_navigation_tasks(tasks, mission_url)
        constraints = dict(mission_spec.constraints or {})
        constraints.setdefault("default_url", mission_url)
        known_urls = list(constraints.get("known_urls") or [])
        if mission_url not in known_urls:
            constraints["known_urls"] = [mission_url] + known_urls
        mission_spec.constraints = constraints
    constraint_flags = mission_spec.constraints if isinstance(mission_spec.constraints, dict) else {}
    login_chain_disabled = bool(constraint_flags.get("disable_login_chain") or constraint_flags.get("disable_login_skill"))
    inject_login_chain = False
    if (
        not login_chain_disabled
        and _should_inject_login_chain(mission_spec.instruction, mission_url)
        and _has_dom_presence_task(tasks)
    ):
        inject_login_chain = True
        tasks = _strip_dom_presence_tasks(tasks)
        tasks = []
    plan_meta = plan.get("meta", {})
    subgoals: List[MissionSubgoal] = []
    for idx, task in enumerate(tasks, start=1):
        actions = task.get("inputs", {}).get("actions", []) or []
        metadata = {
            "plan_id": plan.get("plan_id"),
            "task_id": task.get("id"),
            "bucket": task.get("bucket"),
            "estimated_step_count": len(actions),
            "durability_score": _score_actions(actions, plan_meta.get("durability_summary", {})),
            "capability_requirements": task.get("capabilities", []),
        }
        description = _describe_task(task, idx)
        subgoals.append(MissionSubgoal(id=f"{mission_spec.id}_sg{idx}", description=description, planner_metadata=metadata))
    insertion_offset = 0
    if mission_url:
        nav_metadata = {
            "plan_id": plan.get("plan_id"),
            "bucket": "navigation",
            "estimated_step_count": 1,
            "durability_score": float(_DURABILITY_SCORES["high"]),
            "primary_url": mission_url,
            "bootstrap_actions": [{"action": "navigate", "url": mission_url}],
        }
        nav_description = f"00. navigation: navigate to {mission_url}"
        nav_subgoal = MissionSubgoal(
            id=f"{mission_spec.id}_sg0",
            description=nav_description,
            planner_metadata=nav_metadata,
        )
        subgoals.insert(0, nav_subgoal)
        insertion_offset = 1
    if inject_login_chain:
        login_actions = _build_login_action_chain()
        login_metadata = {
            "plan_id": plan.get("plan_id"),
            "bucket": "login",
            "estimated_step_count": len(login_actions),
            "durability_score": float(_DURABILITY_SCORES["high"]),
            "bootstrap_actions": login_actions,
            "primary_url": mission_url,
        }
        login_description = "01. login: autofill credentials"
        login_subgoal = MissionSubgoal(
            id=f"{mission_spec.id}_sg_login",
            description=login_description,
            planner_metadata=login_metadata,
        )
        subgoals.insert(insertion_offset, login_subgoal)
    if settings and settings.get("demo"):
        subgoals = [sub for sub in subgoals if "dom_presence_check" not in sub.description.lower()]
    return subgoals


def _scrub_constraints(constraints: Dict[str, Any], allow_sensitive: bool) -> Dict[str, Any]:
    if allow_sensitive:
        return constraints
    return {key: _scrub_value(value, parent_key=key) for key, value in constraints.items()}


def _scrub_value(value: Any, parent_key: str | None = None) -> Any:
    if isinstance(value, dict):
        return {k: _scrub_value(v, parent_key=k) for k, v in value.items()}
    if isinstance(value, list):
        return [_scrub_value(item) for item in value]
    if isinstance(value, str):
        if parent_key and parent_key.lower() in _SENSITIVE_KEYS:
            return "[REDACTED]"
        lowered = value.lower()
        if any(token in lowered for token in _SENSITIVE_KEYS):
            return "[REDACTED]"
    return value


def _score_actions(actions: Iterable[Dict[str, Any]], durability_summary: Dict[str, Any]) -> float:
    scores: List[int] = []
    for action in actions:
        durability = action.get("durability") or durability_summary.get(action.get("action"), "medium")
        scores.append(_DURABILITY_SCORES.get(str(durability).lower(), 1))
    if not scores:
        return float(_DURABILITY_SCORES["medium"])
    return round(sum(scores) / len(scores), 2)


def _describe_task(task: Dict[str, Any], idx: int) -> str:
    bucket = task.get("bucket") or "task"
    first_action = (task.get("inputs", {}).get("actions", []) or [{}])[0]
    action_name = first_action.get("action", "step")
    return f"{idx:02d}. {bucket}: {action_name}".strip()


def _extract_mission_url(text: str) -> str | None:
    if not text:
        return None
    match = _MISSION_URL_PATTERN.search(text)
    if not match:
        return None
    candidate = match.group(0)
    return candidate.rstrip(".,)")


def _remove_redundant_navigation_tasks(tasks: List[Dict[str, Any]], mission_url: str) -> List[Dict[str, Any]]:
    filtered: List[Dict[str, Any]] = []
    navigation_removed = False
    for task in tasks:
        actions = task.get("inputs", {}).get("actions", []) or []
        primary = actions[0] if actions else {}
        if (
            not navigation_removed
            and primary.get("action") == "navigate"
        ):
            url = str(primary.get("url") or "")
            if not url or url == mission_url or "example.com" in url:
                navigation_removed = True
                continue
        filtered.append(task)
    return filtered


def _rewrite_navigation_targets(tasks: List[Dict[str, Any]], target_url: str) -> List[Dict[str, Any]]:
    rewritten: List[Dict[str, Any]] = []
    for task in tasks:
        cloned = dict(task)
        inputs = dict(cloned.get("inputs") or {})
        actions = list(inputs.get("actions") or [])
        new_actions: List[Dict[str, Any]] = []
        for action in actions:
            updated = dict(action)
            if (updated.get("action") == "navigate"):
                url = str(updated.get("url") or "")
                if not url or "example.com" in url:
                    updated["url"] = target_url
            new_actions.append(updated)
        inputs["actions"] = new_actions
        cloned["inputs"] = inputs
        rewritten.append(cloned)
    return rewritten


def _strip_dom_presence_tasks(tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    stripped: List[Dict[str, Any]] = []
    for task in tasks:
        actions = task.get("inputs", {}).get("actions", []) or []
        primary = actions[0] if actions else {}
        if primary.get("action") == "dom_presence_check":
            continue
        stripped.append(task)
    return stripped


def _should_inject_login_chain(instruction: str, mission_url: str | None) -> bool:
    if not mission_url or not instruction:
        return False
    if not any(keyword in instruction.lower() for keyword in _LOGIN_KEYWORDS):
        return False
    parsed = urlparse(mission_url)
    path = parsed.path.lower()
    return "login" in path or "signin" in path


def _build_login_action_chain() -> List[Dict[str, Any]]:
    return [
        {"action": "fill", "selector": "#username", "text": _LOGIN_USERNAME},
        {"action": "fill", "selector": "#password", "text": _LOGIN_PASSWORD},
        {"action": "click", "selector": "button#submit"},
        {"action": "wait_for_navigation", "timeout": 8000},
        {"action": "screenshot", "name": "secure_area.png"},
    ]


def _has_dom_presence_task(tasks: List[Dict[str, Any]]) -> bool:
    for task in tasks:
        actions = task.get("inputs", {}).get("actions", []) or []
        for action in actions:
            if (action.get("action") or "").lower() == "dom_presence_check":
                return True
    return False


def _override_demo_navigation_url(
    mission_url: Optional[str],
    mission_spec: MissionSpec,
    settings: Optional[Dict[str, Any]],
) -> Optional[str]:
    if not settings or not settings.get("demo"):
        return mission_url
    if _DEMO_LOGIN_HTML.exists():
        demo_url = _DEMO_LOGIN_HTML.resolve().as_uri()
        if mission_url and mission_url != demo_url:
            print(
                "[DEMO] Overriding mission URL for demo mode",
                "->",
                demo_url,
            )
        return demo_url
    return mission_url


def _build_default_search_url(text: str) -> str:
    query = quote_plus(text.strip() or "web search")
    return f"{_DEFAULT_SEARCH_ENGINE}{query}"


__all__ = ["plan_mission", "MissionPlanningError"]
