from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from eikon_engine.config_loader import load_settings
from eikon_engine.missions.mission_executor import MissionExecutor, run_mission_sync
from eikon_engine.missions.mission_schema import MissionResult, MissionSpec, MissionSubgoalResult, mission_id


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a mission specification through MissionExecutor.")
    parser.add_argument("mission", help="Path to the mission file (JSON or YAML)")
    parser.add_argument("--allow-sensitive", action="store_true", help="Permit workflows that need sensitive data")
    parser.add_argument("--execute", action="store_true", help="Force live Playwright execution even if the mission defaults to dry-run")
    parser.add_argument("--dry-run", action="store_true", help="Force dry-run even if the mission requests execute")
    parser.add_argument("--summary", default=None, help="Optional path to copy the mission summary JSON")
    parser.add_argument("--artifacts-root", default="artifacts", help="Root directory for mission artifacts")
    parser.add_argument("--capability-threshold", type=float, default=None, help="Override capability enforcement threshold (0-1 range)")
    parser.add_argument("--capability-critical", type=float, default=None, help="Override capability critical threshold (0-1 range)")
    parser.add_argument(
        "--auto-approve-capabilities",
        action="store_true",
        help="Auto-approve capability gaps instead of requesting human review",
    )
    parser.add_argument(
        "--fail-on-missing-capability",
        action="store_true",
        help="Escalate when a required capability is missing in the registry",
    )
    parser.add_argument("--require-approval", action="store_true", help="Force human approval before execution")
    parser.add_argument("--approval-timeout", type=int, default=None, help="Seconds to wait for approval before expiring")
    parser.add_argument("--auto-approve-low-risk", action="store_true", help="Auto-approve approval requests marked low risk")
    return parser.parse_args()


def _load_mission(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise SystemExit(
                "PyYAML is required to load YAML missions. Install it or provide JSON instead."
            ) from exc
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise SystemExit("Mission file must define a JSON/YAML object.")
    return data


def _stringify_sequence(values: Iterable[Any]) -> str:
    entries: List[str] = []
    for raw in values:
        item = str(raw).strip()
        if item:
            entries.append(item)
    return "; ".join(entries)


def _clean_reason(value: Any) -> str:
    if value is None:
        return ""
    collapsed = " ".join(str(value).split())
    lowered = collapsed.lower()
    marker = "for more information"
    if marker in lowered:
        collapsed = collapsed[: lowered.index(marker)].rstrip(" ;:,-")
    return collapsed


def _resolve_instruction(mission: Dict[str, Any]) -> str:
    instruction = str(mission.get("instruction", "")).strip()
    if instruction:
        return instruction
    objective = str(mission.get("objective", "")).strip()
    if not objective:
        raise SystemExit("Mission file must include either 'instruction' or 'objective'.")
    segments: List[str] = [objective]
    constraints = mission.get("constraints")
    constraint_rules: Iterable[Any] | None = None
    if isinstance(constraints, dict):
        candidate = constraints.get("rules")
        if isinstance(candidate, Iterable) and not isinstance(candidate, (str, bytes)):
            constraint_rules = candidate
    elif isinstance(constraints, Iterable) and not isinstance(constraints, (str, bytes)):
        constraint_rules = constraints
    if constraint_rules is not None:
        constraint_text = _stringify_sequence(constraint_rules)
        if constraint_text:
            segments.append(f"Constraints: {constraint_text}")
    environment = mission.get("environment")
    env_bits: List[str] = []
    if isinstance(environment, dict):
        for key in ("type", "entry_url", "notes"):
            value = str(environment.get(key, "")).strip()
            if value:
                env_bits.append(f"{key}: {value}")
    elif isinstance(environment, (str, bytes)):
        env_bits.append(str(environment).strip())
    if env_bits:
        segments.append("Environment: " + " | ".join(env_bits))
    context = mission.get("context")
    if isinstance(context, (str, bytes)):
        ctx = str(context).strip()
        if ctx:
            segments.append(ctx)
    return " ".join(segments)


def _normalize_constraints(mission: Dict[str, Any]) -> Dict[str, Any] | None:
    payload: Dict[str, Any] = {}
    constraints = mission.get("constraints")
    if isinstance(constraints, dict):
        for key, value in constraints.items():
            if value is not None:
                payload[key] = value
    elif isinstance(constraints, Iterable) and not isinstance(constraints, (str, bytes)):
        text = [str(item).strip() for item in constraints if str(item).strip()]
        if text:
            payload["rules"] = text
    environment = mission.get("environment")
    if isinstance(environment, dict):
        env_payload = {k: v for k, v in environment.items() if isinstance(v, str) and v.strip()}
        if env_payload:
            payload["environment"] = env_payload
    return payload or None


def _clamp(value: Any, *, minimum: int, maximum: int, default: int) -> int:
    try:
        integer = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, integer))


def _determine_bool(*values: Any) -> bool:
    for value in values:
        if isinstance(value, bool):
            if value:
                return True
        elif isinstance(value, str):
            if value.strip().lower() in {"1", "true", "yes"}:
                return True
        elif value:
            return True
    return False


def _build_executor(
    artifacts_root: Path,
    *,
    capability_overrides: Dict[str, Any] | None = None,
    approval_overrides: Dict[str, Any] | None = None,
) -> MissionExecutor:
    settings = load_settings()
    logging_cfg = settings.setdefault("logging", {})
    logging_cfg["artifact_root"] = str(artifacts_root)
    if capability_overrides:
        enforcement_cfg = settings.setdefault("capability_enforcement", {})
        for key, value in capability_overrides.items():
            if value is None:
                continue
            enforcement_cfg[key] = value
    if approval_overrides:
        approval_cfg = settings.setdefault("approval", {})
        for key, value in approval_overrides.items():
            if value is None:
                continue
            approval_cfg[key] = value
    return MissionExecutor(settings=settings, artifacts_root=artifacts_root)


def _collect_subgoal_events(results: List[MissionSubgoalResult]) -> Tuple[List[str], List[str]]:
    failures: List[str] = []
    recoveries: List[str] = []
    for subgoal in results:
        descriptor = subgoal.description or subgoal.subgoal_id
        reason = subgoal.error or ""
        if not reason and isinstance(subgoal.completion, dict):
            completion_reason = subgoal.completion.get("reason")
            if completion_reason:
                reason = str(completion_reason)
        reason = _clean_reason(reason)
        lowered_reason = reason.lower()
        recovered = False
        if lowered_reason and ("replan" in lowered_reason or "progress" in lowered_reason):
            recoveries.append(f"{descriptor} triggered recovery ({reason})")
            recovered = True
        if subgoal.attempts > 1:
            label = f"{descriptor} requested a retry"
            if reason:
                label += f" ({reason})"
            failures.append(label)
            if subgoal.status == "complete":
                recovery_line = f"{descriptor} recovered after {subgoal.attempts} attempt(s)"
                if reason:
                    recovery_line += f" ({reason})"
                recoveries.append(recovery_line)
                continue
            if not recovered and lowered_reason:
                recoveries.append(f"{descriptor} retried after failure ({reason})")
                recovered = True
        if subgoal.status != "complete":
            status_line = f"{descriptor} ended with status {subgoal.status}"
            if reason:
                status_line += f" ({reason})"
            failures.append(status_line)
            if not recovered and lowered_reason:
                recoveries.append(f"{descriptor} attempted recovery ({reason})")
    return failures, recoveries


def _build_summary_payload(
    *,
    mission_name: str,
    mission_path: Path,
    spec: MissionSpec,
    result: MissionResult,
    failures: List[str],
    recoveries: List[str],
) -> Dict[str, Any]:
    raw_summary = dict(result.summary or {})
    reason = _clean_reason(raw_summary.get("reason_summary") or raw_summary.get("reason"))
    trace_path = raw_summary.get("execution_trace")
    trace_summary_path = raw_summary.get("execution_trace_summary")
    trace_decisions = raw_summary.get("execution_trace_decisions")
    if result.status == "complete":
        display_status = "success"
    elif recoveries:
        display_status = "partial_success"
    else:
        display_status = "unrecoverable_failure"
    return {
        "mission_name": mission_name,
        "mission_path": str(mission_path),
        "mission_id": result.mission_id,
        "instruction": spec.instruction,
        "status": display_status,
        "engine_status": result.status,
        "reason": reason,
        "subgoal_count": len(result.subgoal_results),
        "failures": failures,
        "recoveries": recoveries,
        "artifact_dir": result.artifacts_path,
        "trace_path": trace_path,
        "trace_summary_path": trace_summary_path,
        "trace_decisions_path": trace_decisions,
        "trace_warnings": raw_summary.get("trace_warnings", []),
        "trace_incomplete": bool(raw_summary.get("trace_incomplete", False)),
        "raw_summary": raw_summary,
    }


def _write_summary(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _print_summary(summary: Dict[str, Any]) -> None:
    print("Mission Run Summary")
    print("===================")
    print(f"Mission: {summary['mission_name']} ({summary['mission_path']})")
    print(f"Mission ID: {summary['mission_id']}")
    print(f"Status: {summary['status']} ({summary['engine_status']})")
    reason = summary.get("reason") or "no reason provided"
    print(f"Reason: {reason}")
    failures = summary.get("failures") or []
    if failures:
        print("Failure detected:")
        for entry in failures:
            print(f" - {entry}")
    else:
        print("Failure detected: none")
    recoveries = summary.get("recoveries") or []
    if recoveries:
        print("Recovery applied:")
        for entry in recoveries:
            print(f" - {entry}")
    else:
        print("Recovery applied: not required")
    print(f"Execution continued across {summary['subgoal_count']} subgoal(s)")
    trace_summary = summary.get("trace_summary_path")
    if trace_summary:
        trace_dir = Path(trace_summary).parent
        print(f"Traces recorded in: {trace_dir}")
        print(f"Trace summary: {trace_summary}")
    trace_path = summary.get("trace_path")
    if trace_path:
        print(f"Trace bundle: {trace_path}")
    decisions_path = summary.get("trace_decisions_path")
    if decisions_path:
        print(f"Decision report: {decisions_path}")
    print(f"Artifacts stored in: {summary['artifact_dir']}")
    print(f"Final status resolved: {summary['status']}")


def main() -> int:
    args = _parse_args()
    mission_path = Path(args.mission).resolve()
    if not mission_path.exists():
        raise SystemExit(f"Mission file not found: {mission_path}")

    mission = _load_mission(mission_path)
    instruction = _resolve_instruction(mission)
    constraints = _normalize_constraints(mission)
    timeout_secs = _clamp(mission.get("timeout_secs"), minimum=60, maximum=21600, default=900)
    max_retries = _clamp(mission.get("max_retries"), minimum=0, maximum=5, default=2)
    allow_sensitive = _determine_bool(args.allow_sensitive, mission.get("allow_sensitive"))
    execute_requested = _determine_bool(args.execute, mission.get("execute"))
    execute_flag = False if args.dry_run else execute_requested
    autonomy_budget = mission.get("autonomy_budget")
    safety_contract = mission.get("safety_contract")
    ask_on_uncertainty = bool(mission.get("ask_on_uncertainty", False))
    prefix = str(mission.get("artifact_prefix") or mission.get("name") or mission_path.stem)
    mission_identifier = str(mission.get("mission_id") or mission_id(prefix))

    enforcement_overrides: Dict[str, Any] = {
        "threshold": args.capability_threshold,
        "critical": args.capability_critical,
    }
    if args.auto_approve_capabilities:
        enforcement_overrides["auto_approve_capabilities"] = True
    if args.fail_on_missing_capability:
        enforcement_overrides["fail_on_missing_capability"] = True

    approval_overrides: Dict[str, Any] = {
        "require_approval": args.require_approval,
        "timeout_secs": args.approval_timeout,
        "auto_approve_low_risk": args.auto_approve_low_risk,
    }

    spec = MissionSpec(
        id=mission_identifier,
        instruction=instruction,
        constraints=constraints,
        timeout_secs=timeout_secs,
        max_retries=max_retries,
        allow_sensitive=allow_sensitive,
        execute=execute_flag,
        autonomy_budget=autonomy_budget,
        safety_contract=safety_contract,
        ask_on_uncertainty=ask_on_uncertainty,
    )

    artifacts_root = Path(args.artifacts_root).resolve()
    executor = _build_executor(
        artifacts_root,
        capability_overrides=enforcement_overrides,
        approval_overrides=approval_overrides,
    )
    result = run_mission_sync(spec, executor=executor)
    mission_name = str(mission.get("name") or mission_path.stem)
    failures, recoveries = _collect_subgoal_events(result.subgoal_results)
    summary_payload = _build_summary_payload(
        mission_name=mission_name,
        mission_path=mission_path,
        spec=spec,
        result=result,
        failures=failures,
        recoveries=recoveries,
    )
    artifact_dir = Path(summary_payload["artifact_dir"])
    summary_path = artifact_dir / "mission_summary.json"
    _write_summary(summary_payload, summary_path)
    if args.summary:
        _write_summary(summary_payload, Path(args.summary))
    _print_summary(summary_payload)

    if summary_payload["status"] == "unrecoverable_failure":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
