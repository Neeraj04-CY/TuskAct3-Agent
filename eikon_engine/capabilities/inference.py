from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from .models import CapabilityId, CapabilityRequirement, PlanCapabilityReport
from .registry import CAPABILITY_REGISTRY, Capability

SUBGOAL_CAPABILITY_MAP: Dict[str, List[CapabilityId]] = {
    "navigation": ["web_navigation"],
    "form": ["credential_entry"],
    "listing_extraction": ["data_extraction"],
    "save_file": ["artifact_persistence"],
}

_DEFAULT_CONFIDENCE = 0.9


def _normalize_bucket(bucket: str | None) -> str:
    return (bucket or "").strip().lower() or "misc"


def _dedupe_by_capability(requirements: Iterable[CapabilityRequirement]) -> List[CapabilityRequirement]:
    seen: Dict[str, CapabilityRequirement] = {}
    for req in requirements:
        if req.capability_id not in seen or req.confidence > seen[req.capability_id].confidence:
            seen[req.capability_id] = req
    return list(seen.values())


def infer_capability_requirements(bucket: str | None) -> List[CapabilityRequirement]:
    bucket_label = _normalize_bucket(bucket)
    capability_ids = SUBGOAL_CAPABILITY_MAP.get(bucket_label, [])
    requirements: List[CapabilityRequirement] = []
    for capability_id in capability_ids:
        requirements.append(
            CapabilityRequirement(
                capability_id=capability_id,
                required=True,
                confidence=_DEFAULT_CONFIDENCE,
                reason=f"bucket:{bucket_label}",
            )
        )
    return requirements


def _is_missing(capability_id: str, registry: Mapping[str, Capability]) -> bool:
    return capability_id not in registry


def build_plan_capability_report(
    requirements: Sequence[CapabilityRequirement],
    *,
    registry: Mapping[str, Capability] | None = None,
) -> PlanCapabilityReport:
    available = registry or CAPABILITY_REGISTRY
    deduped = _dedupe_by_capability(requirements)
    missing: List[CapabilityRequirement] = []
    present: List[CapabilityRequirement] = []
    optional: List[CapabilityRequirement] = []
    for req in deduped:
        target = optional if not req.required else present
        target.append(req)
        if _is_missing(req.capability_id, available):
            missing.append(req)
    has_required_missing = any(req.required for req in missing)
    has_optional_missing = any(not req.required for req in missing)
    if has_required_missing:
        risk_level = "high"
    elif has_optional_missing:
        risk_level = "medium"
    else:
        risk_level = "low"
    return PlanCapabilityReport(
        required=[req for req in present if req.required],
        optional=optional,
        missing=missing,
        risk_level=risk_level,
    )


def plan_capability_report_for_tasks(
    tasks: Sequence[Mapping[str, Any]],
    *,
    registry: Mapping[str, Capability] | None = None,
) -> Tuple[PlanCapabilityReport, Dict[str, List[CapabilityRequirement]]]:
    requirements_by_task: Dict[str, List[CapabilityRequirement]] = {}
    aggregate_requirements: List[CapabilityRequirement] = []
    for idx, task in enumerate(tasks):
        bucket = task.get("bucket") if isinstance(task, Mapping) else None
        task_id = str(task.get("id", f"task_{idx+1}")) if isinstance(task, Mapping) else f"task_{idx+1}"
        reqs = infer_capability_requirements(bucket)
        requirements_by_task[task_id] = reqs
        aggregate_requirements.extend(reqs)
    report = build_plan_capability_report(aggregate_requirements, registry=registry)
    return report, requirements_by_task


def requirements_to_payload(requirements: Iterable[CapabilityRequirement]) -> List[Dict[str, Any]]:
    return [asdict(req) for req in requirements]


def report_to_payload(report: PlanCapabilityReport) -> Dict[str, Any]:
    return {
        "required": requirements_to_payload(report.required),
        "missing": requirements_to_payload(report.missing),
        "optional": requirements_to_payload(report.optional),
        "risk_level": report.risk_level,
    }


def requirements_from_payload(payloads: Iterable[Mapping[str, Any]]) -> List[CapabilityRequirement]:
    requirements: List[CapabilityRequirement] = []
    for payload in payloads:
        capability_id = str(payload.get("capability_id", "")).strip()
        if not capability_id:
            continue
        required = bool(payload.get("required", False))
        confidence = float(payload.get("confidence", _DEFAULT_CONFIDENCE))
        reason = str(payload.get("reason") or "payload")
        source = payload.get("source")
        requirements.append(
            CapabilityRequirement(
                capability_id=capability_id,
                required=required,
                confidence=confidence,
                reason=reason,
                source=source if source is None or isinstance(source, str) else str(source),
            )
        )
    return requirements
