from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

from eikon_engine.replay.experience_replay import ExperienceReplayEngine


def run_offline_eval(
    *,
    artifact_root: str | Path = "artifacts",
    output_dir: Optional[str | Path] = None,
    save_hints: bool = False,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    engine = ExperienceReplayEngine(artifact_root)
    curriculum = engine.build_curriculum(limit=limit)
    replay_result = engine.replay_curriculum(curriculum, output_dir=Path(output_dir) if output_dir else None)
    summary = replay_result["summary"]
    out_dir = replay_result["output_dir"]
    report = {
        "artifact_root": str(artifact_root),
        "curriculum_size": len(curriculum),
        "states_processed": summary.get("states_processed", 0),
        "batches": summary.get("batches", []),
        "memory_summary": engine.strategist.agent_memory.summarize_experience(),
    }
    json_path = out_dir / "improvement_report.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path = out_dir / "improvement_report.md"
    md_path.write_text(_render_markdown(report), encoding="utf-8")
    hints_path = engine.save_memory_hints(out_dir) if save_hints else None
    return {
        "summary": report,
        "json_path": json_path,
        "md_path": md_path,
        "hints_path": hints_path,
    }


def _render_markdown(report: Dict[str, Any]) -> str:
    lines = ["# Offline Improvement Report", "", f"- Artifact root: `{report['artifact_root']}`", f"- Curriculum batches: {report['curriculum_size']}", f"- States processed: {report['states_processed']}"]
    memory = report.get("memory_summary") or {}
    lines.append(f"- Memory entries: {memory.get('entries', 0)}")
    lines.append("")
    lines.append("## Batch Effects")
    for batch in report.get("batches", []):
        lines.append(f"- **{batch.get('tag', 'batch')}**: {batch.get('states', 0)} states, reason: {batch.get('reason')}")
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run offline replay and improvement evaluation.")
    parser.add_argument("--artifacts", default="artifacts", help="Root directory containing historical runs")
    parser.add_argument("--output", default=None, help="Directory for replay outputs")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on number of runs to load")
    parser.add_argument("--save-hints", action="store_true", help="Persist agent memory hints to disk")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_offline_eval(
        artifact_root=args.artifacts,
        output_dir=args.output,
        save_hints=args.save_hints,
        limit=args.limit,
    )
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
