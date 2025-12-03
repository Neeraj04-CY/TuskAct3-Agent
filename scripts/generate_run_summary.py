"""Compile a reviewer-friendly summary (Markdown + PDF) for a given run folder."""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Any, Dict, List

try:  # Optional: PDF generation
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
except Exception:  # pragma: no cover - runtime fallback
    letter = None
    canvas = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create run_summary.md/pdf artifacts from a run folder.")
    parser.add_argument("--run", required=True, help="Path to the run directory (e.g. docs/heroku_sample)")
    parser.add_argument("--title", default="Autonomy Demo", help="Title used in the generated summary")
    parser.add_argument("--md", default=None, help="Optional explicit Markdown output path")
    parser.add_argument("--pdf", default=None, help="Optional explicit PDF output path")
    return parser.parse_args()


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    if not path.exists():
        return entries
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


def _collect_screenshots(run_dir: Path) -> List[Path]:
    shots: List[Path] = []
    for step_dir in sorted(run_dir.glob("step_*")):
        if not step_dir.is_dir():
            continue
        shots.extend(sorted(step_dir.glob("*.png")))
    return shots


def _behavior_summary(summary: Dict[str, Any]) -> str:
    preds = summary.get("behavior_predictions") or []
    if not preds:
        return "No behavior predictions recorded"
    latest = preds[-1]
    difficulty = latest.get("difficulty")
    diff_display = f"{difficulty:.2f}" if isinstance(difficulty, (int, float)) else "n/a"
    bias = latest.get("selector_bias") or "unknown"
    likely = "yes" if latest.get("likely_repair") else "no"
    subgoals = latest.get("recommended_subgoals") or []
    return f"difficulty={diff_display}, likely_repair={likely}, selector_bias={bias}, subgoals={', '.join(subgoals) or 'n/a'}"


def build_markdown(run_dir: Path, title: str) -> str:
    summary_json = _read_json(run_dir / "run_summary.json")
    trace_entries = _read_jsonl(run_dir / "trace.jsonl")
    steps_json = _read_jsonl(run_dir / "steps.jsonl")
    screenshots = _collect_screenshots(run_dir)
    goal = summary_json.get("goal") or "(unknown goal)"
    completion = summary_json.get("completion", {})
    completed = completion.get("complete")
    reason = completion.get("reason", "")
    duration = summary_json.get("duration_seconds")
    repairs = summary_json.get("repair_events") or []
    planner_events = summary_json.get("interventions") or []
    behavior_line = _behavior_summary(summary_json)

    lines: List[str] = []
    lines.append(f"# Run Summary — {title}")
    lines.append("")
    lines.append(f"- **Run directory**: `{run_dir}`")
    lines.append(f"- **Goal**: {goal}")
    if duration is not None:
        lines.append(f"- **Duration**: {duration:.2f} seconds")
    lines.append(f"- **Completed**: {'yes' if completed else 'no'} ({reason or 'no reason recorded'})")
    lines.append(f"- **Screenshots captured**: {len(screenshots)}")
    lines.append(f"- **Repairs attempted**: {len(repairs)}")
    lines.append(f"- **Behavior prediction**: {behavior_line}")
    lines.append("")

    lines.append("## Timeline excerpt")
    lines.append("| Step | Action | URL / Notes |")
    lines.append("| --- | --- | --- |")
    if trace_entries:
        for entry in trace_entries[:10]:
            action = entry.get("action") or "n/a"
            url = entry.get("url") or ""
            completion_state = entry.get("completion") or {}
            note = completion_state.get("status") or completion_state.get("info") or url or ""
            lines.append(f"| {entry.get('step_index')} | {action} | {note} |")
    else:
        lines.append("| 0 | no trace | log not available |")
    lines.append("")

    lines.append("## Planner highlights")
    if planner_events:
        for evt in planner_events[:5]:
            lines.append(f"- {evt.get('event') or 'event'} → {evt.get('reason') or evt}")
    else:
        lines.append("- no adaptive planner events captured in this run")
    lines.append("")

    lines.append("## Repairs + guardrails")
    if repairs:
        for idx, evt in enumerate(repairs[:5], start=1):
            patch = evt.get("patch", {})
            reason = patch.get("reason") or evt.get("reason") or "unknown"
            lines.append(f"- Repair {idx}: {patch.get('type', 'patch')} — {reason}")
    else:
        lines.append("- no repairs recorded")
    lines.append("")

    lines.append("## Evidence to inspect")
    if screenshots:
        for shot in screenshots[:5]:
            rel = shot.relative_to(run_dir)
            lines.append(f"- `{rel}`")
    else:
        lines.append("- No screenshots found; run may have been dry-run only")
    dom_paths = list(run_dir.glob("step_*/dom*.html"))
    if dom_paths:
        rel = dom_paths[0].relative_to(run_dir)
        lines.append(f"- DOM snapshot: `{rel}`")
    steps_file = run_dir / "steps.jsonl"
    if steps_file.exists():
        lines.append(f"- Planner trace: `{steps_file.name}`")
    traces_file = run_dir / "trace.jsonl"
    if traces_file.exists():
        lines.append(f"- Worker trace: `{traces_file.name}`")
    lines.append("")

    lines.append("## Step metadata sample")
    if steps_json:
        for entry in steps_json[:5]:
            meta = entry.get("metadata", {})
            action = meta.get("action") or meta.get("action_payload", {}).get("action")
            lines.append(f"- idx {entry.get('idx')}: {action or 'unknown action'}")
    else:
        lines.append("- steps.jsonl missing — Strategist logs were not emitted for this run")
    lines.append("")

    lines.append("_Generated by scripts/generate_run_summary.py_")
    return "\n".join(lines)


def _write_pdf(text: str, path: Path) -> None:
    if not canvas or not letter:
        print("reportlab not available; skipping PDF generation")
        return
    pdf = canvas.Canvas(str(path), pagesize=letter)
    width, height = letter
    y = height - 72
    for paragraph in text.splitlines():
        if not paragraph:
            y -= 14
            continue
        wrapped = textwrap.wrap(paragraph, width=90) or [""]
        for line in wrapped:
            if y < 72:
                pdf.showPage()
                y = height - 72
            pdf.drawString(72, y, line)
            y -= 14
    pdf.save()
    print(f"📝 Wrote PDF summary -> {path}")


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run).resolve()
    if not run_dir.exists():
        raise SystemExit(f"Run directory not found: {run_dir}")
    markdown = build_markdown(run_dir, args.title)
    md_path = Path(args.md) if args.md else run_dir / "run_summary.md"
    md_path.write_text(markdown, encoding="utf-8")
    print(f"🗒️  Wrote Markdown summary -> {md_path}")
    pdf_path = Path(args.pdf) if args.pdf else run_dir / "run_summary.pdf"
    _write_pdf(markdown, pdf_path)


if __name__ == "__main__":  # pragma: no cover
    main()
