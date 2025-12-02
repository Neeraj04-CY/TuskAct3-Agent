"""Helper to publish curated demo artifacts.

Usage:
    python scripts/generate_demo_assets.py --run runs/2024-05-05T12-00-00 --slug heroku_sample

The script copies screenshots/DOM/results into docs/artifacts/<slug>/ and emits a
plan_summary.json that GitHub Pages can serve.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable

ROOT = Path(__file__).resolve().parents[1]
DOCS_ARTIFACTS = ROOT / "docs" / "artifacts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy run artifacts into docs/")
    parser.add_argument("--run", required=True, help="Path to the completed run folder")
    parser.add_argument("--slug", required=True, help="Destination slug under docs/artifacts")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite destination if exists")
    parser.add_argument("--gif", action="store_true", help="Attempt to build an animated gif if ffmpeg is available")
    return parser.parse_args()


def ensure_destination(slug: str, overwrite: bool) -> Path:
    destination = DOCS_ARTIFACTS / slug
    if destination.exists() and overwrite:
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def copy_known_artifacts(run_path: Path, destination: Path) -> Iterable[Path]:
    copied = []
    mapping = {
        "result.json": run_path / "result.json",
        "dom.html": run_path / "dom.html",
    }
    screenshots_dir = run_path / "screenshots"
    if screenshots_dir.exists():
        for image in screenshots_dir.iterdir():
            mapping[image.name] = image
    for name, source in mapping.items():
        if source.exists():
            target = destination / name
            shutil.copy2(source, target)
            copied.append(target)
    return copied


def load_plan_metadata(run_path: Path) -> Dict[str, Any]:
    candidates = ["plan.json", "plan_v3.json", "planner.json", "result.json"]
    for candidate in candidates:
        file_path = run_path / candidate
        if not file_path.exists():
            continue
        try:
            with file_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if "plan_id" in data:
                tasks = data.get("tasks") or data.get("plan", {}).get("tasks") or []
                return {
                    "plan_id": data.get("plan_id"),
                    "goal": data.get("goal", ""),
                    "tasks_count": len(tasks),
                    "score": data.get("meta", {}).get("score"),
                }
            if "steps" in data:
                return {
                    "plan_id": data.get("plan_id") or "result-json",
                    "goal": data.get("goal") or "",
                    "tasks_count": len(data.get("steps", [])),
                    "score": data.get("completion", {}).get("success"),
                }
        except json.JSONDecodeError:
            continue
    return {"plan_id": "unknown", "goal": "", "tasks_count": 0, "score": None}


def maybe_generate_gif(destination: Path, enable: bool) -> None:
    if not enable:
        return
    if shutil.which("ffmpeg") is None:
        return
    images = sorted(destination.glob("*.png"))
    if len(images) < 2:
        return
    gif_path = destination / "timelapse.gif"
    command = [
        "ffmpeg",
        "-y",
        "-pattern_type",
        "glob",
        "-i",
        str(destination / "*.png"),
        "-vf",
        "fps=1",
        str(gif_path),
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, FileNotFoundError):
        if gif_path.exists():
            gif_path.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    run_path = Path(args.run).resolve()
    if not run_path.exists():
        raise FileNotFoundError(f"Run folder not found: {run_path}")
    destination = ensure_destination(args.slug, args.overwrite)
    copy_known_artifacts(run_path, destination)
    summary = load_plan_metadata(run_path)
    summary_path = destination / "plan_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    maybe_generate_gif(destination, args.gif)
    print(f"Artifacts copied to {destination}")


if __name__ == "__main__":
    main()
