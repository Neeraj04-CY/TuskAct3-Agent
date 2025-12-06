from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import imageio.v3 as iio
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

from dashboard.data_loader import load_dashboard_payload
from run_autonomy_demo import run_single_demo
from run_rollout import RolloutEngine


DemoRunner = Callable[..., Dict[str, Any]]
RolloutFactory = Callable[..., RolloutEngine]


@dataclass
class ReleaseBuilder:
    goal: str
    allow_sensitive: bool = False
    execute: bool = False
    rollout_episodes: int = 0
    output_root: Path = Path("release_bundle")
    demo_runner: DemoRunner = run_single_demo
    rollout_factory: RolloutFactory = RolloutEngine
    timestamp: str = field(default_factory=lambda: datetime.utcnow().strftime("%Y%m%d_%H%M%S"))

    def build(self) -> Path:
        payload = self.demo_runner(
            self.goal,
            execute=self.execute,
            allow_sensitive=self.allow_sensitive,
        )
        release_dir = self.output_root / f"release_{self.timestamp}"
        release_dir.mkdir(parents=True, exist_ok=True)
        self._copy_run(Path(payload["run_dir"]), release_dir / "autonomy")

        rollout_summary: Optional[Dict[str, Any]] = None
        if self.rollout_episodes > 0:
            rollout_engine = self.rollout_factory(
                goal=self.goal,
                episodes=self.rollout_episodes,
                allow_sensitive=self.allow_sensitive,
                execute=self.execute,
                output_root=release_dir / "rollouts",
            )
            rollout_summary = rollout_engine.run()

        self._generate_demo_gif(payload["result"], release_dir / "demo.gif")
        self._write_charts(payload["summary"], release_dir / "charts.html")
        self._export_docs(release_dir)
        self._write_dashboard_snapshot(release_dir)
        self._write_manifest(release_dir, payload, rollout_summary)
        return release_dir

    # ------------------------------------------------------------------
    def _copy_run(self, source: Path, destination: Path) -> None:
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source, destination)

    def _generate_demo_gif(self, result: Dict[str, Any], output_path: Path) -> None:
        frames = []
        for entry in result.get("steps", []) or []:
            step_result = entry.get("result", {}) if isinstance(entry, dict) else {}
            screenshot = step_result.get("screenshot_path") or step_result.get("failure_screenshot_path")
            if screenshot and Path(screenshot).exists():
                try:
                    frames.append(iio.imread(screenshot))
                except Exception:
                    continue
        if not frames:
            placeholder = np.full((180, 320, 3), (15, 23, 42), dtype=np.uint8)
            frames = [placeholder]
        iio.imwrite(output_path, frames, duration=0.45, loop=0)

    def _write_charts(self, summary: Dict[str, Any], output_path: Path) -> None:
        reward_trace = summary.get("reward_trace", []) or []
        rewards = [entry.get("reward", 0.0) for entry in reward_trace]
        confidence = [
            (entry.get("confidence", {}) or {}).get("confidence", 0.0)
            for entry in reward_trace
        ]
        html = f"""
<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"UTF-8\" />
  <title>Release Charts</title>
  <script src=\"https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js\"></script>
  <style>
    body {{ font-family: Arial, sans-serif; background: #0f172a; color: #e2e8f0; }}
    canvas {{ max-width: 960px; margin: 24px auto; display: block; background: #1e293b; padding: 16px; border-radius: 12px; }}
  </style>
</head>
<body>
  <h1>Autonomy Demo Metrics</h1>
  <canvas id=\"chart\"></canvas>
  <script>
    const data = {{
      labels: {json.dumps([entry.get('step_id', 'step') for entry in reward_trace])},
      rewards: {json.dumps(rewards)},
      confidence: {json.dumps(confidence)}
    }};
    const ctx = document.getElementById('chart').getContext('2d');
    new Chart(ctx, {{
      type: 'line',
      data: {{
        labels: data.labels,
        datasets: [
          {{ label: 'Reward', data: data.rewards, borderColor: '#38bdf8', tension: 0.3 }},
          {{ label: 'Confidence', data: data.confidence, borderColor: '#fbbf24', tension: 0.3, borderDash: [6,6] }}
        ]
      }},
      options: {{ plugins: {{ legend: {{ labels: {{ color: '#cbd5f5' }} }} }}, scales: {{ x: {{ ticks: {{ color: '#94a3b8' }} }}, y: {{ ticks: {{ color: '#94a3b8' }} }} }} }}
    }});
  </script>
</body>
</html>
"""
        output_path.write_text(html, encoding="utf-8")

    def _export_docs(self, release_dir: Path) -> None:
        docs_dir = release_dir / "docs"
        docs_dir.mkdir(parents=True, exist_ok=True)
        repo_root = Path(__file__).parent
        readme = repo_root / "README.md"
        pitch = repo_root / "docs" / "yc_pitch.md"
        shutil.copy2(readme, docs_dir / "README.md")
        shutil.copy2(pitch, docs_dir / "yc_pitch.md")
        self._markdown_to_pdf(readme, docs_dir / "README.pdf")
        self._markdown_to_pdf(pitch, docs_dir / "yc_pitch.pdf")

    def _write_dashboard_snapshot(self, release_dir: Path) -> None:
        snapshot = load_dashboard_payload()
        (release_dir / "dashboard_snapshot.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")

    def _write_manifest(
        self,
        release_dir: Path,
        payload: Dict[str, Any],
        rollout_summary: Optional[Dict[str, Any]],
    ) -> None:
        manifest = {
            "goal": self.goal,
            "timestamp": self.timestamp,
            "success": bool(payload["summary"].get("completed")),
            "reason": payload["summary"].get("reason"),
            "run_dir": payload["run_dir"],
            "rollout": rollout_summary,
        }
        (release_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def _markdown_to_pdf(self, source: Path, destination: Path) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        text = source.read_text(encoding="utf-8")
        pdf = canvas.Canvas(str(destination), pagesize=letter)
        width, height = letter
        y = height - 72
        for line in text.splitlines():
            if y <= 72:
                pdf.showPage()
                y = height - 72
            pdf.drawString(72, y, line[:110])
            y -= 14
        pdf.save()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a reviewer-ready release bundle.")
    parser.add_argument("goal", nargs="?", default="Showcase the autonomy demo")
    parser.add_argument("--rollout", type=int, default=0, help="Optional rollout episode count")
    parser.add_argument("--execute", action="store_true", help="Use live Playwright for the demo")
    parser.add_argument("--allow-sensitive", action="store_true", help="Allow sensitive resources")
    args = parser.parse_args()

    builder = ReleaseBuilder(
        goal=args.goal,
        rollout_episodes=args.rollout,
        execute=args.execute,
        allow_sensitive=args.allow_sensitive,
    )
    release_dir = builder.build()
    print(f"Release bundle created at {release_dir}")


if __name__ == "__main__":
    main()
