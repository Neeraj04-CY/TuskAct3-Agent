from __future__ import annotations

import json
from pathlib import Path

from build_release import ReleaseBuilder


def _make_demo_payload(tmp_path: Path) -> dict:
    run_dir = tmp_path / "autonomy" / "run_seed"
    run_dir.mkdir(parents=True, exist_ok=True)
    screenshot = run_dir / "shot.png"
    screenshot.write_bytes(b"fake")
    summary = {
        "goal": "demo",
        "completed": True,
        "reason": "ok",
        "reward_trace": [{"step_id": "s1", "reward": 0.6, "confidence": {"confidence": 0.8}}],
    }
    stability = {
        "metrics": {"avg_reward": 0.6, "avg_confidence": 0.8, "repeated_failures": {}},
        "history_snapshot": [],
    }
    result = {
        "run_context": {
            "reward_trace": summary["reward_trace"],
        },
        "steps": [
            {
                "step": {"step_id": "s1", "action": "click"},
                "result": {"screenshot_path": str(screenshot)},
            }
        ],
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (run_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (run_dir / "stability_report.json").write_text(json.dumps(stability, indent=2), encoding="utf-8")
    (run_dir / "stability_report.md").write_text("stability", encoding="utf-8")
    return {
        "summary": summary,
        "result": result,
        "run_dir": str(run_dir),
        "summary_path": str(run_dir / "summary.json"),
    }


class _FakeRollout:
    def __init__(self, *, goal: str, episodes: int, output_root: Path, **_: object) -> None:
        self.output_root = Path(output_root)
        self.episodes = episodes

    def run(self) -> dict:
        self.output_root.mkdir(parents=True, exist_ok=True)
        summary = {"episodes": self.episodes, "successes": self.episodes - 1}
        (self.output_root / "rollout_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary


def test_release_builder_creates_bundle(tmp_path: Path) -> None:
    payload = _make_demo_payload(tmp_path)

    def demo_runner(*_: object, **__: object) -> dict:
        return payload

    builder = ReleaseBuilder(
        goal="demo",
        rollout_episodes=2,
        output_root=tmp_path / "release_bundle",
        demo_runner=demo_runner,
        rollout_factory=_FakeRollout,
    )
    bundle_path = builder.build()

    assert (bundle_path / "demo.gif").exists()
    assert (bundle_path / "charts.html").exists()
    assert (bundle_path / "autonomy" / "summary.json").exists()
    assert (bundle_path / "rollouts" / "rollout_summary.json").exists()
    assert (bundle_path / "docs" / "README.pdf").exists()
    assert (bundle_path / "dashboard_snapshot.json").exists()
    manifest = json.loads((bundle_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["rollout"]["episodes"] == 2
