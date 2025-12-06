"""Integration tests for demo asset helper scripts."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_RUN = PROJECT_ROOT / "docs" / "artifacts" / "heroku_sample"


def _copy_sample_run(tmp_path: Path) -> Path:
    destination = tmp_path / "heroku_sample_copy"
    shutil.copytree(SAMPLE_RUN, destination)
    return destination


def _run_script(args: list[str]) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(PROJECT_ROOT))
    result = subprocess.run(
        [sys.executable, *args],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return result


def test_generate_run_summary_outputs_markdown_and_pdf(tmp_path: Path) -> None:
    run_dir = _copy_sample_run(tmp_path)
    md_path = tmp_path / "summary.md"
    pdf_path = tmp_path / "summary.pdf"

    _run_script(
        [
            "scripts/generate_run_summary.py",
            "--run",
            str(run_dir),
            "--title",
            "Integration Sample",
            "--md",
            str(md_path),
            "--pdf",
            str(pdf_path),
        ]
    )

    assert md_path.exists(), "Markdown summary should be generated"
    assert pdf_path.exists(), "PDF summary should be generated"
    text = md_path.read_text(encoding="utf-8")
    assert "Integration Sample" in text
    assert "Run Summary" in text


def test_make_demo_gif_emits_animation(tmp_path: Path) -> None:
    run_dir = _copy_sample_run(tmp_path)
    gif_path = tmp_path / "demo.gif"

    _run_script(
        [
            "scripts/make_demo_gif.py",
            "--run",
            str(run_dir),
            "--output",
            str(gif_path),
            "--fps",
            "1.5",
        ]
    )

    assert gif_path.exists(), "GIF should be created"
    assert gif_path.stat().st_size > 0, "GIF file should not be empty"
