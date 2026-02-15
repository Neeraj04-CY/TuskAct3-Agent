from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Optional

from .replay_engine import ReplayEngine, ReplaySummary


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean flag: {value}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deterministic replay for TuskAct3 mission traces.")
    parser.add_argument("--trace", required=True, help="Path to trace_*.json file produced by a mission run.")
    parser.add_argument(
        "--headless",
        default="true",
        type=_parse_bool,
        help="Whether to replay headless (default true). Use --headless false to show the browser.",
    )
    parser.add_argument(
        "--artifacts-dir",
        default=None,
        help="Optional directory to store replay artifacts. Defaults to replay_artifacts/<mission_id>.",
    )
    return parser


async def _run_async(args: argparse.Namespace) -> ReplaySummary:
    engine = ReplayEngine()
    trace_path = Path(args.trace)
    artifacts_dir: Optional[Path] = Path(args.artifacts_dir) if args.artifacts_dir else None
    print(f"[REPLAY] Loading trace from {trace_path}")
    summary = await engine.replay(trace_path, headless=args.headless, output_dir=artifacts_dir)
    print(f"[REPLAY] Mission {summary.mission_id} replay {summary.status}")
    if summary.summary_path:
        print(f"[REPLAY] Summary written to {summary.summary_path}")
    if summary.skill_details:
        print(f"[REPLAY] Skills: {summary.skill_details}")
    if summary.divergence:
        print(f"[REPLAY] Divergence details: {summary.divergence}")
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    summary = asyncio.run(_run_async(args))
    if summary.status != "success":
        sys.exit(1)


__all__ = ["main", "build_parser"]
