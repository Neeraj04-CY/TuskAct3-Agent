"""Generate a lightweight animated GIF from a run's screenshots."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import imageio.v2 as imageio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bundle step screenshots into a GIF for docs/README usage.")
    parser.add_argument("--run", required=True, help="Path to the run folder or docs sample containing screenshots.")
    parser.add_argument(
        "--output",
        default="docs/assets/demo.gif",
        help="Where to write the GIF (default: docs/assets/demo.gif).",
    )
    parser.add_argument(
        "--pattern",
        default="step_*/**/*.png",
        help="Glob relative to the run folder to locate screenshots (default: step_*/**/*.png)",
    )
    parser.add_argument("--fps", type=float, default=2.0, help="Frames per second for the animation.")
    parser.add_argument("--max-frames", type=int, default=12, help="Maximum frames to include (default: 12).")
    parser.add_argument("--loop", type=int, default=0, help="Number of times the GIF should loop (0 = infinite).")
    return parser.parse_args()


def _collect_frames(run_dir: Path, pattern: str) -> List[Path]:
    frames = sorted(run_dir.glob(pattern))
    return [frame for frame in frames if frame.is_file()]


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run).resolve()
    if not run_dir.exists():
        raise SystemExit(f"Run directory not found: {run_dir}")
    frames = _collect_frames(run_dir, args.pattern)
    if not frames:
        raise SystemExit("No screenshots found for GIF generation")
    trimmed = frames[: args.max_frames]
    images = [imageio.imread(frame) for frame in trimmed]
    duration = 1.0 / max(args.fps, 0.1)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output, images, duration=duration, loop=args.loop if args.loop >= 0 else 0)
    print(f"📸 Generated GIF with {len(trimmed)} frames -> {output}")


if __name__ == "__main__":
    main()
