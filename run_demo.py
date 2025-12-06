from __future__ import annotations

import argparse
import threading
import time
import webbrowser
from pathlib import Path
from typing import Any, Dict, Tuple

import uvicorn

from dashboard.server import app
from run_autonomy_demo import run_single_demo


def run_demo_once(
    goal: str,
    *,
    execute: bool = False,
    allow_sensitive: bool = False,
    summary_path: str | Path | None = None,
    demo_runner=run_single_demo,
) -> Tuple[Dict[str, Any], str]:
    payload = demo_runner(
        goal,
        execute=execute,
        allow_sensitive=allow_sensitive,
        summary_path=summary_path,
    )
    summary = payload.get("summary", {})
    status = "PASS" if summary.get("completed") else "CHECK"
    message = f"{status}: {summary.get('reason', 'pending')}"
    return payload, message


def _launch_dashboard(host: str, port: int) -> None:
    config = uvicorn.Config(app, host=host, port=port, log_level="info")
    server = uvicorn.Server(config)
    server.run()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a one-shot autonomy demo and open the dashboard.")
    parser.add_argument(
        "goal",
        nargs="?",
        default="Showcase the full autonomy workflow",
    )
    parser.add_argument("--execute", action="store_true", help="Use live Playwright instead of dry-run")
    parser.add_argument("--allow-sensitive", action="store_true", help="Allow sensitive workflows")
    parser.add_argument("--no-dashboard", action="store_true", help="Skip launching the dashboard UI")
    parser.add_argument("--port", type=int, default=8000, help="Dashboard port")
    parser.add_argument("--host", default="127.0.0.1", help="Dashboard host")
    args = parser.parse_args()

    payload, message = run_demo_once(
        args.goal,
        execute=args.execute,
        allow_sensitive=args.allow_sensitive,
    )
    print("Autonomy Demo Result")
    print("====================")
    print(message)
    print(f"Summary stored at {payload['summary_path']}")

    if args.no_dashboard:
        return

    server_thread = threading.Thread(target=_launch_dashboard, args=(args.host, args.port), daemon=True)
    server_thread.start()
    time.sleep(1.2)
    dashboard_url = f"http://{args.host}:{args.port}"
    try:
        webbrowser.open(dashboard_url)
    except Exception:
        pass
    print(f"Dashboard available at {dashboard_url}. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Demo complete. Server shutting down.")


if __name__ == "__main__":
    main()
