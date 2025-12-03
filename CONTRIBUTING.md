# Contributing to EIKON Engine

Thanks for helping polish the Strategist + BrowserWorker stack! This document keeps contributions consistent and reviewer-friendly.

## Development environment

1. Create a virtual environment (one time):
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```
2. Install pinned dependencies and Playwright browsers:
   ```bash
   pip install -r requirements.txt
   python -m playwright install
   ```
3. Copy `.env.template` to `.env` and fill only the keys you actually need. Never commit secrets.

## Workflow checklist

- **Linting**: `black . && isort .`
- **Type checking**: `mypy eikon_engine`
- **Unit tests**: `pytest`
- **Smoke demo** (dry-run only): `python run_autonomy_demo.py --summary tmp_run/autonomy.json`
- **Docs/demo refresh**: `./run_quick_demo.sh` (or `./run_quick_demo.ps1`)

CI enforces these steps; matching them locally keeps reviews fast.

## Guidelines

- Prefer small, focused PRs with before/after context or screenshots when UI/docs change.
- Include or update tests whenever behavior changes.
- Keep new dependencies minimal and pin their versions in `pyproject.toml` + `requirements.txt`.
- When touching strategist or planner logic, add notes to `docs/overview.md` or the reviewer guide so YC reviewers understand the change.

## Reporting issues

Open an issue with:
- Expected vs. actual behavior
- Steps to reproduce (goal text, env vars, command run)
- Relevant logs or artifact paths (e.g., `runs/autonomy_demo_20250101/step_003/step.json`)

Thank you for building a YC-ready autonomy stack! 💫
