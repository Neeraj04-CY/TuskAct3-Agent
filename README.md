<div align="center">

# EIKON ENGINE

[![CI](https://github.com/Neeraj04-CY/TustAct3-Agent/actions/workflows/ci.yml/badge.svg)](https://github.com/Neeraj04-CY/TustAct3-Agent/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/badge/docs-live-blue)](docs/index.html)
[![Demo Assets](https://img.shields.io/badge/demo-heroku_sample-success)](docs/artifacts.md)

> Self-growing agent stack that plans, executes, learns new skills, and ships deterministic evidence for every run.

</div>

## Why it matters

- **Deterministic autonomy** – Strategist + Browser Worker pairings run with Playwright dry-runs by default so YC reviewers can replay with zero external access.
- **Artifact-first storytelling** – Every demo emits screenshots, DOM captures, JSON traces, and a markdown summary that publish directly to GitHub Pages.
- **Tool discovery + memory fusion** – Worker feedback loops update the memory store so the agent ranks skills, retries failures, and keeps improving.

Read the architecture notes in `docs/overview.md` and the YC-ready positioning in `docs/yc_pitch.md`.

---

## Demo at a glance

| Asset | Location | Why it matters |
| --- | --- | --- |
| 📺 **Landing page** | `docs/index.html` | Polished cards showcasing each browser step with CTA buttons for JSON + artifacts. |
| 🧠 **Strategist plan** | `docs/json_viewer.html` (loads `docs/artifacts/heroku_sample/result.json`) | Inspect every chain-of-thought action without running code. |
| 📁 **Artifacts bundle** | `docs/artifacts/heroku_sample/` | Screenshots, DOM, logs, and the generated `run_summary.md` for recruiters. |
| 🎥 **Video script** | `docs/quick_video.md` | 90-second narration ready for Loom / YC video. |
| ✅ **Reviewer checklist** | `docs/reviewer_guide.md` | Five-minute inspection instructions for PMs or partners. |

Need more context? `docs/artifacts.md` explains each file, and `docs/demo_quickstart.md` shows how to refresh them.

---

## One-command quickstart

### Windows PowerShell

```powershell
Set-ExecutionPolicy -Scope Process RemoteSigned
./run_quick_demo.ps1
```

### macOS / Linux

```bash
chmod +x run_quick_demo.sh
./run_quick_demo.sh
```

What this does:

1. Creates/activates `.venv`, installs dependencies, and provisions Playwright browsers.
2. Runs `run_autonomy_demo.py` in deterministic dry-run mode (no external network).
3. Copies the latest `artifacts/autonomy_demo_*` folder into `docs/artifacts/heroku_sample/`.
4. Rebuilds the GIF (`docs/assets/demo.gif`) and `docs/artifacts/heroku_sample/run_summary.md` via `scripts/make_demo_gif.py` and `scripts/generate_run_summary.py`.

When it finishes you can open `docs/index.html` locally or push to GitHub Pages for the exact same experience viewers will see.

---

## Manual setup (if you prefer to step through)

```bash
python -m venv .venv && source .venv/bin/activate  # .venv\Scripts\activate on Windows
pip install -r requirements.txt
python -m playwright install
python run_demo_goal.py --goal "Log into the Heroku test site and capture the Secure Area banner" --summary runs/latest_summary.json
```

Copy the resulting `runs/<timestamp>` folder into `docs/artifacts/<slug>` and update `docs/assets/demo.gif` / `docs/json_viewer.html` paths as needed. The helper script `scripts/generate_demo_assets.py` shows the exact copying rules.

---

## Quick Heroku Login Demo (Recording-Friendly)

Need a lightweight walkthrough for screen recordings? Run the direct Playwright driver:

```bash
python scripts/heroku_login_demo.py
```

What happens:

- Launches Chromium headfully with Windows-safe flags and opens the Heroku sample login page.
- Autofills the demo credentials (`tomsmith` / `SuperSecretPassword!`) and waits for the "You logged into a secure area!" banner.
- Captures `artifacts_demo/secure_area.png` so you have a crisp slide-ready screenshot.
- Keeps the browser open until you press Enter in the terminal, making it perfect for recording live narration.

Use this when you only need a fast visual proof. The full autonomy missions/agent pipeline remains available through the CLI (`python -m eikon_engine.missions.cli ...`) for end-to-end demos, rollouts, and artifact generation.

---

## v2: Memory & Skills

- **Mission memory ledger** – every autonomy run now persists a `MissionMemory` JSON file under `memory_logs/`. Entries capture mission text, detected URL, status, artifacts path, and the skills that fired. Nothing is lost between runs.
- **Memory reader hooks Strategist V2** – before planning a new subgoal, the Strategist inspects prior memories (matching URLs or login intents) and surfaces reusable skill hints so the agent never starts cold.
- **Reusable skills** – `LoginFormSkill` lives in `eikon_engine/skills/login.py` and drives Playwright directly. `BrowserWorker.run_skill()` hands the live `page` object plus credentials so skills can manipulate the DOM without rebuilding pipelines.
- **Skill-aware execution** – when a remembered login mission repeats, `MissionExecutor` invokes `login_form_skill` ahead of the subgoal loop and records the reuse inside both the artifacts and the mission memory entry.
- **Proof it works** – rerun any mission twice. The first run writes a memory entry; the second detects the prior success, loads the login skill, and completes faster while logging the reused skill.

This fulfills the v2 mandate: *“TuskAct3 remembers past missions and reuses learned skills instead of starting from scratch.”*

---

## Architecture snapshot

- **Strategist (`src/strategist/`)** – turns natural language goals into structured plans with guardrails.
- **Worker (`src/worker/`)** – executes browser/API/code steps, logs DOM + screenshots, and emits telemetry.
- **Memory (`src/memory/`)** – persists successes/failures so Strategist can re-rank tools.
- **Discovery + Skills (`src/discovery`, `src/skills`)** – package reusable capabilities that can be swapped in or discovered automatically.
- **Workflow (`src/workflow/`)** – orchestrates Strategist ⇄ Worker ⇄ Memory and handles retries.

Peek at `docs/overview.md` for the full research brief plus the roadmap for adaptive tool discovery.

---

## Documentation map

- `docs/demo_quickstart.md` – step-by-step reproduction guide (Windows + macOS/Linux).
- `docs/artifacts.md` – how to inspect `steps.jsonl`, `trace.jsonl`, screenshots, and summaries.
- `docs/quick_video.md` – narration outline for a ≤2 minute video.
- `docs/reviewer_guide.md` – five-minute checklist for YC/recruiters.
- `docs/yc_pitch.md` – written pitch covering problem/solution/traction/ask.

Everything links back to this README so reviewers never get lost.

---

## Stability Loop

- `StabilityMonitor` records reward drift, confidence deltas, repair counts, duration trends, DOM fingerprint similarity, and repeated failures after every autonomy run.
- Each invocation of `run_autonomy_demo.py` now emits `stability_report.json` and `stability_report.md` next to the summary files, plus appends to `artifacts/stability/history.json` for multi-run trend analysis.
- Strategist V2 feeds these metrics into `AgentMemory`, so future plans inherit selector bias, repair pressure, and success-rate context directly from the stability loop.

---

## Autonomous Rollout Engine: Self-Improvement Across Runs

- `run_rollout.py --n 20` executes N autonomy episodes (dry-run by default), reusing the same Strategist/AgentMemory/StabilityMonitor instances so learning compounds between attempts.
- Each episode stores `reward_trace`, confidence medians, repair events, planner evolution, behavior predictions, DOM fingerprints, and the generated `stability_report.*` under `artifacts/rollouts/run_{i}/`.
- After all runs, `rollout_summary.json` + `.md` capture reward trendlines (linear regression), confidence medians, repair-pressure trends, repeated failure clusters, behavior-model deltas, memory growth, stability drift, and pass/fail classifications per run.
- Use rollout summaries in reviewer packets to prove the agent improves autonomously without manual resets.

---

## Self-Improvement Loop

- `eikon_engine/replay/experience_replay.py` replays historical autonomy runs, captures low-confidence failures, and feeds them into `StrategistV2.learn_from_past` without touching the browser. Alternate subgoals are rescored with the reward model and merged back into AgentMemory.
- `eikon_engine/replay/curriculum_builder.py` clusters runs by difficulty spikes, repeated failure clusters, stability drift, and DOM similarity so the most fragile scenarios get replayed first.
- Run the full offline curriculum plus reporting pipeline with:

```bash
python run_offline_eval.py --artifacts artifacts --output artifacts/replay --save-hints
```

- Need a smaller pass? Limit history and skip persistence:

```bash
python run_offline_eval.py --limit 3
```

- Every replay cycle emits `replay_summary.json`, per-batch reports, and `improvement_report.json/md` that summarize selector bias shifts, subgoal merges, and skill metrics. When `--save-hints` is set the newly learned selectors/subgoals land in `memory_hints.json` for the next live Strategist run.

---

## Showcase Dashboard & Release Bundle

- `python run_demo.py` runs a single autonomy episode, prints a human-readable verdict, spins up the FastAPI dashboard (`dashboard/server.py`), and opens it in your browser. Pass `--no-dashboard` if you only need artifacts.
- Dashboard views: last-run summary, reward/confidence charts, repair timeline, planner evolution log, stability drift history, repeated failure clusters, memory growth, behavior predictions, plus a DOM + screenshot viewer for every step. All data comes from `artifacts/autonomy/run_*` so you can refresh without restarts.
- `python build_release.py --rollout 3` reruns the fast autonomy demo, optionally triggers a short rollout, copies everything into `release_bundle/release_<timestamp>/`, generates `demo.gif`, `charts.html`, dashboard snapshots, and exports `README.md` + `docs/yc_pitch.md` to both Markdown and PDF.
- Ship the `release_bundle/` folder to YC reviewers so they can inspect artifacts, metrics, and videos without reproducing the run locally.

---

## API Usage

- Launch the service with `uvicorn api_server:app --reload`. Endpoints include `/run` (full autonomy demo), `/plan`, `/predict`, `/last_run`, and `/artifacts/{path}`.
- **curl example**:

```bash
curl -X POST http://localhost:8000/run \
	-H "Content-Type: application/json" \
	-d '{"goal":"Generate dashboard summary","execute":false}'
```

- **Python example**:

```python
import requests

resp = requests.post("http://localhost:8000/run", json={"goal": "Demo"})
resp.raise_for_status()
summary = resp.json()["summary"]
print("Status:", summary["reason"])
```

- `/plan` returns the raw planner targets, `/predict` calls the BehaviorLearner for a DOM fingerprint, and `/artifacts/...` streams run artifacts for UI embeddings.

---

## Skill Plugin Overview

- Skills now inherit from the async `Skill` base (`eikon_engine/skills/base.py`) and expose a single `execute(context)` coroutine. Context always includes the Playwright `page` plus mission-provided parameters (like credentials).
- `LoginFormSkill` is the first reusable capability; `FormFillSkill` and `ExtractSkill` remain available for compatibility and return structured status payloads.
- The lightweight `eikon_engine/skills/registry.py` maps names to skill instances and powers both the legacy `SkillRegistry` shim and the new memory-aware planner hints.
- `BrowserWorker.run_skill()` shares its live session so skills can drive the page headlessly or headfully without extra plumbing.

---

## Testing & CI

```bash
pytest
```

GitHub Actions (`.github/workflows/ci.yml`) runs lint + tests on every push and blocks merges if the deterministic demo summary goes stale.

---

## Contributing

1. `scripts/setup_dev.sh` (or the PowerShell commands in `docs/demo_quickstart.md`) bootstraps dependencies.
2. Open an issue describing the improvement.
3. Submit a PR with screenshots of new artifacts + updated summaries.

See `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, and `AUTHORS.md` for the canonical guidelines.

---

## License

Apache-2.0 – see `LICENSE`.
